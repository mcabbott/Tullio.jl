module Tullio

export @tullio, @moltullio

using MacroTools, GPUifyLoops
using Base.Cartesian

"""
    @tullio A[i,j] := B[i] * log(C[j])
    @moltullio A[i,j] := B[i] * log(C[j])

This loops over all indices, but unlike `@einsum`/`@vielsum` the idea is to experiment with
map/generator expressions, GPUifyLoops.jl, and loop unrolling...

    @tullio A[i] := B[i,j] * C[j]
    @tullio A[i] := B[i,j] * C[j]  (+, j<=10)
    @tullio A[] := B[i,j] * C[j]  (+, i, unroll, j<=10)

Reduction by summing over `j`, but with all the same things.
"""
macro tullio(exs...)
    _tullio(exs...)
end

macro moltullio(exs...)
    _tullio(exs...; multi=true)
end

function _tullio(leftright, after=nothing; multi=false)
    store = (axes=Dict(), flags=Set(), checks=[], rightind=[],
        redop=Ref(:+), loop=[], unroll=[], rolln=Ref(0), init=Ref{Any}(:(zero(T))),)

    #===== parse input =====#

    newarray = @capture(leftright, left_ := right_ )
    newarray ||  @capture(leftright, left_ = right_ ) || error("wtf?")

    @capture(left, Z_[leftind__]) || error("can't understand LHS")

    readred(after, store)
    MacroTools.postwalk(sizewalk(store), leftright)

    #===== start massaging RHS =====#

    right2 = MacroTools.postwalk(indexwalk(leftind, :φ), right)

    redind = setdiff(store.rightind, leftind)

    isempty(store.loop) && isempty(store.unroll) ?
        append!(store.loop, redind) :
        @assert sort(redind) == sort(vcat(store.loop, store.unroll)) "if you give any reduction indices, you must give them all"

    nφ = length(leftind)
    nγ = length(redind)

    #===== preliminary expressions =====#

    outex = quote end

    append!(outex.args, store.checks)

    if newarray
        Alist = []
        right1 = MacroTools.postwalk(typewalk(Alist), right)
        unique!(Alist)
        push!(outex.args, :( local h($(Alist...)) = $right1 ))
        push!(outex.args, :( local T = typeof(h($(Alist...))) ))
    else
         push!(outex.args, :( local T = eltype($Z) ))
    end

    if isempty(redind)
         #===== simple function f(φ_...) =====#

         push!(outex.args, :( local @inline $(ifunc(:f,nφ,:φ)) = begin @inbounds $right2 end))
    else
        #===== reduction function f(φ_...) = sum(g(φ_...)) =====#

        right3 = MacroTools.postwalk(indexwalk(redind, :γ), right2)
        push!(outex.args, :( local $(ifunc(:g,nφ,:φ, nγ,:γ)) = begin @inbounds $right3 end))

        ax = map(i -> store.axes[i], redind)
        push!(outex.args, :( local @inline $(ifunc(:f,nφ,:φ)) = $(sumloops(ax, nφ, store)) ))
    end

    #===== final loops =====#

    if newarray
        Zsize = map(i -> :(length($(store.axes[i]))), leftind)
        push!(outex.args, :( $Z = Array{T,$nφ}(undef, $(Zsize...)) ))
    end
    push!(outex.args, outwrite(Z, length(leftind)) )
    push!(outex.args, Z)

    # return esc(MacroTools.prewalk(unblock, outex))
    esc(outex)
end

#===== ingestion =====#

typewalk(list) = ex -> begin
        @capture(ex, A_[inds__]) || return ex
        push!(list, A)
        :( $A[$(map(i->1,inds)...)] )
    end

indexwalk(leftind, sy) = ex -> begin
        @capture(ex, A_[inds__]) || return ex
        indsI = map(indexreplace(leftind, sy), inds)
        :( $A[$(indsI...)] )
    end

indexreplace(leftind, sy) = i -> begin
        d = findfirst(isequal(i),leftind)
        isnothing(d) ? i : Symbol(sy,:_,d)
    end

sizewalk(store) = ex -> begin
        @capture(ex, A_[inds__]) && saveaxes(A, inds, store, true)
        ex
    end

saveaxes(A, inds, store, onright) =
    for (d,i) in enumerate(inds)
        i isa Symbol || continue
        if haskey(store.axes, i)
            str = "range of index $i must agree"
            push!(store.checks, :( @assert $(store.axes[i]) == axes($A,$d) $str ))
        else
            store.axes[i] = :( axes($A,$d) )
        end
        if onright
            push!(store.rightind, i)
        end
    end

readred(ex, store) =
    if @capture(ex, (op_, inds__,)) ||  @capture(ex, op_Symbol)
        store.redop[] = op
        store.init[] = op == :* ? :(one(T)) :
            op == :max ? :(typemin(T)) :
            op == :min ? :(typemin(T)) :
            :(zero(T))
        foreach(savered(store), something(inds,[]))
    end

savered(store) = i ->
    if i == :unroll
        push!(store.flags, :unroll)
    elseif @capture(i, unroll(n_Int))
        push!(store.flags, :unroll)
        store.rolln[] = n

    elseif i isa Symbol
        unrollpush(store, i)
    elseif @capture(i, j_ <= m_)
        unrollpush(store, j)
        store.axes[j] = :( Base.OneTo($m) )
    elseif @capture(i, n_ <= j_ <= m_)
        unrollpush(store, j)
        store.axes[j] = :( $n:$m )

    elseif  @capture(i, init = z_)
        store.init[] = z==0 ? :(zero(T)) : z==1 ? :(one(T)) : z
    else
        @warn "wtf is index $i"
    end

unrollpush(store, i) = (:unroll in store.flags) ? push!(store.unroll, i) : push!(store.loop, i)

#===== digestion =====#

outwrite(Z, n) =
    macroexpand(@__MODULE__, quote
        @inbounds( @nloops $n φ $Z  begin
            (@nref $n $Z φ) = (@ncall $n f φ)
        end )
    end)

sumloops(ax, nφ, store, nγ=length(ax)) =
    macroexpand(@__MODULE__, quote
        local σ = $(store.init[])
        @nloops $nγ γ d->$ax[d] begin
            σ = $(store.redop[])(σ, $(ifunc(:g,nφ,:φ, nγ,:γ)))
        end
        σ
    end)

ilist(n, sy=:φ) = map(d -> Symbol(sy,:_,d), 1:n)
ifunc(f, n1, sy1=:φ, n2=0, sy2=:γ) = :( $f($(ilist(n1,sy1)...), $(ilist(n2,sy2)...)) )

#===== action =====#

struct UndefArray{T,N} end

UndefArray{T,N}(ax...) where {T,N} = Array{T,N}(undef, map(length, ax)...)


#= === TODO ===

* GPUifyloops -- first for unrolling of sums?
* threads, easy

* index shifts including reverse should be easy to allow, adjust the ranges... but not yet.
* constant indices A[i,3,$k] will be easy

=#
end # module
