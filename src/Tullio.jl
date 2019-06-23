module Tullio

export @tullio, @moltullio

using MacroTools, GPUifyLoops
using Base.Cartesian

"""
    @tullio A[i,j] := B[i] * log(C[j])
    @moltullio A[i,j] := B[i] * log(C[j])

This loops over all indices, but unlike `@einsum`/`@vielsum` the idea is to experiment
map/generator expressions, GPUifyLoops.jl, and loop unrolling...

    @tullio A[i] := B[i,j] * C[j]
    @tullio A[i] := B[i,j] * C[j]  (+, j<=10)
    @tullio A[] := B[i,j] * C[j]  (+, i, unroll, j<=10)

    @moltullio A[i] := B[i,j] * C[j] (+)       # threaded

Reduction by summing over `j`, but with all the same things.
"""
macro tullio(exs...)
    _tullio(exs...)
end

macro moltullio(exs...; multi=true)
    _tullio(exs...)
end

function _tullio(leftright, after=nothing; multi=false)
    store = (axes=Dict(), flags=Set(), checks=[], rightind=[], redop=Ref(:+), loop=[], unroll=[])

    #===== parse input =====#
    @capture(leftright, (left_ := right_) | (left_ = right_) ) || error("wtf?")
    @capture(left, Z_[leftind__]) || error("can't understand LHS")

    readred(after, store)
    MacroTools.postwalk(sizewalk(store), right)

    #===== start on RHS =====#
    right2 = MacroTools.postwalk(indexwalk(leftind, :LI), right)

    redind = setdiff(store.rightind, leftind)
    isempty(store.loop) && isempty(store.unroll) ?
        append!(store.loop, redind) :
        @assert sort(redind) == sort(vcat(store.redind, store.unroll)) "if you give any reduction indices, you must give them all"

    #===== output shape =====#
    outex = quote end
    inplace = @capture(leftright, left_ = right_ )
    if inplace
        push!(outex.args, :( ZI = CartesianIndices($Z) ))
    else
        Zaxes = map(i -> store.axes[i], leftind)
        push!(outex.args, :( ZI = CartesianIndices(($(Zaxes...),)) ))
    end

    if isempty(redind)
         #===== simple function =====#
         push!(outex.args, :( @inline f(LI) = @inbounds $right2 ))
    else
        #===== reduction function =====#
        right3 = MacroTools.postwalk(indexwalk(redind, :RI), right2)
        ax = map(i -> store.axes[i], redind)
        redaxes = :( CartesianIndices(($(ax...),)) )
        right4 = :( sum($right3 for RI in SI )  )

        push!(outex.args, :( SI = $redaxes ))
        push!(outex.args, :( f(LI) = $right4 ))
    end

    #===== final map =====#
    if inplace
        push!(outex.args, :( map!(f, $Z, ZI) ))
    else
        push!(outex.args, :( $Z = map(f, ZI) ))
        # push!(outex.args, :( $Z = [f(I) for I in  ZI] ))
    end

    return esc(outex)
end

#===== ingestion =====#

indexwalk(leftind, CI) = ex -> begin
        @capture(ex, A_[inds__]) || return ex
        indsI = map(indexreplace(leftind, CI), inds)
        :( $A[$(indsI...)] )
    end

indexreplace(leftind, CI) = i -> begin
        d = findfirst(isequal(i),leftind)
        isnothing(d) ? i : :( $CI.I[$d] )
    end

sizewalk(store) = ex -> begin
        @capture(ex, A_[inds__]) && saveaxes(A, inds, store, true)
        return ex
    end

saveaxes(A, inds, store, onright) =
    for (d,i) in enumerate(inds)
        if haskey(store.axes, i)
            push!(store.checks, :( @assert $(store.axes[i]) == axes($A,$d) ) )
        else
            store.axes[i] = :( axes($A,$d) )
        end
        if onright
            push!(store.rightind, i)
        end
    end

function readred(ex, store)
    @capture(ex, (op_, inds__,) ) || return
    store.redop[] = op
    foreach(savered(store), inds)
end

savered(store) = i ->
    if i == :unroll
        push!(store.flags, :unroll)
    elseif i isa Symbol
        unrollpush(store, i)
    elseif @capture(i, j_ <= m_)
        unrollpush(store, i)
        store.axes[j] = :( Base.OneTo($m) ) # store something more literal? save sizes elsewhere?
    elseif @capture(i, n_ <= j_ <= m_)
        unrollpush(store, i)
        store.axes[j] = :( $n:$m )
    end

unrollpush(store, i) = (:unroll in store.flags) ? push!(store.unroll, i) : push!(store.loop, i)

#===== digestion =====#

# outnew(Z, store) = zeros()

outwrite(Z, store) = :( map!(f, $Z, ZI) )



#= === TODO ===

* GPUifylooos -- only in the reduction?
* unrolling of sums?

* index shifts including reverse should be easy to allow, adjust the ranges... but not yet.
* threadmap will be trivial, can also wait
* creation of new arrays: by map() to keep it generic? map(i->1, CartesianIndices((1:3, 3:4)))
* checks aren't used, but that's easy
* constant indices A[i,3,$k] will be easy

=#
end # module
