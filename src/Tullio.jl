module Tullio

export @tullio, @moltullio

using MacroTools, GPUifyLoops, TiledIteration

const UNROLLOK = VERSION >= v"1.2.0-DEV.462"

"""
    @tullio A[i,j] := B[i] * log(C[j])
    @moltullio A[i,j] := B[i] * log(C[j])

This loops over all indices, just like `@einsum`/`@vielsum`.
But idea is to experiment with various things...

    @tullio A[i] := B[i,j] * C[k]                            # implicit sum over j
    @tullio A[i] := B[i,j] * C[k]  (+, j,k)                  # explicit sum, must give all indices
    @tullio A[i] := B[i,j] * C[k]  (+, j, unroll(4), k)      # unrolled by 4 only k loop, innermost
    @tullio A[i] := B[i,j] * C[k]  (+, unroll, j<=10, k<=10) # completely unroll j and k

Reduction by summing over `j`, but with all the same things.
If given, the order of the summed indices is the order of the loops, i.e. `k` changes fastest.
Unrolling uses `GPUifyLoops.jl` which only works on Julia 1.2.

    @tullio A[i,j] := B[i] * log(C[j])       {tile, i,j}
    @tullio A[i] := B[i,j] * C[k]  (+, j,k)  {tile(256), i,j,k}

This adds an outermost loop over tiles from TiledIteration.jl

    @tullio A[i,j] := B[i] * log(C[j])       {static, i<=10,j<=10}

Not yet, but perhaps this should unroll the LHS loops too.

    @tullio A[i,j] := exp(B.data[i]) / C[j].field[k]

The aim is to allow about this level of complexity.
"""
macro tullio(exs...)
    _tullio(exs...)
end

macro moltullio(exs...)
    _tullio(exs...; multi=true)
end

function _tullio(leftright, after1=nothing, after2=nothing; multi=false)

    @capture(leftright, left_ += right_ ) &&
        return _tullio(:($left = $left + $right); after=after, multi=multi)
    @capture(leftright, left_ -= right_ ) &&
        return _tullio(:($left = $left - ($right) ); after=after, multi=multi)
    @capture(leftright, left_ *= right_ ) &&
        return _tullio(:($left = $left * ($right) ); after=after, multi=multi)

    leftright = MacroTools.postwalk(primewalk, leftright)

    #===== parse input =====#

    store = (axes=Dict(), flags=Set(), checks=[], arrays=[], rightind=[],
        redop=Ref(:+), loop=[], unroll=[], rolln=Ref(0), init=Ref{Any}(:(zero(T))),
        tile=[], tilesize=Ref(512))

    newarray = @capture(leftright, left_ := right_ )
    newarray || @capture(leftright, left_ = right_ ) || error("wtf?")

    @capture(left, Z_[leftind__] | [leftind__] ) || error("can't understand LHS")
    isnothing(Z) && @gensym Z

    if @capture(after1, (stuff__,))
        readred(after1, store)
        tiles = readtiles(after2, store)
    else
        tiles = readtiles(after1, store)
    end

    MacroTools.postwalk(rightwalk(store), right)
    unique!(store.arrays)
    unique!(store.rightind)
    redind = setdiff(store.rightind, leftind)

    isempty(store.loop) && isempty(store.unroll) ?
        append!(store.loop, redind) :
        @assert sort(redind) == sort(vcat(store.loop, store.unroll)) "if you give any reduction indices, you must give them all"

    isempty(store.unroll) || UNROLLOK ||
        @warn "can't unroll loops on Julia $VERSION" maxlog=1 _id=hash(leftright)

    #===== preliminary expressions =====#

    outex = quote end

    rhsfunc = :( rhs($(store.rightind...), $(store.arrays...)) )
    push!(outex.args, :( local @inline $rhsfunc = begin @inbounds $right end))

    if newarray
        allfirst = map(i -> :(first($(store.axes[i]))), store.rightind)
        push!(outex.args, :( local T = typeof(rhs($(allfirst...), $(store.arrays...))) ))
        Zsize = map(i -> :(length($(store.axes[i]))), leftind) # before tiles!
    else
        MacroTools.postwalk(leftwalk(store), right)
        push!(outex.args, :( local T = eltype($Z) )) # before checks
    end

    append!(outex.args, store.checks)

    #===== tiles =====#

    if tiles
        nt = length(store.tile)
        tsz = trunc(Int, (store.tilesize[])^(1/nt))
        ttup = ntuple(_ -> tsz, nt)

        taxes = map(i -> store.axes[i], store.tile)
        ex = :( local tiles = collect(TileIterator(($(taxes...),), $ttup)) )
        push!(outex.args, ex)

        for (n,i) in enumerate(store.tile)
            store.axes[i] = :( tile[$n] )
        end
    end

    if isempty(redind)
        #===== no reduction =====#

         rex = :( $Z[$(leftind...)] = $rhsfunc )
    else
        #===== reduction =====#

        if multi
            push!(outex.args, :(local cache = Vector{T}(undef, Threads.nthreads()) ))
            σ, cache = :( cache[Threads.threadid()] ), [:cache]
        else
            σ, cache = :σ, []
        end

        ex = :( $σ = $(store.redop[])($σ, $rhsfunc ) )
        ex = recurseloops(ex, store)
        ex = :( $σ = $(store.init[]); $ex; ) #return $σ; )

        rex = :( $ex; $Z[$(leftind...)] = $σ )
    end

    #===== final loops =====#

    if newarray
        push!(outex.args, :( $Z = Array{T,$(length(leftind))}(undef, $(Zsize...)) ))
    end

    ministore = (axes=store.axes, loop=reverse(leftind), unroll=[], rolln=Ref(0))
    ex = recurseloops( :( @inbounds $rex ), ministore)

    if tiles
        ex = :( for tile in tiles; $ex; end )
    end

    if multi
        push!(outex.args, :( Threads.@threads $ex ))
    else
        push!(outex.args, ex)
    end

    #===== done! =====#

    push!(outex.args, Z)
    esc(outex)
end

#===== ingestion =====#

primewalk(ex) = begin
    @capture(ex, A_[inds__]) || return ex
    map!(i -> @capture(i, j_') ? Symbol(j,'′') : i, inds, inds)
    return :( $A[$(inds...)] )
end

rightwalk(store) = ex -> begin
        @capture(ex, A_[inds__]) || return ex
        push!(store.arrays, A)
        append!(store.rightind, filter(i -> i isa Symbol, inds))
        saveaxes(A, inds, store)
    end

leftwalk(store) = ex -> begin
        @capture(ex, A_[inds__]) || return ex
        saveaxes(A, inds, store)
    end

saveaxes(A, inds, store) =
    for (d,i) in enumerate(inds)
        i isa Symbol || continue
        if haskey(store.axes, i)
            str = "range of index $i must agree"
            push!(store.checks, :( @assert $(store.axes[i]) == axes($A,$d) $str ))
        else
            store.axes[i] = :( axes($A,$d) )
        end
    end

readred(ex::Nothing, store) = nothing
readred(ex, store) =
    if @capture(ex, (op_Symbol, inds__,)) ||  @capture(ex, op_Symbol)
        store.redop[] = op
        store.init[] = op == :* ? :(one(T)) :
            op == :max ? :(typemin(T)) :
            op == :min ? :(typemin(T)) :
            :(zero(T))
        foreach(savered(store), something(inds,[]))
    else
        error("expected something like (+,i,j,unroll,k) but got $ex")
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

readtiles(ex::Nothing, store) = false
readtiles(ex, store) =
    if @capture(ex, {op_, inds__})
        if @capture(op, tile(n_) )
            store.tilesize[] = n
        elseif op != :tile
            error("expected something like {tile(128),i,j} but got $ex")
        end
        append!(store.tile, inds)
        return true
    else
        error("expected something like {tile(128),i,j} but got $ex")
    end

#===== digestion =====#

recurseloops(ex, store) =
    if !isempty(store.unroll)
        i = pop!(store.unroll)

        r = store.axes[i]
        ex = iszero(store.rolln[]) ?
            :(@unroll for $i in $r; $ex; end) :
            :(@unroll $(store.rolln[]) for $i in $r; $ex; end)
        return recurseloops(ex, store)

    elseif !isempty(store.loop)
        i = pop!(store.loop)
        r = store.axes[i]
        ex = :(for $i in $r; $ex; end)
        return recurseloops(ex, store)

    else
        return ex
    end

#===== action =====#

struct UndefArray{T,N} end

UndefArray{T,N}(ax...) where {T,N} = Array{T,N}(undef, map(length, ax)...)


#= === TODO ===

* index shifts including reverse should be easy to allow, adjust the ranges... but not yet.
* constant indices A[i,3,$k] will be easy

=#
end # module
