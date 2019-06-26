module Tullio

export @tullio, @moltullio

using MacroTools, GPUifyLoops, TiledIteration

const UNROLLOK = VERSION >= v"1.2.0-DEV.462"
const TILESIZE = 1024

"""
    @tullio A[i,j] := B[i] * C[j]
    @tullio A[i,j] := B[i,j] * log(C[j]) + D[i]

Index notation macro, a lot like `@einsum`. This package exists to experiment
with various additions.
Like `@einsum` it's not strictly Einstein: it sums the entire expression
over any indices not appearing on the left, and allows almost any functions.

    @tullio A[i] := B[i,j] * C[k]                            # implicit sum over j & k
    @tullio A[i] := B[i,j] * C[k]  (+, j,k)                  # explicit sum, must give all indices
    @tullio A[i] := B[i,j] * C[k]  (+, j, unroll(4), k)      # unroll k loop by 4, innermost
    @tullio A[i] := B[i,j] * C[k]  (+, unroll, j<=10, k<=10) # completely unroll j and k

Reduction by summing over `j` and `k`, controlled by things in `()`.
If given, the order of the summed indices is the order of the loops, i.e. `k` changes fastest.
The reduction operation is always first,
and with something other than `(+)` or `(*)`, you may provide `(f, init=1)`.

Unrolling uses `GPUifyLoops.jl` which only works on Julia 1.2.

    @tullio A[i,j] := B[j,i] / C[j,i]  (+,k)  {j,i}          # loops with i innermost, default
    @tullio A[i,j] := B[i] * log(C[i,j]/B[j]) {thread}       # multithreading over outermost, j
    @tullio A[i,j] := B[j,i] * C[k]           {tile, thread} # multithreading over tiles for i,j
    @tullio A[i] := B[i,j] * C[k]  (+, j,k)   {tile(2^10), i,j,k}

Loops over indices on the left are controlled by `{}`.
Default order for `A[i,j]` is {j,i} meaning `for j ∈..., i ∈...` i.e. `i` changes fastest.
Magic word `thread` (or `threads`) adds `Threads.@threads` to outermost loop.

Magic word `tile` adds an outer loop over tiles from `TiledIteration.jl`.
This can include indices being reduced over, if given explicitly.
The given size is the (maximum) product of tile dimensions. Default is `$TILESIZE`.

    @tullio A[i,_,j] := exp(B.data[i]) / C[j].vec[k] + f(D)[i,J[j],4] (+)

This shows the level of complexity that is allowed. In particular constant indices
are fine (`_` means `1`, and only `1` on the left with `:=`), arrays-of-arrays are fine
on the right, as are arrays indexed by other arrays.
Functions like `f(D)` will be called on every iteration.
The eltype of `A` comes from `T = typeof(rhs(1,1,1, B,C,J,D))`.
"""
macro tullio(exs...)
    _tullio(exs...; mod=__module__)
end

macro moltullio(exs...)
    _tullio(exs...; multi=true, mod=__module__)
end

function _tullio(leftright, after1=nothing, after2=nothing; multi=false, mod=Main)

    @capture(leftright, left_ += right_ ) &&
        return _tullio(:($left = $left + $right); after1=after1, after2=after2, multi=multi)
    @capture(leftright, left_ -= right_ ) &&
        return _tullio(:($left = $left - ($right) ); after1=after1, after2=after2, multi=multi)
    @capture(leftright, left_ *= right_ ) &&
        return _tullio(:($left = $left * ($right) ); after1=after1, after2=after2, multi=multi)

    leftright = MacroTools.postwalk(primewalk, leftright)

    @gensym Tsym Ssym Csym # output elType, Scalar accumulator, per-thread Cache
    @gensym Fsym Gsym Isym # Function, tile Grid, tile Index

    #===== parse input =====#

    store = (axes=Dict{Any,Any}(1 => 1), flags=Set(), checks=[], arrays=[], rightind=[],
        redop=Ref(:+), loop=[], unroll=[], rolln=Ref(0), init=Ref{Any}(:(zero($Tsym))),
        curly=[], tilesize=Ref(0), multi=Ref(multi))

    newarray = @capture(leftright, left_ := right_ )
    newarray || @capture(leftright, left_ = right_ ) || error("wtf?")

    @capture(left, Z_[leftind__] | [leftind__] ) ||
        error("can't understand LHS, expected A[i,j,k], got $left")
    isnothing(Z) && @gensym Z

    if @capture(after1, {stuff__} )
        readcurly(after1, store)
    else
        readred(after1, store, Tsym)
        readcurly(after2, store)
    end

    MacroTools.postwalk(rightwalk(store), right)
    unique!(store.arrays)
    unique!(store.rightind)
    redind = setdiff(store.rightind, leftind)

    isempty(store.loop) && isempty(store.unroll) ?
        append!(store.loop, redind) :
        sort(redind) == sort(vcat(store.loop, store.unroll)) ||
        error("if you give any reduction indices, you must give them all: $(Tuple(redind))")

    isempty(store.unroll) || UNROLLOK ||
        @warn "can't unroll loops on Julia $VERSION" maxlog=1 _id=hash(leftright)

    store.tilesize[] == 0 ||
        isempty(intersect(store.unroll, store.curly)) ||
        error("can't unroll and tile the same index")

    #===== preliminary expressions =====#

    outex = quote end

    funwithargs = :( $Fsym($(store.rightind...), $(store.arrays...)) )
    push!(outex.args, :( local $funwithargs = @inbounds $right ))

    if newarray
        allfirst = map(i -> :(first($(store.axes[i]))), store.rightind)
        push!(outex.args, :( local $Tsym = typeof($Fsym($(allfirst...), $(store.arrays...))) ))
        outsize = map(i -> :(length($(store.axes[i]))), leftind) # before tiles!
    else
        MacroTools.postwalk(leftwalk(store), left) # before checks!
        push!(outex.args, :( local $Tsym = eltype($Z) ))
    end
    push!(outex.args, :( $Tsym == Any && @warn "eltype is Any, sorry" maxlog=1 _id=$(hash(right))))

    append!(outex.args, store.checks)

    #===== tiles =====#

    if store.tilesize[] > 0
        length(store.curly) == 0 && append!(store.curly, reverse(filter(i -> i isa Symbol, leftind)))
        nt = length(store.curly)
        tsz = trunc(Int, (store.tilesize[])^(1/nt))
        ttup = ntuple(_ -> tsz, nt)

        taxes = map(i -> store.axes[i], store.curly)
        ex = :( local $Gsym = collect(Tullio.TiledIteration.TileIterator(($(taxes...),), $ttup)) )
        push!(outex.args, ex)

        for (n,i) in enumerate(store.curly)
            store.axes[i] = :( $Isym[$n] )
        end
    end

    if isempty(redind)
        #===== no reduction =====#

         rex = :( $Z[$(leftind...)] = $funwithargs ) # note that leftind may include constants
    else
        #===== reduction =====#

        if store.multi[]
            push!(outex.args, :(local $Csym = Vector{$Tsym}(undef, Threads.nthreads()) ))
            σ = :( $Csym[Threads.threadid()] )
        else
            σ = Ssym
        end

        ex = :( $σ = $(store.redop[])($σ, $funwithargs ) )
        ex = recurseloops(ex, store)
        ex = :( $σ = $(store.init[]); $ex; ) #return $σ; )

        rex = :( $ex; $Z[$(leftind...)] = $σ )
    end

    #===== final loops =====#

    if newarray
        push!(outex.args, :( $Z = Array{$Tsym,$(length(leftind))}(undef, $(outsize...)) ))
    end

    leftind = reverse(filter(i -> i isa Symbol, leftind)) # default set
    leftcurly = intersect(store.curly, leftind)           # take order from {}, may be super/subset
    loopind = vcat(leftcurly, setdiff(leftind, leftcurly))
    ministore = (axes=store.axes, loop=loopind, unroll=[], rolln=Ref(0))

    ex = recurseloops( :( @inbounds $rex ), ministore)

    if store.tilesize[] > 0
        ex = :( for $Isym in $Gsym; $ex; end )
    end

    if isempty(loopind) # scalar output needs a let block in global scope
        push!(outex.args, :( let; $ex; end ))
    elseif store.multi[]
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
    map!(inds, inds) do i
        @capture(i, j_') ? Symbol(j,'′') :  # normalise i' to i′
        i == :_ ? 1 : i                     # and A[i,_] to A[i,1]
    end
    return :( $A[$(inds...)] )
end

rightwalk(store) = ex -> begin
        @capture(ex, A_[inds__]) || return ex
        push!(store.arrays, arrayonly(A))
        append!(store.rightind, filter(i -> i isa Symbol, inds)) # skips constant indices
        saveaxes(arrayfirst(A), inds, store)
        return ex
    end

leftwalk(store) = ex -> begin
        @capture(ex, A_[inds__]) || return ex
        saveaxes(A, inds, store)
    end

saveaxes(A, inds, store) = begin
    for (d,i) in enumerate(inds)
        i isa Symbol || continue
        if haskey(store.axes, i)
            str = "range of index $i must agree"
            push!(store.checks, :( @assert $(store.axes[i]) == axes($A,$d) $str ))
        else
            store.axes[i] = :( axes($A,$d) )
        end
    end
    n = length(inds)
    str = "expected a $n-array $A" # already arrayfirst(A)
    push!(store.checks, :( @assert ndims($A) == $n $str ))
end

arrayonly(A::Symbol) = A   # this is for rhs(i,j,k, A,B,C)
arrayonly(A::Expr) =
    if @capture(A, B_[inds__].field_) || @capture(A, B_[inds__])
        return B
    elseif @capture(A, B_.field_) || @capture(A, f_(B_) )
        return B
    end

arrayfirst(A::Symbol) = A  # this is for axes(A,d), axes(first(B),d), etc.
arrayfirst(A::Expr) =
    if @capture(A, B_[inds__].field_)
        return :( first($B).$field )
    elseif @capture(A, B_[inds__])
        return :( first($B) )
    elseif @capture(A, B_.field_)
        return A
    elseif @capture(A, f_(B_) )
        return A
    end

readred(ex::Nothing, store, Tsym) = nothing
readred(ex, store, Tsym) =
    if @capture(ex, (op_Symbol, inds__,)) ||  @capture(ex, op_Symbol)
        store.redop[] = op
        store.init[] = op == :* ? :(one($Tsym)) :
            op == :max ? :(typemin($Tsym)) :
            op == :min ? :(typemin($Tsym)) :
            :(zero($Tsym))
        foreach(savered(store, Tsym), something(inds,[]))
    else
        error("expected something like (+,i,j,unroll,k) but got $ex")
    end

savered(store, Tsym) = i ->
    if i == :unroll
        push!(store.flags, :unroll)  # flag used only by unrollpush
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
        store.init[] = z==0 ? :(zero($Tsym)) : z==1 ? :(one($Tsym)) : z
    else
        error("don't know what to do with index $i")
    end

unrollpush(store, i) = (:unroll in store.flags) ? push!(store.unroll, i) : push!(store.loop, i)

readcurly(ex::Nothing, store) = nothing
readcurly(ex, store) = @capture(ex, {stuff__}) ?
    foreach(savecurly(store), stuff) :
    error("expected something like {tile(2^10),i,j,k} but got $ex")

savecurly(store) = i ->
    if @capture(i, tile(n_) | tiles(n_) )
        store.tilesize[] = n
    elseif i==:tile || i==:tiles
        store.tilesize = TILESIZE
    elseif i==:thread || i==:threads
        store.multi[] = true

    elseif i isa Symbol
        push!(store.curly, i)
    elseif @capture(i, j_ <= m_)
        push!(store.curly, j)
        store.axes[j] = :( Base.OneTo($m) )
    elseif @capture(i, n_ <= j_ <= m_)
        push!(store.curly, j)
        store.axes[j] = :( $n:$m )

    else
        error("don't know what to do with index $i")
    end

#===== digestion =====#

recurseloops(ex, store) =
    if !isempty(store.unroll)
        i = pop!(store.unroll)

        r = store.axes[i]
        ex = iszero(store.rolln[]) ?
            :(Tullio.GPUifyLoops.@unroll for $i in $r; $ex; end) :
            :(Tullio.GPUifyLoops.@unroll $(store.rolln[]) for $i in $r; $ex; end)
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

* actual GPU loops, {gpu}... if limited to 3 loops, and in-place, then perhaps not too hard

* allow unrolling of LHS indices too? Then {static}...

* index shifts including reverse should be easy to allow, adjust the ranges... but not yet.
* constant indices A[i,$k] will be easy
* option {zero} for A[i,i] := ...
* allow {zygote} which wraps the whole thing in a function & adds @adjoint?
  could even be default based on isdefined...

=#
end # module
