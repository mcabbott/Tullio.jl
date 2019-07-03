module Tullio

export @tullio, @moltullio

using MacroTools, GPUifyLoops, TiledIteration

const UNROLLOK = VERSION >= v"1.2.0-DEV.462"
const TILESIZE = 1024

"""
    @tullio A[i,j] := B[i] * C[j]                            # outer product
    @tullio A[i,j] = B[i,j] * log(C[j]) + D[i]               # A .= B .* log.(C') .+ D

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
    @tullio A[i] := B[i,j] * C[k]  (+, j,k)   {tile(8^3), i,j,k}

Loops over indices on the left are controlled by `{}`.
Default order for `A[i,j]` is {j,i} meaning `for j ∈..., i ∈...` i.e. `i` changes fastest.
Magic word `{thread}` (or `threads`) adds `Threads.@threads` to outermost loop.

Magic word `{tile}` adds an outer loop over tiles from `TiledIteration.jl`.
The given size is the (maximum) product of tile dimensions. Default is `$TILESIZE`.
You can tile over reduction indices, if you give them explicitly, but this is not thread safe.

    @tullio A[i,_,j] := exp(B.data[i]) / C[j].vec[k] + f(D)[i,j,4] (+)
    @tullio A[i,i,k] = B[J[i], J[i], K[k]]  {zero}

This shows the level of complexity that is allowed. In particular constant indices
are fine (`_` means `1`, and on the left with `:=` only `1` is allowed), arrays-of-arrays are fine
on the right, as are arrays indexed by other arrays.
Functions like `f(D)` will be called on every iteration.
The eltype of `A` comes from `T = typeof(rhs(1,1,1, B,C,J,D))`.

        @tullio A[i] := B[i] + C[i]              # range is i ∈ (1:∞) ∩ axes(B,1) ∩ axes(C,1)
        @tullio A[i] := B[i] - C[i]    {strict}  # asserts axes(B,1) == axes(C,1)
        @tullio A[i] := B[i-2] * B[i]  {offset}  # range is i ∈ axes(B,1).+2 ∩ axes(B,1)
        @tullio A[i] := B[i+5] / B[i]  {cyclic}  # contains B[mod(i+5, axes(B,1)]

By default indexing runs over the largest shared range. Saying `{strict}` demands that ranges agree;
`{cyclic}` treats any shifted index modulo the size of that array.
When creating a new `Array`, a shared range starting at `i<=0` will be clipped to `i>=1`,
and a shared range starting at `i>=2` will give an error.
But if you say `{offset}` to produce an `OffsetArray`, then the output can have any range.
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
    @gensym Fsym Gsym Isym Nsym # Function, tile Grid, tile Index, tilesize Ntuple
    @gensym Ksym Rsym # mutating Kernel function, name for Range_i

    #===== parse input =====#

    store = (ranges=Dict{Any,Any}(1 => 1, Isym => Gsym), constraints=Dict(), rsym=Rsym,
        flags=Set{Any}(multi ? [:multi] : []),
        redop=Ref{Any}(:+), init=Ref{Any}(:(zero($Tsym))),
        arrays=[], rightind=[], curly=[], checks=[], loop=[], unroll=[],
        tilesize=Ref{Any}(0), rolln=Ref(0))

    newarray = @capture(leftright, left_ := right_ )
    newarray || @capture(leftright, left_ = right_ ) || error("expected A[] := B[] or A[] = B[], got $ex")

    @capture(left, Z_[leftraw__] | [leftraw__] ) ||
        error("can't understand LHS, expected A[i,j,k], got $left")
    isnothing(Z) && @gensym Z
    leftind = reverse(filter(i -> i isa Symbol, leftraw)) # default outer loop variables

    if @capture(after1, {stuff__} )
        readcurly(after1, store)
    else
        readred(after1, store, Tsym)
        readcurly(after2, store)
    end

    modright = MacroTools.postwalk(rightwalk(store), right)
    unique!(store.arrays)
    unique!(store.rightind)

    if newarray && !(:offset in store.flags)
        for i in leftind
            get!(store.constraints, i, Any[])
            push!(store.constraints[i], :( 1:typemax(Int) ))
        end
    elseif !newarray
        saveconstraintsmod(Z, leftraw, store, false) # replaces leftwalk
    end

    #===== process & check input =====#

    resolveranges(store)

    redind = setdiff(store.rightind, leftind)

    isempty(store.loop) && isempty(store.unroll) ?
        append!(store.loop, redind) :
        sort(redind) == sort(vcat(store.loop, store.unroll)) ||
        error("if you give any reduction indices, you must give them all: $(Tuple(redind))")

    isempty(store.unroll) || UNROLLOK ||
        @warn "can't unroll loops on Julia $VERSION" maxlog=1 _id=hash(leftright)

    for j in setdiff(leftind, store.rightind)
        haskey(store.ranges, j) || error("unable to infer range of index $j")
        push!(store.rightind, j) # I guess right now means correct!
    end

    isempty(setdiff(store.curly, store.rightind)) || error("some indices in {} are not in expression")

    if newarray && !(:offset in store.flags) # && !(:zero in store.flags)
        for i in leftind
            str = "range of index $i must start at one, try {offset}, using OffsetArrays"
            push!(store.checks, :( @assert 1 == minimum($(store.ranges[i])) $str ))
        end
    end

    #===== rhs function, and eltype(Z) =====#

    outex = quote end

    append!(outex.args, store.checks)

    funwithargs = :( $Fsym($(store.rightind...), $(store.arrays...)) )
    push!(outex.args, :( local $funwithargs = @inbounds $modright ))

    if newarray
        allfirst = map(i -> :(first($(store.ranges[i]))), store.rightind)
        push!(outex.args, :( local $Tsym = typeof($Fsym($(allfirst...), $(store.arrays...))) ))
        outsizes = map(i -> :(length($(store.ranges[i]))), leftraw) # before tiles!
        outranges = map(i -> store.ranges[i], leftraw)
    else
        push!(outex.args, :( local $Tsym = eltype($Z) ))
    end
    push!(outex.args, :( $Tsym == Any && @warn "eltype is Any, sorry" maxlog=1 _id=$(hash(right))))

    #===== tiles =====#

    if store.tilesize[] != 0
        length(store.curly) == 0 && append!(store.curly, leftind)

        nt = length(store.curly)
        if store.tilesize[] isa Int
            tsz = trunc(Int, (store.tilesize[])^(1/nt))
            tiletup = ntuple(_ -> tsz, nt)
        else # you got tiles(8^3) or something, calculate tile sizes later:
            push!(outex.args, :( $Nsym = ntuple(_ -> trunc(Int, ($(store.tilesize[]))^(1/$nt)), $nt) ))
            tiletup = Nsym
        end

        taxes = map(i -> store.ranges[i], store.curly)
        ex = :( local $Gsym = collect(Tullio.TiledIteration.TileIterator(($(taxes...),), $tiletup)) )
        push!(outex.args, ex)

        for (n,i) in enumerate(store.curly)
            store.ranges[i] = :( $Isym[$n] )
        end

        nt==1 && @warn "tiling over just one index!" maxlog=1 #_id=hash(leftright)

        if length(intersect(store.curly, redind)) > 0
            (:thread in store.flags) &&
                error("tiling over a reduction index is not safe with multithreading, right now")
            ntin = length(intersect(store.curly, redind))
            condex = Expr(:comparison, [isodd(m) ? :($Isym[$(nt - (m-1)÷2)]) : :(==) for m=1:2*ntin]..., 1)
            store.init[] = :( ifelse($condex, $(store.init[]), $Z[$(leftraw...)]) )
        end
    end

    #===== gpu loop ranges =====#

    if :gpu in store.flags
        newarray && error("can only use gpu loops in-place, A[] = B[]")
        store.tilesize[] == 0 || error("can't do tiled iteration and gpu loops")

        loopind = vcat(store.loop, leftind)

        for (i,r) in zip(loopind, gpuranges(length(loopind)))
            store.ranges[i] = :( $(store.ranges[i]) ; $r )
        end
    end

    if isempty(redind)
        #===== no reduction =====#

         rex = :( $Z[$(leftraw...)] = $funwithargs ) # note that leftraw may include constants
    else
        #===== reduction loops =====#

        if (:thread in store.flags)
            push!(outex.args, :(local $Csym = Vector{$Tsym}(undef, Threads.nthreads()) ))
            σ = :( $Csym[Threads.threadid()] )
        else
            σ = Ssym
        end

        ex = :( $σ = $(store.redop[])($σ, $funwithargs ) )
        ex = recurseloops(ex, store)
        ex = :( $σ = $(store.init[]); $ex; )

        rex = :( $ex; $Z[$(leftraw...)] = $σ )
    end

    #===== output array =====#

    newex = quote end

    if newarray && (:offset in store.flags)
        push!(newex.args, :( $Z = OffsetArrays.OffsetArray{$Tsym,$(length(leftraw))}(undef, $(outranges...)) ))
    elseif newarray
        push!(newex.args, :( $Z = Array{$Tsym,$(length(leftraw))}(undef, $(outsizes...)) ))
    end
    if :zero in store.flags
        push!(newex.args, :( $Z .= zero($Tsym) ))
    end

    #===== decide on final loops =====#

    leftcurly = intersect(store.curly, leftind) # take order from {}, may be super/subset
    loopind = vcat(leftcurly, setdiff(leftind, leftcurly))
    if store.tilesize[] != 0
        pushfirst!(loopind, Isym) # range Gsym is already in dict
    end
    ministore = (ranges=store.ranges, loop=copy(loopind), unroll=[], rolln=Ref(0), flags=store.flags)

    if :generate in store.flags
        newarray || error("can't use {generate} on in-place operations")
        nope = intersect(store.flags, [:thread, :forward, :gpu])
        isempty(nope) || error("can't use {generate} with $nope")

        #===== generator instead of nested loops =====#
        # this is a bit of an ugly hack! Unsure I want it anyway.

        genex = recursegenerator(isempty(redind) ? funwithargs : :($ex; $σ), ministore)

        push!(outex.args, :( $Z = $genex ))

    else
        #===== nested loops =====#

        loopex = recurseloops(rex, ministore)

        if isempty(loopind) # scalar output needs a let block in global scope
            loopex = :( let; @inbounds $loopex; end )
        elseif (:thread in store.flags)
            loopex = :( @inbounds Threads.@threads $loopex )
        else
            loopex = :( @inbounds $loopex )
        end
    end

    if :forward in store.flags
        newarray || error("can't use {forward} on in-place operations")

        #===== zygote adjointed function =====#

        push!(newex.args, loopex)
        push!(newex.args, Z)
        fex = :( $Ksym(($(store.arrays...),)) = $newex ) # function takes a tuple!
        push!(outex.args, fex)

        push!(outex.args, :( $Z = Zygote.forwarddiff($Ksym, ($(store.arrays...),)) ))

    elseif :gpu in store.flags

        #===== out of body cognition =====#

        fex = :( $Ksym($Z, $(store.arrays...)) = ($loopex; Tullio.GPUifyLoops.@synchronize; nothing) )
        push!(outex.args, fex)

        cuex = :($Ksym($Z::CuArray, $(store.arrays...)) = begin
                Tullio.GPUifyLoops.launch(Tullio.GPUifyLoops.CUDA(), $Ksym, $Z, $(store.arrays...); threads=size($Z))
                end)
        push!(outex.args, cuex)

        push!(outex.args, :($Ksym($Z, $(store.arrays...)) )) # make K!(Z) to do the work!

        # Base.find_package("CuArrays") == nothing && error("can't use {gpu} without a GPU!")

    elseif !(:generate in store.flags)
        #===== simply run the loops! =====#

        append!(outex.args, newex.args) # first make the output

        push!(outex.args, loopex) # then run.
    end

    #===== done! =====#

    push!(outex.args, Z)
    esc(outex)
end

#===== parsing expressions =====#

primewalk(ex) = begin # e.g. @macroexpand1 @tullio A[i'] := f(B[i',:,$c] )
    @capture(ex, j_Symbol') && return Symbol(j,'′') # normalise i' to i′
    @capture(ex, A_[iraw__]) || return ex
    inds = map(iraw) do i
        i == :_ ? 1 :                             # and A[i,_] to A[i,1]
        isdollar(i) ? :(identity($(i.args[1]))) : # trick to protect constants
        i == :(:) ? :(Colon()) :                  # and treat : like a constant
        i
    end
    return :( $A[$(inds...)] )
end

isdollar(i) = false
isdollar(i::Expr) = i.head == :$

rightwalk(store) = ex -> begin
        @capture(ex, A_[inds__]) || return ex
        push!(store.arrays, arrayonly(A))
        saveconstraintsmod(A, inds, store, true)
    end

saveconstraintsmod(A, inds, store, right=true) = begin
    A1 = arrayfirst(A)
    for (d,ex) in enumerate(inds)
        for i in indicesonly(ex)
            right && push!(store.rightind, i)
            ri = indexrange(i, ex, A1, d, store)
            get!(store.constraints, i, Any[])
            isnothing(ri) || push!(store.constraints[i], ri)
        end
        if (:cyclic in store.flags) && !(ex isa Symbol)
            inds[d] = :( 1 + mod($ex - 1, size($A1,$d)) )
        end
    end
    n = length(inds)
    str = "expected a $n-array $A1" # already arrayfirst(A)
    push!(store.checks, :( @assert ndims($A1) == $n $str ))
    return :( $A[$(inds...)] )
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

#===== range calculation =====#

resolveranges(store) = (:strict in store.flags) ?
    map(resolvestrict(store), store.rightind) :
    map(resolveintersect(store), store.rightind)

resolvestrict(store) = i ->
    for ax in store.constraints[i]
        if haskey(store.ranges, i)
            str = "range of index $i must agree"
            push!(store.checks, :( @assert $(store.ranges[i]) == $ax $str ))
        else
            store.ranges[i] = ax
        end
    end

resolveintersect(store) = i ->
    if haskey(store.ranges, i)
        # for ax in store.constraints[i]
        #     str = "range of index $i must fit within given arrays"
        #     push!(store.checks, :( @assert issubset($(store.ranges[i]), $ax) $str ))
        # end
    else
        res = length(store.constraints[i])==1 ?
            store.constraints[i][1] : # because intersect(1:3) isa Vector, wtf?
            :( intersect($(store.constraints[i]...)) )
        r_i = Symbol(store.rsym,:_, i) # could in fact generate here, not using rsym elsewhere
        push!(store.checks, :( local $r_i = $res ))
        store.ranges[i] = r_i
    end

indicesonly(n) = Symbol[]
indicesonly(i::Symbol) = i == :end ? Symbol[] : [i]
indicesonly(ex::Expr) =
    if @capture(ex, -j_ )
        return [j]
    elseif @capture(ex, (s_ * j_) | (-s_ * j_))
        return vcat(indicesonly(s), indicesonly(j))
    elseif @capture(ex, (j_ + k_) | (j_ - k_) )
        return vcat(indicesonly(j), indicesonly(k))
    elseif @capture(ex, (s_ * j_ + k_) | (s_ * j_ - k_) ) # not sure this will work
        return vcat(indicesonly(s), indicesonly(j), indicesonly(k))
    else
        return Symbol[]
    end

indexrange(i, ex::Symbol, A, d, store) = :( axes($A, $d) )
indexrange(i, ex::Expr, A, d, store) = begin
    s, k = indexscaleshift(i, ex)
    k = MacroTools.postwalk(x -> x == :end ? :(size($A,$d)) : x, k)

    (:cyclic in store.flags) && return :(axes($A, $d)) # for s=+-1 only!

    if s==1 # meaning i + k
        if isconst(k, store)
            if ispos(k)
                return :(axes($A, $d) .- ($k)) # also runs fine for k<0
            else
                return :(axes($A, $d) .+ $(makepos(k))) # prettier
            end
        else
            # r_k = Symbol(store.rsym,:_, makepos(k)) # need to ensure this gets defined first!
            # and that these weaker things come later & are skipped... 3rd category?
            haskey(store.ranges, makepos(k)) || error("need an explicit range for $(makepos(k))")
            r_k = store.ranges[makepos(k)]
            if ispos(k)
                return :( range(minimum(axes($A, $d))-minimum($r_k), stop=maximum(axes($A, $d))-maximum($r_k)) )
            else
                return :( range(minimum(axes($A, $d))+maximum($r_k), stop=maximum(axes($A, $d))+minimum($r_k)) )
            end
        end

    elseif s==-1 # meaning -i + k
        if isconst(k, store)
            return :(axes($A, $d) .- size($A, $d) .+ ($k) .- 1)
        else
            # r_k = Symbol(store.rsym,:_, makepos(k))
    haskey(store.ranges, makepos(k)) || error("need an explicit range for $(makepos(k))")
            r_k = store.ranges[makepos(k)]
            if ispos(k)
                return :( range(-maximum(axes($A, $d))+maximum($r_k), stop=-minimum(axes($A, $d))+minimum($r_k)) )
            else
                return :( range(-maximum(axes($A, $d))-minimum($r_k), stop=-minimum(axes($A, $d))-maximum($r_k)) )
            end
        end
    else
        error("can't handle $ex yet, sorry: only +-$i + stuff")
    end
end

indexscaleshift(i, ex) = begin
    ex == i && return 1, 0
    @capture(ex, -$i ) && return -1, 0

    (@capture(ex, $i + k_ ) || @capture(ex, k_ + $i )) && return 1, k
    (@capture(ex, $i - k_ ) || @capture(ex, -k_ + $i )) && return 1, :(-$k)
    (@capture(ex, -$i + k_ ) || @capture(ex, k_ - $i )) && return -1, k
    (@capture(ex, -$i - k_ ) || @capture(ex, -k_ - $i )) && return -1, :(-$k)

    @capture(ex, s_ * $i ) && return s, 0
    @capture(ex, -s_ * $i ) && return :(-$s), 0

    (@capture(ex, s_ * $i + k_ ) || @capture(ex, k_ + s_ * $i )) && return s, k
    (@capture(ex, s_ * $i - k_ ) || @capture(ex, -k_ + s_ * $i )) && return s, :(-$k)
    (@capture(ex, - s_ * $i + k_ ) || @capture(ex, k_ - s_ * $i )) && return :(-$s), k
    (@capture(ex, - s_ * $i - k_ ) || @capture(ex, -k_ - s_ * $i )) && return :(-$s), :(-$k)

    error("confused about $i inside $ex, sorry")
end

isneg(s::Int) = s<0
isneg(s::Expr) = @capture(s, -σ_)
isneg(s::Symbol) = false
ispos(s) = !isneg(s)
makepos(s::Int) = abs(s)
makepos(s::Expr) = @capture(s, -σ_) ? σ : s
makepos(s::Symbol) = s

isconst(s::Int, store) = true
isconst(s::Symbol, store) = !(s in store.rightind)
isconst(s::Expr, store) = begin
    res = [true]
    MacroTools.postwalk(s) do x
        isdollar(x) && return nothing
        x isa Symbol || return x
        x in store.rightind && (res[]=false)
    end
    res[]
end


#===== parsing option tuples =====#

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
    elseif @capture(i, (j_ in r_) | (j_ ∈ r_) | (j_ = r_))
        unrollpush(store, j)
        store.ranges[j] = r
    elseif @capture(i, j_ <= m_)
        unrollpush(store, j)
        store.ranges[j] = :( Base.OneTo($m) )

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
    elseif i in (:tile, :tiles)
        store.tilesize[] = TILESIZE
    elseif i in (:thread, :threads, :zero, :gpu, :cyclic, :strict, :offset, :forward, :generate, :generator)
        push!(store.flags, spellcheck(i))

    elseif i isa Symbol
        push!(store.curly, i)
    elseif @capture(i, (j_ in r_) | (j_ ∈ r_) | (j_ = r_))
        push!(store.curly, j)
        store.ranges[j] = r
    elseif @capture(i, j_ <= m_)
        push!(store.curly, j)
        store.ranges[j] = :( Base.OneTo($m) )

    else
        error("don't know what to do with index $i")
    end

spellcheck(s) = s==:threads ? :thread : s==:tiles ? :tile : s==:generator ? :generate : s

#===== making loops =====#

recurseloops(ex, store) =
    if !isempty(store.unroll)
        i = pop!(store.unroll)

        r = store.ranges[i]
        ex = iszero(store.rolln[]) ?
            :(Tullio.GPUifyLoops.@unroll for $i in $r; $ex; end) :
            :(Tullio.GPUifyLoops.@unroll $(store.rolln[]) for $i in $r; $ex; end)
        return recurseloops(ex, store)

    elseif !isempty(store.loop)
        i = pop!(store.loop)
        r = store.ranges[i]
        ex = (:gpu in store.flags) ?
            :( Tullio.GPUifyLoops.@loop for $i in $r; $ex; end ) :
            :(for $i in $r; $ex; end)
        return recurseloops(ex, store)

    else
        return ex
    end

recursegenerator(ex, store) =
    if !isempty(store.loop)
        i = pop!(store.loop)
        r = store.ranges[i]
        return recursegenerator(:( $ex for $i in $r ), store)
    else
        return :( collect( $ex ) )
    end

gpuranges(n) = n>3 ? error("only 3 gpu loops for now") :
    [:( threadIdx().x ), :( threadIdx().y ), :( threadIdx().z )][1:n]

#===== making loops =====#

"""
    Tullio.@einsum  A[i,j] := B[i] * C[j]

Since this package is a superset of `Einsum.jl`, you can drop that and
write `using Tullio: @einsum` to use the new macro under the old name.
"""
macro einsum(exs...)
    _tullio(exs...; mod=__module__)
end

macro vielsum(exs...)
    _tullio(exs...; multi=true, mod=__module__)
end

#===== piracy =====#

# precisely https://github.com/JuliaLang/julia/pull/32463
Base.issubset(r::Base.OneTo, s::Base.OneTo) = r.stop <= s.stop
Base.issubset(r::AbstractUnitRange{<:Integer}, s::AbstractUnitRange{<:Integer}) =
    first(r) >= first(s) && last(r) <= last(s)
# more general case, one day https://github.com/JuliaLang/julia/pull/32003
Base.issubset(r::AbstractRange, s::AbstractRange) = begin
        for i in r
            i in s || return false
        end
        return true
    end


#= === TODO ===

* actual GPU loops, {gpu}... if limited to 3 loops, and in-place, then perhaps not too hard

* allow unrolling of LHS indices too? Then {static}...

=#
end # module
