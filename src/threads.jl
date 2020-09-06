
#========== cost "model" ==========#

const BLOCK = Ref(2^18)
# matmul: crossover about 70x70 on my laptop, 70^3 = 343_000, log2(70^3) = 18.3, but only 30% effect at 100^3=10^6
# batchmul: crossover between 20 & 30, log2(20^4) == 17.3, log2(30^4) == 19.6
# contract01: 1500 * 100, length 15_000, doesn't want threading
# cosine01: block 65_536, not sure if it wants
# log: vector crossover about length 10_000

"""
    COSTS = Dict(:* => 0, :log =>10, ...)

Initial cost is `1`, and every other function call adds the value from this dictionary.
Then `n = BLOCK[] ÷ cost` is the number of iterations at which the macro thinks it
worthwhile to turn on threading; you can override this with keyword `threads=n`.
"""
const COSTS = Dict(:+ => 0, :- => 0, :* => 0,
    :conj => 0, :adjoint => 0, :abs =>0, abs2 => 0,
    :getindex => 0, :getproperty => 0, :getfield => 0,
    :^ => 2, :/ => 2, :div =>2, :rem =>2, :mod =>2,
    :log => 10, :exp => 10) # and all others 10, plus 1 initially

callcost(sy, store) = store.cost += get(COSTS, sy, 10)

#========== runtime functions ==========#

"""
    threader(f!,T, Z, (A,B), (1:5,1:6), (1:7), +, block=100, keep=nothing)

Calling `f!(T, Z,A,B, 1:5,1:6, 1:7, nothing)` should do the work.
But if there are enough elements (meaning `5*6*7 > 100`)
then this will call `f!` many times in different threads.
(`block=nothing` turns the whole thing off.)

The first tuple of ranges are supposed to be safe to thread over,
probably the axes of the output `Z`.
It will subdivide the longest until either there are too few elements,
or it has spent its spawning budget, `nthreads()`.

For a scalar reduction the first tuple will be empty, and `length(Z)==1`.
Then it divides up the other axes, each accumulating in its own copy of `Z`.

`keep=nothing` means that it overwrites the array, anything else (`keep=true`) adds on.
"""
@inline function threader(fun!::F, ::Type{T}, Z::AbstractArray, As::Tuple, I0s::Tuple, J0s::Tuple, redfun, block, keep=nothing) where {F <: Function, T}
    if isnothing(block) # then threading is disabled
        fun!(T, Z, As..., I0s..., J0s..., keep)
        return nothing
    elseif !all(r -> r isa AbstractUnitRange, I0s) || !all(r -> r isa AbstractUnitRange, J0s)
        # don't thread ranges like 10:-1:1, and disable @avx too
        fun!(Array, Z, As..., I0s..., J0s..., keep)
        return nothing
    end

    Is = map(UnitRange, I0s)
    Js = map(UnitRange, J0s)
    Ielements = productlength(Is)
    Jelements = productlength(Js)
    threads = min(Threads.nthreads(), cld(Ielements * Jelements, block), Ielements)

    if length(Is) >= 1 && threads>1
        thread_halves(fun!, T, (Z, As...), Is, Js, threads, keep)
    else
        tile_halves(fun!, T, (Z, As...), Is, Js, keep)
    end
    nothing
end


"""
    ∇threader(f!,T, (dA,dB,dZ,A,B), (1:5), (1:6,1:7), block)

Again, calling `f!(T, dA,dB,dZ,A,B, 1:5,1:6, 1:7)` should do the work.

The first tuple of ranges should be safe to thread over, e.g. those in common
to all output arrays.

If there are none, then it should to take a second strategy
of dividing up the other ranges into tiles disjoint in every index,
and giving those to different threads. But this was only right for 2 indices,
and is now disabled.
"""
function ∇threader(fun!::F, ::Type{T}, As::Tuple, I0s::Tuple, J0s::Tuple, block) where {F <: Function, T}
    if isnothing(block) # then threading is disabled
        fun!(T, As..., I0s..., J0s...)
        return nothing
    elseif !all(r -> r isa AbstractUnitRange, I0s) || !all(r -> r isa AbstractUnitRange, J0s)
        # don't thread ranges like 10:-1:1, and disable @avx too
        fun!(Array, As..., I0s..., J0s...)
        return nothing
    end

    Is = map(UnitRange, I0s)
    Js = map(UnitRange, J0s)
    Ielements = productlength(Is)
    Jelements = productlength(Js)
    threads = min(Threads.nthreads(), cld(Ielements * Jelements, block), Ielements)

    if threads > 1
        thread_halves(fun!, T, As, Is, Js, threads)
    else
        tile_halves(fun!, T, As, Is, Js)
    end
    nothing
end

function thread_halves(fun!::F, ::Type{T}, As::Tuple, Is::Tuple, Js::Tuple, threads::Int, keep=nothing) where {F <: Function, T}
    if threads > 2 && rem(threads,3) == 0 # not always halves!
        I1s, I2s, I3s = trisect(Is)
        task1 = Threads.@spawn begin
            thread_halves(fun!, T, As, I1s, Js, threads÷3, keep)
        end
        task2 = Threads.@spawn begin
            thread_halves(fun!, T, As, I2s, Js, threads÷3, keep)
        end
        thread_halves(fun!, T, As, I3s, Js, threads÷3, keep)
        wait(task1)
        wait(task2)
    elseif threads > 1
        I1s, I2s = cleave(Is, maybe32divsize(T))
        task = Threads.@spawn begin
            thread_halves(fun!, T, As, I1s, Js, threads÷2, keep)
        end
        thread_halves(fun!, T, As, I2s, Js, threads÷2, keep)
        wait(task)
    else
        tile_halves(fun!, T, As, Is, Js, keep)
    end
    nothing
end

function tile_halves(fun!::F, ::Type{T}, As::Tuple, Is::Tuple, Js::Tuple, keep=nothing, final=true) where {F <: Function, T}
    # keep == nothing || keep == true || error("illegal value for keep")
    # final == nothing || final == true || error("illegal value for final")
    maxI, maxJ = maximumlength(Is), maximumlength(Js)
    maxL = tile_maxiter(T)
    if maxI < maxL && maxJ < maxL
        fun!(T, As..., Is..., Js..., keep, final)
    elseif maxI > maxJ
        I1s, I2s = cleave(Is)
        tile_halves(fun!, T, As, I1s, Js, keep, final)
        tile_halves(fun!, T, As, I2s, Js, keep, final)
    else
        J1s, J2s = cleave(Js)
        tile_halves(fun!, T, As, Is, J1s, keep, nothing)
        tile_halves(fun!, T, As, Is, J2s, true, final)
    end
    nothing
end

"""
    TILE[] = $(TILE[])
    tile_maxiter(Array{Float64}) == 64?

This now sets the maximum length of iteration of any index,
before it gets broken in half to make smaller tiles.
`TILE[]` is in bytes.
"""
const TILE = Ref(512) # this is now a length, in bytes!

function tile_maxiter(::Type{<:AbstractArray{T}}) where {T}
    isbitstype(T) || return TILE[] ÷ 8
    max(TILE[] ÷ sizeof(T), 4)
end
tile_maxiter(::Type{AT}) where {AT} = TILE[] ÷ 8 # treat anything unkown like Float64

#=

using Tullio
Z = zeros(Int, 11,9);
cnt = 0
f!(::Type, Z, i, j, ♻️, 💀) = begin
    global cnt
    Z[i,j] .= (global cnt+=1)
end
Tullio.tile_halves(f!, Array, (Z,), UnitRange.(axes(Z)), (), 4, nothing, true)
Z

  1   1   3   3   5   5   7   7   7
  1   1   3   3   5   5   7   7   7
  2   2   4   4   6   6   8   8   8
  2   2   4   4   6   6   8   8   8
  9   9  10  10  13  13  14  14  14
  9   9  10  10  13  13  14  14  14
  9   9  10  10  13  13  14  14  14
 11  11  11  11  15  15  16  16  16
 11  11  11  11  15  15  16  16  16
 12  12  12  12  15  15  16  16  16
 12  12  12  12  15  15  16  16  16

using TiledIteration
function colour!(A, n=1)
    for (i,t) in enumerate(TileIterator(axes(A), ntuple(_->n, ndims(A))))
        A[t...] .= i
    end
    A
end;
colour!(zeros(Int, 11,9), 2)

 1  1   7   7  13  13  19  19  25
 1  1   7   7  13  13  19  19  25
 2  2   8   8  14  14  20  20  26
 2  2   8   8  14  14  20  20  26
 3  3   9   9  15  15  21  21  27
 3  3   9   9  15  15  21  21  27
 4  4  10  10  16  16  22  22  28
 4  4  10  10  16  16  22  22  28
 5  5  11  11  17  17  23  23  29
 5  5  11  11  17  17  23  23  29
 6  6  12  12  18  18  24  24  30

=#

#========== scalar case ==========#

"""
    thread_scalar(f,T, Z, (A,B), (1:5,1:6), +, block=100, keep=nothing)

Just like `threader`, but doesn't take any safe indices `Is`.
And `f` doesn't actually mutate anything, it returns the value.
`Z` is a trivial array which serves mostly to propagate an eltype.
"""
@inline function thread_scalar(fun!::F, ::Type{T}, Z::AbstractArray, As::Tuple, J0s::Tuple, redfun, block, keep=nothing)::eltype(T) where {F <: Function, T}
    if isnothing(block) # then threading is disabled
        return fun!(T, Z, As..., J0s..., keep)
    elseif !all(r -> r isa AbstractUnitRange, J0s)
        # don't thread ranges like 10:-1:1, and disable @avx too
        return fun!(Array, Z, As..., J0s..., keep)
    end

    Js = map(UnitRange, J0s)
    Jelements = productlength(Js)
    threads = min(Threads.nthreads(), cld(Jelements, block), Jelements)

    if threads < 2
        return fun!(T, Z, As..., Js..., keep)
    else
        return scalar_halves(fun!, T, Z, As, Js, redfun, threads, keep)
    end
end

function scalar_halves(fun!::F, ::Type{T}, Z::AbstractArray, As::Tuple, Js::Tuple, redfun, threads, keep=nothing)::eltype(T) where {F <: Function, T}
    if threads < 1
        return fun!(T, Z, As..., Js..., keep)
    else
        J1s, J2s = cleave(Js)
        S1 = first(Z) # scope
        task = Threads.@spawn begin
            S1 = scalar_halves(fun!, T, Z, As, J1s, redfun, threads÷2, nothing)
        end
        S2 = scalar_halves(fun!, T, Z, As, J2s, redfun, threads÷2, keep)
        wait(task)
        return redfun(S1, S2)
    end
end

#========== tuple functions ==========#

@inline productlength(Is::Tuple) = prod(length.(Is))
@inline productlength(Is::Tuple, Js::Tuple) = productlength(Is) * productlength(Js)

@inline maximumlength(Is::Tuple) = max(length.(Is)...)
@inline maximumlength(::Tuple{}) = 0

@inline maybe32divsize(::Type{<:AbstractArray{T}}) where T<:Number = max(1, 32 ÷ sizeof(T))
@inline maybe32divsize(::Type) = 4

"""
    cleave((1:10, 1:20, 5:15)) -> lo, hi
Picks the longest of a tuple of ranges, and divides that one in half.
"""
@inline cleave(::Tuple{}, n::Int=4) = (), ()
@inline function cleave(ranges::Tuple{UnitRange}, step::Int=4)
    r1 = first(ranges)
    cleft = findcleft(r1, step)
    tuple(first(r1):cleft), tuple(cleft+1:last(r1))
end
@inline function cleave(ranges::Tuple{UnitRange,UnitRange}, step::Int=4)
    r1, r2 = ranges
    if length(r1) > length(r2)
        cleft = findcleft(r1, step)
        return tuple(first(r1):cleft, r2), tuple(cleft+1:last(r1), r2)
    else
        cleft = findcleft(r2, step)
        return tuple(r1, first(r2):cleft), tuple(r1, cleft+1:last(r2))
    end
end
@inline @generated function cleave(ranges::Tuple{Vararg{<:UnitRange,N}}, step::Int=4) where {N}
    ex_finds = [quote
        li = length(ranges[$i])
        if li>l
            c = $i
            l = li
        end
    end for i in 1:N]
    ex_alpas = [:($i==c ? (first(ranges[$i]):cleft) : (ranges[$i])) for i in 1:N]
    ex_betas = [:($i==c ? (cleft+1:last(ranges[$i])) : (ranges[$i])) for i in 1:N]
    quote
        c, l = 0, 0
        $(ex_finds...)
        cleft = findcleft(ranges[c], step)
        tuple($(ex_alpas...)), tuple($(ex_betas...))
    end
end

@inline function findcleft(r::UnitRange, step::Int)
    if length(r) >= 2*step
        minimum(r) - 1 + step * div(length(r), step * 2)
    else
        # minimum(r) - 1 + div(length(r), 2, RoundNearest) # not in Julia 1.3
        minimum(r) - 1 + round(Int, length(r)/2)
    end
end

#=
@btime Tullio.cleave(z[],4)  setup=(z=Ref((1:200, 1:500, 1:300)))
@btime Tullio.cleave(z[],4)  setup=(z=Ref((1:200, 1:50)))
@btime Tullio.cleave(z[],4)  setup=(z=Ref((5:55,)))
=#

"""
    trisect((1:10, 1:20, 5:15)) -> lo, mid, hi

Just like `cleave`, but makes 3 pieces, for 6-core machines.
"""
@inline trisect(::Tuple{}) = (), (), ()
@inline trisect(ranges::Tuple{UnitRange}) = map(tuple, findthree(first(ranges)))
@inline function trisect(ranges::Tuple{UnitRange,UnitRange})
    r1, r2 = ranges
    if length(r1) > length(r2)
        a,b,c = findthree(r1)
        return (a,r2), (b,r2), (c,r2)
    else
        a,b,c = findthree(r2)
        return (r1,a), (r1,b), (r1,c)
    end
end
@inline @generated function trisect(ranges::Tuple{Vararg{<:UnitRange,N}}) where {N}
    ex_finds = [quote
        li = length(ranges[$i])
        if li>l
            c = $i
            l = li
        end
    end for i in 1:N]
    ex_alpas = [:($i==c ? (lo) : (ranges[$i])) for i in 1:N]
    ex_betas = [:($i==c ? (mid) : (ranges[$i])) for i in 1:N]
    ex_gammas = [:($i==c ? (hi) : (ranges[$i])) for i in 1:N]
    quote
        c, l = 0, 0
        $(ex_finds...)
        lo,mid,hi = findthree(ranges[c])
        tuple($(ex_alpas...)), tuple($(ex_betas...)), tuple($(ex_gammas...))
    end
end

@inline function findthree(r::UnitRange)
    d = div(length(r), 3)
    i0 = first(r)
    (i0 : i0+d-1), (i0+d : i0+2d-1), (i0+2d : i0+length(r)-1)
end

#========== the end ==========#

