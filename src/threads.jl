
#========== cost "model" ==========#

const BLOCK = Ref(2^19)
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
    :/ => 2, :div =>2, :rem =>2, :mod =>2,
    :log => 10, :exp => 10) # plus 1 initially

callcost(sy, store) = store.cost += get(COSTS, sy, 10)

const TILE = Ref(64^3) # 2x quicker matmul at size 1000

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
@inline function threader(fun!::Function, T::Type, Z::AbstractArray, As::Tuple, I0s::Tuple, J0s::Tuple, redfun, block, keep=nothing)
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
    spawns, breaks = threadlog2s(Is, Js, block)

    if spawns<1 && breaks<1 # then skip all the Val() stuff
        fun!(T, Z, As..., Is..., Js..., keep)
#     else
#         _threader(fun!, T, Z, As, Is, Js, redfun, Val(spawns), Val(breaks), keep)
#     end
#     nothing
# end
# function _threader(fun!, T, Z, As, Is, Js, redfun, Val_spawns, Val_breaks, keep)
#     if length(Is) >= 1
    elseif length(Is) >= 1
        Val_spawns = Val(spawns)
        Val_breaks = Val(breaks)
        thread_halves(fun!, T, (Z, As...), Is, Js, Val_spawns, Val_breaks, keep)
    elseif length(Z) == 1 && eltype(Z) <: Number
        scalar_spawns, _ = threadlog2s(Js, (), block)
        Val_spawns = Val(scalar_spawns)
        thread_scalar(fun!, T, Z, As, Js, redfun, Val_spawns, keep)
    else
        fun!(T, Z, As..., Is..., Js..., keep)
    end
    nothing
end

@inline function threadlog2s(Is, Js, block)
    Ielements = productlength(Is)
    Jelements = productlength(Js)

    spawns = max(0,min(
        ceil(Int, log2(min(Threads.nthreads(), Ielements * Jelements / block))),
        floor(Int, log2(Ielements)),
        ))

    breaks = max(0,min(
        round(Int, log2(Ielements * Jelements / TILE[])),
        floor(Int, log2(Ielements)) + floor(Int, log2(Jelements)),
        ) - spawns)

    spawns, breaks
end


"""
    ∇threader(f!,T, (dA,dB,dZ,A,B), (1:5), (1:6,1:7), block=100)

Again, calling `f!(T, dA,dB,dZ,A,B, 1:5,1:6, 1:7)` should do the work.

The first tuple of ranges should be safe to thread over, e.g. those in common
to all output arrays.

If there are none, then it should to take a second strategy
of dividing up the other ranges into tiles disjoint in every index,
and giving those to different threads. But this was only right for 2 indices,
and is now disabled.
"""
function ∇threader(fun!::Function, T::Type, As::Tuple, I0s::Tuple, J0s::Tuple, block)
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
    spawns, breaks = threadlog2s(Is, Js, block)

    if (spawns<1 && breaks<1)
        fun!(T, As..., Is..., Js...)
    else
        thread_halves(fun!, T, As, Is, Js, Val(spawns), Val(breaks))
    end

    # elseif length(Is) >= 1
    #     thread_halves(fun!, T, As, Is, Js, block, Threads.nthreads())
    # else
    #     thread_quarters(fun!, T, As, Js, block, Threads.nthreads())
    # end
    nothing
end


@inline function thread_halves(fun!::Function, T::Type, As::Tuple, Is::Tuple, Js::Tuple, ::Val{spawns}, valb::Val{breaks}, keep=nothing) where {spawns, breaks}
    if spawns > 0
        I1s, I2s = cleave(Is, maybe32divsize(T))

        # Base.@sync begin # 35.024 μs (52 allocations: 81.13 KiB) # maybe
        #     Threads.@spawn thread_halves(fun!, T, As, I1s, Js, Val(spawns-1), valb, keep)
        #     Threads.@spawn thread_halves(fun!, T, As, I2s, Js, Val(spawns-1), valb, keep)
        # end

        # Base.@sync begin #
        #     Threads.@spawn thread_halves(fun!, T, As, I1s, Js, Val(spawns-1), valb, keep)
        #     thread_halves(fun!, T, As, I2s, Js, Val(spawns-1), valb, keep)
        # end

        # t1 = Threads.@spawn thread_halves(fun!, T, As, I1s, Js, Val(spawns-1), valb, keep)
        # t2 = Threads.@spawn thread_halves(fun!, T, As, I2s, Js, Val(spawns-1), valb, keep)
        # wait(t1); wait(t2) # 51.024 μs (35 allocations: 80.44 KiB)

        task = Threads.@spawn thread_halves(fun!, T, As, I1s, Js, Val(spawns-1), valb, keep)
        thread_halves(fun!, T, As, I2s, Js, Val(spawns-1), valb, keep)
        wait(task) # 28.371 μs (25 allocations: 79.53 KiB)

    else
        tile_halves(fun!, T, As, Is, Js, valb, keep)
    end
    nothing
end

@inline function tile_halves(fun!::Function, T::Type, As::Tuple, Is::Tuple, Js::Tuple, ::Val{0}, keep=nothing, final=true)
    fun!(T, As..., Is..., Js..., keep, final)
end
@inline function tile_halves(fun!::Function, T::Type, As::Tuple, Is::Tuple, Js::Tuple, ::Val{breaks}, keep=nothing, final=true) where {breaks}
    # keep == nothing || keep == true || error("illegal value for keep")
    # final == nothing || final == true || error("illegal value for final")
    if maximumlength(Is) > maximumlength(Js)
        I1s, I2s = cleave(Is)
        tile_halves(fun!, T, As, I1s, Js, Val(breaks-1), keep, final)
        tile_halves(fun!, T, As, I2s, Js, Val(breaks-1), keep, final)
    else
        J1s, J2s = cleave(Js)
        tile_halves(fun!, T, As, Is, J1s, Val(breaks-1), keep, nothing)
        tile_halves(fun!, T, As, Is, J2s, Val(breaks-1), true, final)
    end
    nothing
end

#=

using Tullio
Z = zeros(Int, 11,9);
cnt = 0
f!(::Type, Z, i, j, keep) = begin
    global cnt
    Z[i,j] .= (global cnt+=1)
end
Tullio.tile_halves(f!, Array, (Z,), UnitRange.(axes(Z)), (), Val(4), nothing)
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

@inline function thread_scalar(fun!::Function, T::Type, Z::AbstractArray, As::Tuple, Js::Tuple, redfun, ::Val{spawns}, keep=nothing) where {spawns}
    if spawns < 1
        fun!(T, Z, As..., Js..., keep)
    else
        J1s, J2s = cleave(Js)
        Znew = similar(Z)
        task = Threads.@spawn begin
            thread_scalar(fun!, T, Znew, As, J1s, redfun, Val(spawns-1), nothing)
        end
        thread_scalar(fun!, T, Z, As, J2s, redfun, Val(spawns-1), keep)
        wait(task)
        Z[1] = redfun(Z[1], Znew[1])
    end
    nothing
end

#=
function thread_quarters(fun!::Function, T::Type, As::Tuple, Js::Tuple, block::Int, spawns::Int)
    if productlength(Js) <= block || count(r -> length(r)>=2, Js) < 2 || spawns < 4
        return fun!(T, As..., Js...)
    else
        Q11, Q12, Q21, Q22 = quarter(Js, maybe32divsize(T))
        Base.@sync begin
            Threads.@spawn thread_quarters(fun!, T, As, Q11, block, div(spawns,4))
            thread_quarters(fun!, T, As, Q22, block, div(spawns,4))
        end
        Base.@sync begin
            Threads.@spawn thread_quarters(fun!, T, As, Q12, block, div(spawns,4))
            thread_quarters(fun!, T, As, Q21, block, div(spawns,4))
        end
    end
    nothing
end
=#

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
@inline function cleave(ranges::Tuple{UnitRange}, step::Int=4)
    r1 = first(ranges)
    cleft = findcleft(r1, step)
    return tuple(minimum(r1):cleft), tuple(cleft+1:maximum(r1))
end
@inline function cleave(ranges::Tuple{Vararg{<:UnitRange,N}}, step::Int=4) where {N}
    longest = foldl((l,r) -> length(l) >= length(r) ? l : r, ranges; init=1:0)
    c = foldl((l,i) -> ranges[i]==longest ? i : l, ntuple(identity, N); init=0)

    cleft = findcleft(longest, step)
    alpha = ntuple(Val(N)) do i
        ri = ranges[i]
        i == c ? (minimum(ri):cleft) : (minimum(ri):maximum(ri))
    end
    beta = ntuple(Val(N)) do i
        ri = ranges[i]
        i == c ? (cleft+1:maximum(ri)) : (minimum(ri):maximum(ri))
    end
    return alpha, beta
end
@inline cleave(::Tuple{}, n::Int=4) = (), ()

@inline function findcleft(r::UnitRange, step::Int)
    if length(r) >= 2*step
        minimum(r) - 1 + step * div(length(r), step * 2)
    else
        # minimum(r) - 1 + div(length(r), 2, RoundNearest) # not in Julia 1.3
        minimum(r) - 1 + round(Int, length(r)/2)
    end
end

#=
@btime Tullio.cleave((1:100, 1:50, 50:90)) # 15.378 ns (0 allocations: 0 bytes)
@code_warntype Tullio.cleave((1:100, 1:50, 50:90), 4)
@btime Tullio.cleave((1:13,))
=#

"""
    quarter((1:10, 1:20, 3:4)) -> Q11, Q12, Q21, Q22
Picks the longest two ranges, divides each in half, and returns the four quadrants.
"""
function quarter(ranges::Tuple{Vararg{<:UnitRange,N}}, step::Int=4) where {N}
    c::Int, long::Int = 0, 0
    ntuple(Val(N)) do i
        li = length(ranges[i])
        if li > long
            c = i
            long = li
        end
    end
    d::Int, second::Int = 0,0
    ntuple(Val(N)) do j
        j == c && return
        lj = length(ranges[j])
        if lj > second
            d = j
            second = lj
        end
    end

    cleft = findcleft(ranges[c], step)
    delta = findcleft(ranges[d], step)

    Q11 = ntuple(Val(N)) do i
        ri = ranges[i]
        (i == c) ? (minimum(ri):cleft) : (i==d) ? (minimum(ri):delta) : (minimum(ri):maximum(ri))
    end
    Q12 = ntuple(Val(N)) do i
        ri = ranges[i]
        (i == c) ? (minimum(ri):cleft) : (i==d) ? (delta+1:maximum(ri)) : (minimum(ri):maximum(ri))
    end
    Q21 = ntuple(Val(N)) do i
        ri = ranges[i]
        (i == c) ? (cleft+1:maximum(ri)) : (i==d) ? (minimum(ri):delta) : (minimum(ri):maximum(ri))
    end
    Q22 = ntuple(Val(N)) do i
        ri = ranges[i]
        (i == c) ? (cleft+1:maximum(ri)) : (i==d) ? (delta+1:maximum(ri)) : (minimum(ri):maximum(ri))
    end
    return Q11, Q12, Q21, Q22
end

#========== the end ==========#

