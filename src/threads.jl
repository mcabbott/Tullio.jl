
#========== cost "model" ==========#

BLOCK = Ref(2^19)
# matmul: crossover about 70x70 on my laptop, 70^3 = 343_000, log2(70^3) = 18.3, but only 30% effect at 100^3=10^6
# batchmul: crossover between 20 & 30, log2(20^4) == 17.3, log2(30^4) == 19.6
# contract01: 1500 * 100, length 15_000, doesn't want threading
# cosine01: block 65_536, not sure if it wants
# log: vector crossover about length 10_000

COSTS = Dict(:* => 0, :/ => 2, :log => 10, :exp => 10) # plus 1 initially

callcost(sy, store) =
    if haskey(COSTS, sy)
        store.cost += COSTS[sy]
    end

# Then block = BLOCK[] ÷ store.cost[] is the number of iterations at which threading is turned on.

#========== runtime functions ==========#

"""
    threader(f!,T, Z, (A,B), (1:5,1:6), (1:7); block=100, keep=nothing)

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
@inline function threader(fun!::Function, T::Type, Z::AbstractArray, As::Tuple, I0s::Tuple, J0s::Tuple, block, keep=nothing)
    if !all(r -> r isa AbstractUnitRange, I0s) || !all(r -> r isa AbstractUnitRange, J0s)
        fun!(T, Z, As..., I0s..., J0s..., keep) # don't thread ranges like 10:-1:1
        return nothing
    end
    Is = map(UnitRange, I0s)
    Js = map(UnitRange, J0s)
    if isnothing(block)
        fun!(T, Z, As..., Is..., Js..., keep)
    elseif length(Is) >= 1
        thread_halves(fun!, T, (Z, As...), Is, Js, block, Threads.nthreads(), keep)
    elseif length(Z) == 1 && eltype(Z) <: Number
        thread_scalar(fun!, T, Z, As, Js, block, Threads.nthreads(), keep)
    else
        fun!(T, Z, As..., Is..., Js..., keep)
    end
    return nothing
end

"""
    ∇threader(f!,T, (dA,dB,dZ,A,B), (1:5), (1:6,1:7); block=100)

Again, calling `f!(T, dA,dB,dZ,A,B, 1:5,1:6, 1:7)` should do the work.

The first tuple of ranges should be safe to thread over, e.g. those in common
to all output arrays. If there are none, then it takes a second strategy
of dividing up the other ranges into blocks disjoint in every index,
and giving those to different threads.
"""
function ∇threader(fun!::Function, T::Type, As::Tuple, I0s::Tuple, J0s::Tuple, block)
    Is = map(UnitRange, I0s)
    Js = map(UnitRange, J0s)
    if isnothing(block)
        fun!(T, As..., Is..., Js...)
    elseif length(Is) >= 1
        thread_halves(fun!, T, As, Is, Js, block, Threads.nthreads())
    else
        thread_quarters(fun!, T, As, Js, block, Threads.nthreads())
    end
    nothing
end


function thread_halves(fun!::Function, T::Type, As::Tuple, Is::Tuple, Js::Tuple, block::Int, spawns::Int, keep=nothing)
    if spawns >= 2 && productlength(Is,Js) > block && productlength(Is) > 2
        I1s, I2s = cleave(Is, maybe32divsize(T))
        Base.@sync begin
            Threads.@spawn thread_halves(fun!, T, As, I1s, Js, block, div(spawns,2), keep)
            Threads.@spawn thread_halves(fun!, T, As, I2s, Js, block, div(spawns,2), keep)
        end
    elseif length(Is) + length(Js) >= 2
        block_halves(fun!, T, As, Is, Js, keep)
    else
        fun!(T, As..., Is..., Js..., keep)
    end
    nothing
end

const MINIBLOCK = Ref(64^3) # 2x quicker matmul at size 1000

function block_halves(fun!::Function, T::Type, As::Tuple, Is::Tuple, Js::Tuple, keep=nothing, final=true)
    if productlength(Is,Js) <= MINIBLOCK[]
        return fun!(T, As..., Is..., Js..., keep, final)
    elseif maximumlength(Is) > maximumlength(Js)
        I1s, I2s = cleave(Is)
        block_halves(fun!, T, As, I1s, Js, keep, final)
        block_halves(fun!, T, As, I2s, Js, keep, final)
    else
        J1s, J2s = cleave(Js)
        block_halves(fun!, T, As, Is, J1s, keep, false)
        block_halves(fun!, T, As, Is, J2s, true, final)
    end
    nothing
end


#=

using Tullio
Tullio.MINIBLOCK[] = 4
Z = zeros(Int, 11,9);
cnt = 0
f!(::Type, Z, i, j, keep) = begin
    global cnt
    Z[i,j] .= (global cnt+=1)
end
Tullio.block_halves(f!, Array, (Z,), UnitRange.(axes(Z)), (), nothing)
Z

  1   1   3   3   5   5   7   8   8
  1   1   3   3   5   5   7   8   8
  2   2   4   4   6   6   9  10  10
  2   2   4   4   6   6   9  10  10
 11  11  13  13  19  19  21  21  21
 12  12  14  14  20  20  22  23  23
 12  12  14  14  20  20  22  23  23
 15  15  16  16  24  24  26  27  27
 15  15  16  16  24  24  26  27  27
 17  17  18  18  25  25  28  29  29
 17  17  18  18  25  25  28  29  29

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

function thread_scalar(fun!::Function, T::Type, Z::AbstractArray, As::Tuple, Js::Tuple, block::Int, spawns::Int, keep=nothing)
    if productlength(Js) <= block || spawns < 2
        # @info "thread_scalar on $(Threads.threadid())" Js
        return fun!(T, Z, As..., Js..., keep)
    else
        Z1, Z2 = similar(Z), similar(Z)
        J1s, J2s = cleave(Js)
        Base.@sync begin
            Threads.@spawn thread_scalar(fun!, T, Z1, As, J1s, block, div(spawns,2))
            Threads.@spawn thread_scalar(fun!, T, Z2, As, J2s, block, div(spawns,2))
        end
        if keep === nothing
            Z .= Z1 .+ Z2
        else
            Z .+= Z1 .+ Z2
        end
    end
    nothing
end

function thread_quarters(fun!::Function, T::Type, As::Tuple, Js::Tuple, block::Int, spawns::Int)
    if productlength(Js) <= block || count(r -> length(r)>=2, Js) < 2 || spawns < 4
        return fun!(T, As..., Js...)
    else
        Q11, Q12, Q21, Q22 = quarter(Js, maybe32divsize(T))
        Base.@sync begin
            Threads.@spawn thread_quarters(fun!, T, As, Q11, block, div(spawns,4))
            Threads.@spawn thread_quarters(fun!, T, As, Q22, block, div(spawns,4))
        end
        Base.@sync begin
            Threads.@spawn thread_quarters(fun!, T, As, Q12, block, div(spawns,4))
            Threads.@spawn thread_quarters(fun!, T, As, Q21, block, div(spawns,4))
        end
    end
    nothing
end

productlength(Is::Tuple) = prod(length.(Is))
productlength(Is::Tuple, Js::Tuple) = productlength(Is) * productlength(Js)

maximumlength(Is::Tuple) = max(length.(Is)...)
maximumlength(::Tuple{}) = 0

maybe32divsize(::Type{<:AbstractArray{T}}) where T<:Number = max(1, 32 ÷ sizeof(T))
maybe32divsize(::Type) = 4

"""
    cleave((1:10, 1:20, 5:15)) -> lo, hi
Picks the longest of a tuple of ranges, and divides that one in half.
"""
function cleave(ranges::Tuple{Vararg{<:UnitRange,N}}, step::Int=4) where {N}
    longest = foldl((l,r) -> length(l) >= length(r) ? l : r, ranges; init=1:0)
    c = foldl((l,i) -> ranges[i]==longest ? i : l, ntuple(identity, N); init=0)

    cleft = if length(longest) >= 2*step
        minimum(longest) - 1 + step * div(length(longest), step * 2)
    else
        minimum(longest) - 1 + div(length(longest), 2)
    end
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
cleave(::Tuple{}, n::Int=4) = (), ()

#=
@btime Tullio.cleave((1:100, 1:50, 50:90)) # was OK
@code_warntype Tullio.cleave((1:100, 1:50, 50:90), 4)
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

    cleft = if long >= 2*step
        minimum(ranges[c]) - 1 + step * div(long, step * 2)
    else
        minimum(ranges[c]) - 1 + div(long, 2)
    end
    delta = if second >= 2*step
        minimum(ranges[d]) - 1 + step * div(second, step * 2)
    else
        minimum(ranges[d]) - 1 + div(second, 2)
    end

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

