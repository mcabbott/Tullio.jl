
#========== cost model ==========#

COST = Ref(5001)
# matmul: 1_000 < length < 10_000
# batchmul: 1_000 < length < 27_000
# log: crossover about length 10_000

COSTS = Dict(:* => 1, :/ => 2, :log => 1, :exp => 1) # plus 1 initially

callcost(sy, store) =
    if haskey(COSTS, sy)
        store.cost[] += COSTS[sy]
    end
    # if sy in [:log, :exp]
    #     store.cost[] += 10
    # elseif sy in [:/]
    #     store.cost[] += 2
    # elseif sy in [:+, :-, :*]
    #     store.cost[] += 1
    # end

# Then block = COST[] ÷ store.cost[] is the number of iterations at which threading is turned on.

#========== one runtime function ==========#

"""
    divide(apply!, (Z,A,B), (1:5,1:6), (1:7), 100)

Calling `apply!(Z,A,B,C, 1:5,1:6, 1:7)` should do the work.
But if there are enough elements (meaning `5*6*7 > 100`)
then this will call `apply!` many times in different threads.

The first list of indices should safe to partition among threads,
write to different parts of `Z`.
If those in the second list get divided among `apply!` calls,
then it needs to know how to accumulate results (and doesn't yet).

Ideally this would do some smart recursive blocking thing.
But for for now it just divides up the first index among all threads.

The case of complete reduction assumes that `Z` is a 0-dim array,
and calls `apply!(view(ZT, tid), ...)` where `ZT` has space for each thread.
"""
function divide(fun!::Function, As::Tuple, Is::Tuple, Js::Tuple, block::Int)
    if Threads.nthreads() == 1 || (prod(length.(Is)) * prod(length.(Js)) <= block)
        return fun!(As..., Is..., Js...)
    end

    if length(Is) >= 1 # not a complete reduction
        n4 = 4 * ((length(first(Is)) ÷ 4) ÷ Threads.nthreads())
        Base.@sync for I1 in Iterators.partition(first(Is), max(n4,1))
        # Base.@sync for I1 in Iterators.partition(first(Is), length(first(Is)) ÷ Threads.nthreads())
            Threads.@spawn begin
                @sync fun!(As..., I1, Base.tail(Is)..., Js...)#, zero)
            end
        end

    else
        Z = first(As)
        ndims(Z)==0 || error("my threaded reduction won't work here")

        ZN = similar(Z, Threads.nthreads())
        n4 = 4 * ((length(first(Js)) ÷ 4) ÷ Threads.nthreads())
        Base.@sync for J1 in Iterators.partition(first(Js), max(n4,1))
        # Base.@sync for J1 in Iterators.partition(first(Js), length(first(Js)) ÷ Threads.nthreads())
            Threads.@spawn begin
                ZT = view(ZN, Threads.threadid())
                @sync fun!(ZT, Base.tail(As)..., J1, Base.tail(Js)...)#, zero)
            end
        end
        for t in 1:Threads.nthreads()
            Z[] += ZN[t]
        end
    end
end

# function divider!(fun!::Function, As::Tuple, Is::Tuple, Js::Tuple, block::Int)
#     Is1, Is2 = divideranges(Is, block)
#     Base.@sync begin
#         Threads.@spawn fun!(Z, Is1..., Js..., block)
#         divider!(fun!, Is2..., Js..., block)
#     end
# end

#=

using Tullio: divide

function f!(Z, A, I, J)
    Z[I] .= Threads.threadid()
    A[J] .= Threads.threadid()
    nothing
end

zz = zeros(Int, 20); aa = zeros(20); divide(f!, (zz, aa,), (1:20,), (1:20,), 1000); zz

zz = zeros(Int, 20); aa = zeros(20); divide(f!, (zz, aa,), (1:20,), (1:20,), 3); zz


--------+-------+
|       |       |
|   1   |       |
|       |       |
--------+-------+
|       |       |
|       |   2   |
|       |       |
--------+-------+

--------+-------+
|       |       |
|   X   |   2   |
|       |       |
--------+-------+
|       |       |
|   1   |   X   |
|       |       |
--------+-------+


=#
