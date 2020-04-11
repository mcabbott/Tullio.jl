

#========== cost model ==========#

BLOCKTIMES = 10^6      # how many simple evaluations make it worth starting a thread?

BLOCKLOG = 10^4        # and how many expensive evaluations?

LOGEXP = [:log, :exp]  # operations which trigger that

NOAVX = []             # operations which block @avx

#========== one runtime function ==========#

"""
    divide(apply!, Z, (A,B), (1:5,1:6), (1:7), 10^3)

Calling `apply!(Z, A,B,C, 1:5,1:6, 1:7)` should do the work.
But if there are enough elements, this will call `apply!` many times
in different threads.

Ideally it would do some smart recursive blocking thing.
The first list of indices are safe to partition among threads,
if those in the second list get divided they need to accumulate results.

For now it just divides up the first index among threads.

The case of complete reduction assumes that `Z` is a 0-dim array.
"""
function divide(fun!::Function, Z::AbstractArray, As::Tuple, Is::Tuple, Js::Tuple, block::Int)
    if prod(length, Is) * prod(length, Js) <= block # not worth threading
        return fun!(Z, As..., Is..., Js...)
    end

    if length(Is) >= 1 # not a complete reduction
        Base.@sync for I1 in Iterators.partition(first(Is), length(first(Is)) ÷ Threads.nthreads())
            Threads.@spawn begin
                @sync fun!(Z, As..., I1, Base.tail(Is)..., Js..., zero)
            end
        end

    else
        ndims(Z)==0 || error("my threaded reduction won't work here")

        ZN = similar(Z, Threads.nthreads())
        Base.@sync for J1 in Iterators.partition(first(Js), length(first(Js)) ÷ Threads.nthreads())
            Threads.@spawn begin
                ZT = view(ZN, Threads.threadid())
                @sync fun!(ZT, As..., Is..., J1, Base.tail(Js)..., zero)
            end
        end
        for t in 1:Threads.nthreads()
            Z[] += ZN[t]
        end
    end
end

# function divider!(fun!::Function, Z::AbstractArray, As::Tuple, Is::Tuple, Js::Tuple, block::Int)
#     Is1, Is2 = divideranges(Is, block)
#     Base.@sync begin
#         Threads.@spawn fun!(Z, As..., Is1..., Js..., block)
#         divider!(fun!, Z, Is2..., Js..., block)
#     end
# end
