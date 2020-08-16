
using .ReverseDiff

(ev::Eval)(A::ReverseDiff.TrackedArray, args...) = ReverseDiff.track(ev, A, args...)
(ev::Eval)(A, B::ReverseDiff.TrackedArray, args...) = ReverseDiff.track(ev, A, B, args...)
(ev::Eval)(A::ReverseDiff.TrackedArray, B::ReverseDiff.TrackedArray, args...) = ReverseDiff.track(ev, A, B, args...)

ReverseDiff.@grad function (ev::Eval)(args...)
    Z = ev.fwd(ReverseDiff.value.(args)...)
    Z, Δ -> begin
        isnothing(ev.rev) && error("no gradient definition here!")
        ev.rev(ReverseDiff.value(Δ), Z, ReverseDiff.value.(args)...)
    end
end
