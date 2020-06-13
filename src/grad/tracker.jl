
using .Tracker

(ev::Eval)(A::Tracker.TrackedArray, args...) = Tracker.track(ev, A, args...)
(ev::Eval)(A, B::Tracker.TrackedArray, args...) = Tracker.track(ev, A, B, args...)
(ev::Eval)(A::Tracker.TrackedArray, B::Tracker.TrackedArray, args...) = Tracker.track(ev, A, B, args...)

Tracker.@grad function (ev::Eval)(args...)
    Z = ev.fwd(Tracker.data.(args)...)
    Z, Δ -> begin
        isnothing(ev.rev) && error("no gradient definition here!")
        tuple(ev.rev(Δ, Z, Tracker.data.(args)...)...)
    end
end
