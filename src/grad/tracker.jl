
using .Tracker

(ev::Eval)(A::Tracker.TrackedArray, args...) = Tracker.track(ev, A, args...)
(ev::Eval)(A, B::Tracker.TrackedArray, args...) = Tracker.track(ev, A, B, args...)
(ev::Eval)(A::Tracker.TrackedArray, B::Tracker.TrackedArray, args...) = Tracker.track(ev, A, B, args...)

Tracker.@grad function (ev::Eval)(args...)
    Z = ev.fwd(Tracker.data.(args)...)
    Z, Δ -> begin
        ev.rev===nothing && throw("No gradient definition found! Running `@tullio` with keyword `verbose=true` may print the reason")
        tuple(ev.rev(Tracker.data(Δ), Z, Tracker.data.(args)...)...)
    end
end
