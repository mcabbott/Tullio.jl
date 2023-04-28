module TullioTrackerExt

if !isdefined(Base, :get_extension)
    using ..Tullio, ..Tracker
else
    using Tullio, Tracker
end

(ev::Tullio.Eval)(A::Tracker.TrackedArray, args...) = Tracker.track(ev, A, args...)
(ev::Tullio.Eval)(A, B::Tracker.TrackedArray, args...) = Tracker.track(ev, A, B, args...)
(ev::Tullio.Eval)(A::Tracker.TrackedArray, B::Tracker.TrackedArray, args...) = Tracker.track(ev, A, B, args...)

Tracker.@grad function (ev::Tullio.Eval)(args...)
    Z = ev.fwd(Tracker.data.(args)...)
    Z, Δ -> begin
        isnothing(ev.rev) && error("no gradient definition here!")
        tuple(ev.rev(Tracker.data(Δ), Z, Tracker.data.(args)...)...)
    end
end

end
