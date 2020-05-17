
#========== master evaluator ==========#

"""
    Eval(fwd, rev)
    (e::Eval)(A,B) = fwd(A,B)

This holds the functions `$MAKE` which creates the output array
(before calling `threader($ACT!,...)` to fill it)
and the function `∇$MAKE` which does the reverse pass for differentiation.

It exists so that gradient hooks for various packages can be attached to this,
once. Then `$MAKE` need only be defined in local scope.
"""
struct Eval{F,R}
    fwd::F
    rev::R
end

(ev::Eval)(args...) = ev.fwd(args...)

#========== gradient hooks ==========#


using Requires

@init @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    using .Zygote

    Zygote.@adjoint function (ev::Eval)(args...)
        ev.fwd(args...), Δ -> begin
            isnothing(ev.rev) && error("no gradient definition here!")
            tuple(nothing, ev.rev(Δ, args...)...)
        end
    end

    $promote_storage(T::Type, ::Type{<:Zygote.Fill}) = T
    $promote_storage(::Type{<:Zygote.Fill}, T::Type) = T

end

@init @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
    using .Tracker
    using .Tracker: @grad, TrackedArray, track, data

    (ev::Eval)(A::TrackedArray, args...) = track(ev, A, args...)
    (ev::Eval)(A, B::TrackedArray, args...) = track(ev, A, B, args...)
    (ev::Eval)(A::TrackedArray, B::TrackedArray, args...) = track(ev, A, B, args...)

    @grad function (ev::Eval)(args...)
        ev.fwd(data.(args)...), Δ -> begin
            isnothing(ev.rev) && error("no gradient definition here!")
            tuple(ev.rev(Δ, data.(args)...)...)
        end
    end

    # (ev::Eval)(A::Tracker.TrackedArray, args...) = Tracker.track(ev, A, args...)
    # (ev::Eval)(A, B::Tracker.TrackedArray, args...) = Tracker.track(ev, A, B, args...)
    # (ev::Eval)(A::Tracker.TrackedArray, B::Tracker.TrackedArray, args...) = Tracker.track(ev, A, B, args...)

    # Tracker.@grad function (ev::Eval)(args...)
    #     ev.fwd(Tracker.data.(args)...), Δ -> begin
    #         isnothing(ev.rev) && error("no gradient definition here!")
    #         tuple(ev.rev(Δ, Tracker.data.(args)...)...)
    #     end
    # end

end

@init @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
    using .ReverseDiff

    (ev::Eval)(A::ReverseDiff.TrackedArray, args...) = ReverseDiff.track(ev, A, args...)
    (ev::Eval)(A, B::ReverseDiff.TrackedArray, args...) = ReverseDiff.track(ev, A, B, args...)
    (ev::Eval)(A::ReverseDiff.TrackedArray, B::ReverseDiff.TrackedArray, args...) = ReverseDiff.track(ev, A, B, args...)

    ReverseDiff.@grad function (ev::Eval)(args...)
        ev.fwd(ReverseDiff.value.(args)...), Δ -> begin
            isnothing(ev.rev) && error("no gradient definition here!")
            ev.rev(Δ, ReverseDiff.value.(args)...)
        end
    end

end


#=
@init @requite Yota = "cd998857-8626-517d-b929-70ad188a48f0" begin
    using .Yota

#     for (n,A) in enumerate(store.arrays)
#         push!(evalex, quote
#             Yota.@diffrule  $make($(store.arrays...), $(store.scalars...))  $A  getindex($∇make(dy, $(store.arrays...), $(store.scalars...)), $n)
#         end)
#     end

end
=#


#========== not gradients ==========#

using Requires

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    using .CuArrays

    $threader(fun!::Function, T::Type{<:CuArray},
        Z::AbstractArray, As::Tuple, Is::Tuple, Js::Tuple; block=0, keep=nothing) =
        fun!(T, Z, As..., Is..., Js..., keep)

    $∇threader(fun!::Function, T::Type{<:CuArray},
        As::Tuple, Is::Tuple, Js::Tuple; block=0) =
        fun!(T, As..., Is..., Js...,)

end

@init @require LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890" begin

    # some missing definitions, should live SLEEFpirates?
    using .LoopVectorization: SVec
    @inline svec(tup::NTuple{N,T}) where {N,T} = SVec{N,T}(tup...)
    @inline Base.inv(sv::SVec{N,<:Integer}) where {N} = svec(ntuple(n -> inv(sv[n]), N))
    @inline Base.sqrt(sv::SVec{N,<:Integer}) where {N} = svec(ntuple(n -> sqrt(sv[n]), N))
    @inline Base.trunc(T::Type, sv::SVec{N}) where {N} = svec(ntuple(n -> trunc(T, sv[n]), N))

    @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
        # dual numbers + svec, should live in PaddedMatricesForwardDiff?
        # (And where would the conditional loading go, still here?)
        include("avxdual.jl")
    end
end

#========== fin ==========#
