
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

#========== scalar struct ==========#

"""
    OneBox(val)

Trivial 1-element vector, used for scalar redutcions,
to pass the eltype to `∇$ACT!(AT, 𝛥A, ::AbstractArray{$TYP}, ...)`
"""
struct OneBox{T} <: AbstractVector{T}
    val::T
end
Base.size(::OneBox) = (1,)
Base.getindex(o::OneBox, i::Integer...) = o.val

#========== gradient hooks ==========#
# Macros like @adjoint need to be hidden behind include(), it seems:

# @init @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include("grad/zygote.jl")

# @init @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" include("grad/reverse.jl")

# Provided via en extension on 1.9+
if !isdefined(Base, :get_extension)
import ChainRulesCore

function ChainRulesCore.rrule(ev::Eval, args...)
    Z = ev.fwd(args...)
    Z, function tullio_back(Δ)
        isnothing(ev.rev) && error("no gradient definition here!")
        dxs = map(ev.rev(Δ, Z, args...)) do dx
            dx === nothing ? ChainRulesCore.ZeroTangent() : dx
        end
        tuple(ChainRulesCore.ZeroTangent(), dxs...)
    end
end
end

if !isdefined(Base, :get_extension)
using Requires
end

@static if !isdefined(Base, :get_extension)
function __init__()
    @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("../ext/TullioTrackerExt.jl")
    @require FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b" include("../ext/TullioFillArraysExt.jl")
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/TullioCUDAExt.jl")
end
end

#========== vectorised gradients ==========#

@inline onlyone(cond::Bool) = cond
@inline onlyone(cond::Bool, seen::Int) = cond && iszero(seen)

@inline anyone(cond::Bool) = cond

#=

@init @require LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890" begin
    using .LoopVectorization # version 0.9+ only now
    using .LoopVectorization.VectorizationBase: Vec, Mask, prevpow2
    SVec{N,T} = Vec{N,T}
    end
    # Functions needed for safe vectorised max gradient
    @inline Tullio.onlyone(cond::Bool, seen::SVec) = cond && allzero(seen)

    @inline Tullio.onlyone(cond::Mask{W}) where {W} = Mask{W}(prevpow2(cond.u))
    @inline Tullio.onlyone(cond::Mask, seen::Union{Int,SVec}) =
        Tullio.allzero(seen) ? Tullio.onlyone(cond) : zero(cond)

    @inline allzero(seen::Integer) = iszero(seen)
    @inline allzero(seen::SVec) = iszero((!iszero(seen)).u)

    @inline Tullio.anyone(cond::Mask) = !iszero(cond.u)
end

=#

#========== storage unwrapper ==========#

"""
    storage_type(adjoint(view(A,...))) == Array{Int,2}
    storage_type(A, B, C) == Array{Int,N} where N

Recursively unwraps wrappers, and combines approximately with `promote_type`.
(Used as the trait to send CuArray to KernelAbstractions and Array{Float or Int}
to LoopVectorization.)
"""
function storage_type(A::AbstractArray)
    P = parent(A)
    typeof(A) === typeof(P) ? typeof(A) : storage_type(P)
end
storage_type(A) = typeof(A)
storage_type(A, Bs...) = promote_storage(storage_type(A), storage_type(Bs...))
storage_type() = AbstractArray

promote_storage(::Type{A}, ::Type{B}) where {A <: Array{T,N}, B <: Array{S,M}} where {T,N,S,M} =
    N==M ? Array{promote_type(T,S), N} : Array{promote_type(T,S)}
promote_storage(::Type{A}, ::Type{B}) where {A <: Array{T,N}, B <: AbstractRange{S}} where {T,N,S} =
    N==1 ? Vector{promote_type(T,S)} : Array{promote_type(T,S)}
promote_storage(::Type{A}, ::Type{B}) where {A <: AbstractRange{T}, B <: Array{S,M}} where {T,S,M} =
    M==1 ? Vector{promote_type(T,S)} : Array{promote_type(T,S)}
promote_storage(T, S) = promote_type(T, S)

#========== fin ==========#
