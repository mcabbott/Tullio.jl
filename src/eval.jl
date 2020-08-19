
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
# Macros like @adjoint need to be hidden behind include(), it seems:

using Requires

@init @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include("grad/zygote.jl")

@init @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("grad/tracker.jl")

@init @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" include("grad/reverse.jl")

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

getonly(a::AbstractArray) = first(a) # just avoid first(::CuArray)

using Requires

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    using .CuArrays

    Tullio.threader(fun!::F, ::Type{T},
        Z::AbstractArray, As::Tuple, Is::Tuple, Js::Tuple,
        redfun, block=0, keep=nothing) where {F<:Function, T<:CuArray} =
        fun!(T, Z, As..., Is..., Js..., keep)

    Tullio.∇threader(fun!::F, ::Type{T},
        As::Tuple, Is::Tuple, Js::Tuple, block=0) where {F<:Function, T<:CuArray} =
        fun!(T, As..., Is..., Js...,)

    Tullio.getonly(a::CuArray) = sum(a)

end

# CUDA replaces CuArrays, Julia >=1.4 only.
@init @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
    using .CUDA

    Tullio.threader(fun!::F, ::Type{T},
        Z::AbstractArray, As::Tuple, Is::Tuple, Js::Tuple,
        redfun, block=0, keep=nothing) where {F<:Function, T<:CuArray} =
        fun!(T, Z, As..., Is..., Js..., keep)

    Tullio.∇threader(fun!::F, ::Type{T},
        As::Tuple, Is::Tuple, Js::Tuple, block=0) where {F<:Function, T<:CuArray} =
        fun!(T, As..., Is..., Js...,)

    Tullio.getonly(a::CuArray) = sum(a)

end

@init @require LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890" begin
    @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
        # dual numbers + svec, should live in PaddedMatricesForwardDiff?
        # (And where would the conditional loading go, still here?)
        include("grad/avxdual.jl")
    end
end


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
