module Tullio

#========== ⚜️ ==========#

export @tullio

@nospecialize

include("tools.jl")

include("macro.jl")

include("symbolic.jl")

include("forward.jl")

include("einsum.jl")

@specialize

include("shifts.jl")

include("threads.jl")


#========== ⚜️ ==========#

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

#========== ⚜️ ==========#

using Requires

function __init__()
    @require LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890" begin

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
end

#========== ⚜️ ==========#

end # module
