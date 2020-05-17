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

include("eval.jl")

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

end # module
