module Tullio

#========== ⚜️ ==========#

export @tullio

include("tools.jl")

include("macro.jl")

include("shifts.jl")

include("threads.jl")

include("symbolic.jl")

include("forward.jl")

#========== ⚜️ ==========#

"""
    storage_type(adjoint(view(A,...))) == Array{Int,2}
    storage_type(A, B, C) == Array{Int,N} where N

Recursively unwraps wrappers, and combines with `promote_type`.
"""
function storage_type(A::AbstractArray)
    P = parent(A)
    typeof(A) === typeof(P) ? typeof(A) : storage_type(P)
end
storage_type(A) = typeof(A)
storage_type(A, Bs...) = Base.promote_type(storage_type(A), storage_type(Bs...))

storage_typejoin(A, Bs...) = Base.promote_typejoin(storage_type(A), storage_typejoin(Bs...))
storage_typejoin(A) = storage_type(A)

"""
    Tullio.@einsum  A[i,j] += B[i] * C[j]

Since this package is almost superset of `Einsum.jl`, you can probable drop that and
write `using Tullio: @einsum` to use the new macro under the old name. Differences:
* Constants need dollar signs like `A[i,1,\$c] + \$d`, as the macro creates a function
  which may not run in the caller's scope.
* Updating `A` with weird things like `*=` won't work.
"""
macro einsum(exs...)
    _tullio(exs...; mod=__module__)
end

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
