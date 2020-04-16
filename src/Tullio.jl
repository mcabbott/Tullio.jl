module Tullio

using MacroTools

using LoopVectorization, ForwardDiff

include("macro.jl")
export @tullio

include("shifts.jl")

# include("names.jl")

include("forward.jl")

include("symbolic.jl")

# Faster loading on Julia 1.5, like https://github.com/JuliaPlots/Plots.jl/pull/2544
# if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
#     @eval Base.Experimental.@optlevel 1
# end
# But a few things must be shielded from that:
# module Fast

    include("threads.jl")

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

    # Tullio.storage_type(A::StructArray) = Tullio.storage_type(first(fieldarrays(sa)))

    # promote_storage(T,S) = Base.promote_type(T, S)
    # promote_storage(::Type{AT}, ::Type{BT}) where {AT<:AbstractArray{T,NA}, BT<:AbstractArray{S,NB}} =
    #     Base.promote_type(T, S)

    storage_typejoin(A, Bs...) = Base.promote_typejoin(storage_type(A), storage_typejoin(Bs...))
    storage_typejoin(A) = storage_type(A)

# end
# using .Fast
# for name in names(Fast, all=true)
#     startswith(string(name), "#") && continue
#     name in [:eval, :include] && continue
#     @eval import .Fast: $name
# end

"""
    Tullio.@einsum  A[i,j] += B[i] * C[j]

Since this package is almost superset of `Einsum.jl`, you can probable drop that and
write `using Tullio: @einsum` to use the new macro under the old name. Differences:
* Constants need dollar signs like `A[i,1,\$c] + \$d`, as the macro created a function
  which may not run in the caller's scope.
* Updating `A` with weird things like `*=` won't work.
"""
macro einsum(exs...)
    _tullio(exs...; mod=__module__)
end

end # module
