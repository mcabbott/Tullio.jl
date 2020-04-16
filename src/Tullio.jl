module Tullio

export @tullio

# Faster loading on Julia 1.5? like https://github.com/JuliaPlots/Plots.jl/pull/2544
if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
    @eval Base.Experimental.@optlevel 1
end

# using MacroTools

include("tools.jl")

include("macro.jl")

include("shifts.jl")

include("symbolic.jl")

using LoopVectorization

using Requires

@init @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
    include("forward.jl")
end

module Fast # shield a few things from from @optlevel 1 above (untimed)
    # shoud maybe incude range calculators too?

    include("threads.jl")

    using LoopVectorization: SVec
    @inline svec(tup::NTuple{N,T}) where {N,T} = SVec{N,T}(tup...)
    @inline Base.inv(sv::SVec{N,<:Integer}) where {N} = svec(ntuple(n -> inv(sv[n]), N))
    @inline Base.sqrt(sv::SVec{N,<:Integer}) where {N} = svec(ntuple(n -> sqrt(sv[n]), N))
    @inline Base.trunc(T::Type, sv::SVec{N}) where {N} = svec(ntuple(n -> trunc(T, sv[n]), N))

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

end
using .Fast
for name in names(Fast, all=true)
    startswith(string(name), "#") && continue
    name in [:eval, :include] && continue
    @eval import .Fast: $name
end
#=
# on 1.5, this is 0.5s better than without, but 4s better than version with ForwardDiff:

julia> @time (using Tullio; Tullio._tullio(:( A[i] := (1:10)[i+j] + (1:3)[j]) ));
  5.424640 seconds (15.60 M allocations: 783.174 MiB, 4.61% gc time)

# without @capture: another second?

julia> @time (using Tullio; Tullio._tullio(:( A[i] := (1:10)[i+j] + (1:3)[j]) ));
  4.630637 seconds (11.98 M allocations: 609.563 MiB, 7.71% gc time)

[ Info: Basic tests took 32.7 seconds # on 1.5, 2nd run
[ Info: Basic tests took 46.2 seconds # on 1.4

=#


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
