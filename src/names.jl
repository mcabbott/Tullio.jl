
#========== a few functions for handling names ==========#

using NamedDims

getnames(A::NamedDimsArray) = dimnames(A)
getnames(A::AbstractArray) = begin
    P = parent(A)
    typeof(P) === typeof(A) ? nothing : getnames(P) # only for trivial wrappers
    # typeof(P) === typeof(A) ? nothing : outmap(A, getnames(P))
end

hasnames(A::NamedDimsArray) = true
hasnames(A::AbstractArray) = begin
    P = parent(A)
    typeof(P) === typeof(A) ? false : hasnames(P)
end

permuteby(A::AbstractArray, ns::Tuple) = begin
    hasnames(A) || throw(ArgumentError("expected an array with names matching $ns"))
    perm = NamedDims.dim(getnames(A), ns)
    PermutedDimsArray(NamedDims.unname(A), perm)
end


#=
Would be nice not to depend on NamedDims.

You could directly unwrap names like so... in the hope of going fast... however it's less general:

push!(store.outex, :(
    A = PermutedDimsArray(NamedDims.unname(A), NamedDims.dim(NamedDims.dimnames(A), $target))
))

Maybe don't, just index by keyword, and on LHS add wrapper at the end.

=#
