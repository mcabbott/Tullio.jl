
using Tullio, Test
using CUDA, CUDAKernels, KernelAbstractions
CUDA.allowscalar(false)
using Tracker, ForwardDiff
@tullio grad=Base

# matmul
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

A = rand(3,40); B = rand(40,500);
@test A * B ≈ mul(A, B)
@test cu(A * B) ≈ mul(cu(A), cu(B))

# gradient
ΔA = Tracker.gradient((A,B) -> sum(mul(A, B)), A, B)[1]
@test ΔA ≈ ones(3,500) * B'
@test cu(ΔA) ≈ Tracker.gradient((A,B) -> sum(mul(A, B)), cu(A), cu(B))[1]

# shifts
@tullio D[i,j] := A[i,j+k]  k in 0:10
@test axes(D) == (1:3, 1:30)
@tullio cD[i,j] := cu(A)[i,j+k]  k in 0:10
@test cD isa CuArray
@test cD ≈ cu(D)

#=
# ranges
@tullio E[i,j] := A[i,j+k-1] + (-1:0.5:1)[k]
@test axes(E) == (1:3, 1:36)
@tullio cE[i,j] := cu(A)[i,j+k-1] + (-1:0.5:1)[k]
@test cE isa CuArray
@test cE ≈ cu(E)
=#

# product
@tullio (*) F[j] := A[i,j]
@test F ≈ vec(prod(A, dims=1))
@tullio (*) cF[j] := cu(A)[i,j]
@test cF ≈ cu(F)

# maximum
g(A) = @tullio (max) G[j] := A[i,j]
@test g(A) == vec(maximum(A, dims=1))
A0 = zero(A);
A0[findmax(A, dims=1)[2]] .= 1
@test A0 ≈ Tracker.gradient(sum∘g, A)[1]
@test g(cu(A)) isa CuArray
@test g(cu(A)) ≈ cu(g(A))
@test cu(A0) ≈ Tracker.gradient(sum∘g, cu(A))[1]

# functions
h(A) = @tullio H[j] := exp(A[i,j]) / log(A[i,j])
@test h(cu(A)) isa CuArray
@test h(cu(A)) ≈ cu(h(A))
A1 = Tracker.gradient(sum∘h, A)[1]
@test cu(A1) ≈ Tracker.gradient(sum∘h, cu(A))[1]

#= # broken by https://github.com/mcabbott/Tullio.jl/pull/31
# scalar
@tullio s := cu(A)[i,j]^2
@test s ≈ sum(abs2, A)
@tullio s += cu(B)[i,j]^2
@test s ≈ sum(abs2, A) + sum(abs2, B)
=#
