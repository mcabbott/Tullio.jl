using Tullio, Test
using Metal, KernelAbstractions
using Tracker, ForwardDiff
@tullio grad=Base

# matmul
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

A = rand(3,40); B = rand(40,500);
@test A * B ≈ mul(A, B)
@test mtl(A * B) ≈ mul(mtl(A), mtl(B))

# gradient
# FIXME: Broken I think
ΔA = Tracker.gradient((A,B) -> sum(mul(A, B)), A, B)[1]
@test ΔA ≈ ones(3,500) * B'
@test mtl(ΔA) ≈ Tracker.gradient((A,B) -> sum(mul(A, B)), mtl(A), mtl(B))[1]

# shifts
@tullio D[i,j] := A[i,j+k]  k in 0:10
@test axes(D) == (1:3, 1:30)
@tullio D_dev[i,j] := mtl(A)[i,j+k]  k in 0:10
@test D_dev isa MtlArray
@test D_dev ≈ mtl(D)

# product
@tullio (*) F[j] := A[i,j]
@test F ≈ vec(prod(A, dims=1))
@tullio (*) F_dev[j] := mtl(A)[i,j]
@test F_dev ≈ mtl(F)

# maximum
g(A) = @tullio (max) G[j] := A[i,j]
@test g(A) == vec(maximum(A, dims=1))
A0 = zero(A);
A0[findmax(A, dims=1)[2]] .= 1
@test A0 ≈ Tracker.gradient(sum∘g, A)[1]
@test g(mtl(A)) isa MtlArray
@test g(mtl(A)) ≈ mtl(g(A))
@test mtl(A0) ≈ Tracker.gradient(sum∘g, mtl(A))[1]

# functions
h(A) = @tullio H[j] := exp(A[i,j]) / log(A[i,j])
@test h(mtl(A)) isa MtlArray
@test h(mtl(A)) ≈ mtl(h(A))
A1 = Tracker.gradient(sum∘h, A)[1]
@test mtl(A1) ≈ Tracker.gradient(sum∘h, mtl(A))[1]

A, B, C = Metal.rand(2, 2, 2), Metal.rand(2, 2), Metal.rand(2, 2, 2);
@tullio A[k,i,a] = tanh(B[i,a] + C[k,i,a])
A2 = similar(A)
struct Bee{T}; B::T; end
B2 = Bee(B)
@test A ≈ @tullio A2[k,i,a] = tanh(B2.B[i,a] + C[k,i,a])
