using Test
using Tullio, Einsum

@testset "basics" begin

    A = rand(2,3)
    B = rand(3,4)
    V = rand(3)

    @einsum C1[j,i] := A[i,j] + V[j]
    @tullio C2[j,i] := A[i,j] + V[j]
    @moltullio C3[j,i] := A[i,j] + V[j]
    @test C1 == C2 == C3

    C4,C5,C6 = similar(C1), similar(C1), similar(C1)
    @einsum C4[j,i] = A[i,j] + V[j]
    @tullio C5[j,i] = A[i,j] + V[j]
    @moltullio C6[j,i] = A[i,j] + V[j]
    @test C4 == C5 == C6

end
