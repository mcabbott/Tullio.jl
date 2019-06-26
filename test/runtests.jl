using Test
using Tullio, Einsum

@testset "cast" begin

    A = rand(2,3)
    V = rand(3)

    @einsum C1[j,i] := A[i,j] + V[j]
    @test C1 == @tullio C2[j,i] := A[i,j] + V[j]
    @test C1 == @tullio C3[j,i] := A[i,j] + V[j] {thread}
    @test C1 == @tullio C4[j,i] := A[i,j] + V[j] {tile}
    @test C1 == @tullio C5[j,i] := A[i,j] + V[j] {tile(10),i,j}

    D1,D2,D3,D4,D5 = similar(C1), similar(C1), similar(C1), similar(C1), similar(C1)
    @test C1 == @einsum D1[j,i] = A[i,j] + V[j]
    @test C1 == @tullio D2[j,i] = A[i,j] + V[j]
    @test C1 == @tullio D3[j,i] = A[i,j] + V[j] {thread}
    @test C1 == @tullio D4[j,i] := A[i,j] + V[j] {tile}
    @test C1 == @tullio D5[j,i] := A[i,j] + V[j] {tile(10),i,j}

end
@testset "reduce" begin

    A = rand(2,10)
    B = rand(3,10)

    @einsum Z1[i,k] := A[i,j] + B[k,j]/2
    @test Z1 == @tullio Z2[i,k] := A[i,j] + B[k,j]/2
    @test Z1 == @tullio Z3[i,k] := A[i,j] + B[k,j]/2 {thread}
    @test Z1 == @tullio Z4[i,k] := A[i,j] + B[k,j]/2 {tile,i,j,k}

    @test Z1 == @tullio Z5[i,k] := A[i,j] + B[k,j]/2 (+,j)
    @test Z1 == @tullio Z6[i,k] := A[i,j] + B[k,j]/2 (+,unroll,j)
    @test Z1 == @tullio Z7[i,k] := A[i,j] + B[k,j]/2 (+,unroll(3),j)
    @test Z1 == @tullio Z8[i,k] := A[i,j] + B[k,j]/2 (+,unroll,j) {tile,i,j,k}
    @test Z1 == @tullio Z9[i,k] := A[i,j] + B[k,j]/2 (+,unroll,j) {tile,i,k,thread}

end
@testset "types" begin

    A = reshape(1:12,3,4)
    @tullio B[i,j] := A[j,i]/2
    @test eltype(B) == Float64

    C = rand(Float32, 3,4)
    @tullio D[i,j] := log(C[j,i])
    @test eltype(D) == Float32

    @tullio E[i,j] := C[j,i] + im
    @test eltype(E) == Complex{Float32}

end
@testset "special" begin

    A = rand(2,3)

    @test reshape(A,2,1,3) == @tullio B[i,_,j] := A[i,j]

    @test A[:,3] == @tullio C[i] := A[i,3]
    i=99; k=3
    @test A[:,3] == @tullio C[i] := A[i,$k]
    @test i==99

    @tullio D1[i] := A[i,j] (+,j)
    @test D1 == @tullio D2[i] := sum(A[i,:])

    @test A' == @tullio E[a,a'] := A[a′,a]

end
