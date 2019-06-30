using Test
using Tullio, Einsum, LinearAlgebra, OffsetArrays

@info "running tests with $(Threads.nthreads()) threads, Julia $VERSION"

@testset "cast" begin

    A = rand(7,10)
    V = rand(10)

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
    @test C1 == @tullio D5[j,i] := A[i,j] + V[j] {tile(5^2),i,j}

end
@testset "reduce" begin

    A = rand(10,10)
    B = rand(10,10)

    @einsum Z1[i,k] := A[i,j] + B[k,j]/2
    @test Z1 == @tullio Z2[i,k] := A[i,j] + B[k,j]/2
    @test Z1 == @tullio Z3[i,k] := A[i,j] + B[k,j]/2 {thread}
    @test Z1 == @tullio Z4[i,k] := A[i,j] + B[k,j]/2 {tile,i,k}
    @test Z1 == @tullio Z4[i,k] := A[i,j] + B[k,j]/2 {tile(100),i,j,k}

    @test Z1 == @tullio Z5[i,k] := A[i,j] + B[k,j]/2 (+,j)
    @test Z1 == @tullio Z6[i,k] := A[i,j] + B[k,j]/2 (+,unroll,j)
    @test Z1 == @tullio Z7[i,k] := A[i,j] + B[k,j]/2 (+,unroll(3),j)
    @test Z1 == @tullio Z8[i,k] := A[i,j] + B[k,j]/2 (+,unroll,j) {tile(100),i,j,k}
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
    k=3
    @test A[:,3] == @tullio C[i] := A[i,$k]

    i=99;
    @tullio C[i] := A[i,3]
    @test i==99

    @tullio D1[i] := A[i,j] (+,j)
    @test D1 == @tullio D2[i] := sum(A[i,:])

    @test A' == @tullio E[a,a'] := A[a′,a]

    @test Diagonal(C) == @tullio F[d,d] := C[d] {zero}

end
@testset "cyclic" begin

    V = 1:4
    @test [2,3,4,1] == @tullio A[i] := V[i+1] {cyclic}
    @test [4,1,2,3] == @tullio A[i] := V[i-1] {cyclic}
    @test [4,3,2,1] == @tullio A[i] := V[1-i] {cyclic}
    # @test [4,3,2,1] == @tullio A[1-i] := V[i] {cyclic}
    @test [10,10,10,10] == @tullio A[i] := V[i+j] (+,j) {cyclic}

    M = [i + 100j for i=1:4, j=1:4]
    @test circshift(M, (-1,0)) == @tullio B[i,j] := M[i+1,j] {cyclic}
    @tullio B[i,j] := M[i+k,j-k] (+,k) {cyclic}
    @test all(B .== 1010)

    X = [1,1,0,0]
    Y = [1,-1,0,0]
    @test [1,0,-1,0] == @tullio C[i] := X[i+k] * Y[k] {cyclic}

end
@testset "shifted" begin

    V = 1:4
    @test [2,3,4] == @tullio A[i] := V[i+1]

    @tullio A[α] := V[α-1]  {offset}
    @test A isa OffsetArray
    @test axes(A,1) == 2:5
    @test A[2] == 1

    @tullio B[β] := V[1-β]  {offset}
    @test B isa OffsetArray
    @test axes(B,1) == -3:0
    @test B[-2] == 3

end
@static if false # Base.find_package("CuArrays") !== nothing

    @info "found CuArrays, starting GPU tests"

    using CuArrays
    using CUDAnative
    CuArrays.allowscalar(false)

    @testset "gpu cast" begin

        V = rand(10)
        cV = cu(V)
        W, cW = similar(V), similar(cV)

        @test V == @tullio W[i] = V[i] {gpu}
        @test cV == @tullio cW[i] = cV[i] {gpu}

        M = rand(10,10)
        cM = cu(M)
        N, cN = similar(M), similar(cM)

        @test M' == @tullio N[i,j] = M[j,i] {gpu}
        @test_broken cM' == @tullio cN[i,j] = cM[j,i] {gpu} # broken by allowscalar(false)

    end
    @testset "gpu reduce" begin

        V = rand(10)
        cV = cu(V)
        W, cW = similar(V), similar(cV)

        M = rand(10,10)
        cM = cu(M)

        @tullio W[i] = M[i,j] + V[j] {gpu}
        @test_broken cu(W) == @tullio cW[i] = cM[i,j] + cV[j] {gpu}

    end
else
    @info "did not find CuArrays, omitting GPU tests"
end
@testset "errors" begin

    @test true

end

