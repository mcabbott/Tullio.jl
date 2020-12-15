#===== stuff =====#

t2 = time()

@testset "parsing all the things" begin include("parsing.jl") end

@testset "tests from Einsum.jl" begin include("einsum.jl") end

@info @sprintf("Basic tests took %.1f seconds", time()-t2)

@testset "internal pieces" begin include("utils.jl") end

@testset "matrix multiplication" begin
    # size 200 is big enough to test block_halves even with MINIBLOCK = 64^3
    @testset "size $N, elements $T" for N in [2, 20, 200], T in [1:99, Float32, Float64, ComplexF64]
        for f in [identity, adjoint]
            A = f(rand(T, N,N));
            B = f(rand(T, N,N));
            @test A * B ≈ @tullio C[i,k] := A[i,j] * B[j,k]
        end
        if N < 200
            X = rand(T, N,N+1);
            Y = rand(T, N+1,N+2);
            Z = rand(T, N+2,N+1);
            @test X * Y * Z ≈ @tullio C[a,d] := X[a,b] * Y[b,c] * Z[c,d]
        end
    end
    @testset "@allocated" begin
        m!(C,A,B) = @tullio C[i,k] = A[i,j] * B[j,k] threads=false
        C1, A1, B1 = rand(4,4), rand(4,4), rand(4,4)
        @allocated m!(C1, A1, B1)
        @test 0 == @allocated m!(C1, A1, B1)
    end
end

#===== Tracker =====#

t3 = time()
using Tracker

GRAD = :Tracker
_gradient(x...) = Tracker.gradient(x...)

@tullio grad=Base
@testset "gradients: Tracker + DiffRules" begin include("gradients.jl") end

@tullio grad=Dual
@testset "gradients: Tracker + ForwardDiff" begin include("gradients.jl") end

@info @sprintf("Tracker tests took %.1f seconds", time()-t3)
