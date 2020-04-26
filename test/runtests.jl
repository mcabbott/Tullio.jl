
using Test, Printf

@info "Testing with $(Threads.nthreads()) threads"

t1 = @elapsed using Tullio
@info @sprintf("Loading Tullio took %.1f seconds", t1)

Tullio.BLOCK[] = 32 # use threading even on small arrays
Tullio.MINIBLOCK[] = 32

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
end

#===== Tracker =====#

using ForwardDiff

t3 = time()
using Tracker

unfill(x) = x  # gradient of sum returns a FillArrays.Fill

_gradient(x...) = Tracker.gradient(x...)
@testset "backward gradients: Tracker" begin include("gradients.jl") end

@info @sprintf("Tracker tests took %.1f seconds", time()-t3)

#===== Yota =====#
#=
t4 = time()
using Yota

_gradient(x...) = Yota.grad(x...)[2]
@testset "backward gradients: Yota" begin include("gradients.jl") end

@info @sprintf("Yota tests took %.1f seconds", time()-t4)
=#
#===== Zygote =====#

t5 = time()
using Zygote

_gradient(x...) = Zygote.gradient(x...)
@testset "backward gradients: Zygote" begin include("gradients.jl") end

@info @sprintf("Zygote tests took %.1f seconds", time()-t5)

#===== ReverseDiff =====#

t6 = time()
using ReverseDiff

_gradient(f, xs...) = ReverseDiff.gradient(f, xs)
@testset "backward gradients: ReverseDiff" begin include("gradients.jl") end

@info @sprintf("ReverseDiff tests took %.1f seconds", time()-t6)

#===== done! =====#
