
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

t3 = time()
using Tracker

GRAD = :Tracker
_gradient(x...) = Tracker.gradient(x...)

@tullio grad=Base
@testset "gradients: Tracker + DiffRules" begin include("gradients.jl") end

@tullio grad=Dual
@testset "gradients: Tracker + ForwardDiff" begin include("gradients.jl") end

@info @sprintf("Tracker tests took %.1f seconds", time()-t3)

#===== KernelAbstractions =====#

t4 = time()
using KernelAbstractions

@testset "KernelAbstractions + gradients" begin
    A = (rand(3,4));
    B = (rand(4,5));
    @tullio C[i,k] := A[i,j] * B[j,k]  threads=false  # verbose=2
    @test C ≈ A * B

    @tullio threads=false
    include("gradients.jl")
    @tullio threads=true
end

@info @sprintf("KernelAbstractions tests took %.1f seconds", time()-t4)

#===== Zygote =====#

t5 = time()
using Zygote

GRAD = :Zygote
_gradient(x...) = Zygote.gradient(x...)

@tullio grad=Base
@testset "gradients: Zygote + DiffRules" begin include("gradients.jl") end

@tullio grad=Dual
@testset "gradients: Zygote + ForwardDiff" begin include("gradients.jl") end

@tullio grad=Base
@testset "complex gradients with Zygote" begin

    x0 = [1,2,3] .+ [5im, 0, -11im]
    # y0 = rand(Int8,3) .+ im .* rand(Int8,3) .+ 0.0
    @testset "analytic" begin

        g1 = _gradient(x -> real(sum(x)), x0)[1]
        g1i = _gradient(x -> imag(sum(x)), x0)[1]
        @test g1 ≈ _gradient(x -> real(@tullio y := x[i]), x0)[1]
        @test g1i ≈ _gradient(x -> imag(@tullio y := x[i]), x0)[1]

        g2 = _gradient(x -> real(sum(exp, x)), x0)[1]
        g2i = _gradient(x -> imag(sum(exp, x)), x0)[1]
        @test g2 ≈ _gradient(x -> real(@tullio y := exp(x[i])), x0)[1]
        @test g2i ≈ _gradient(x -> imag(@tullio y := exp(x[i])), x0)[1]

        g3 = _gradient(x -> real(sum(1 ./ (x.+im).^2)), x0)[1]
        g3i = _gradient(x -> imag(sum(1 ./ (x.+im).^2)), x0)[1]
        @test g3 ≈ _gradient(x -> real(@tullio y := 1/(x[i] + im)^2), x0)[1]
        @test g3 ≈ _gradient(x -> real(@tullio y := inv(x[i] + im)^2), x0)[1]
        @test g3i ≈ _gradient(x -> imag(@tullio y := 1/(x[i] + im)^2), x0)[1]
        @test g3i ≈ _gradient(x -> imag(@tullio y := inv(x[i] + im)^2), x0)[1]

    end
    @testset "non-analytic" begin

        g4 = _gradient(x -> real(sum(x * x')), x0)[1]
        g4i = _gradient(x -> imag(sum(x * x')), x0)[1] # zero!
        @test_broken g4 ≈ _gradient(x -> real(@tullio y := x[i] * conj(x[j])), x0)[1]
        @test_broken g4i ≈ _gradient(x -> imag(@tullio y := x[i] * conj(x[j])), x0)[1]
        @test_broken g4 ≈ _gradient(x -> real(@tullio y := x[i] * adjoint(x[j])), x0)[1]
        @test_broken g4i ≈ _gradient(x -> imag(@tullio y := x[i] * adjoint(x[j])), x0)[1]

        g5 = _gradient(x -> real(sum(abs2.(x .+ 2 .+ im))), x0)[1]
        g5i = _gradient(x -> imag(sum(abs2.(x .+ 2 .+ im))), x0)[1] # zero!
        @test_broken g5 ≈ _gradient(x -> real(@tullio y := abs2(x[i] + 2 + im)), x0)[1]
        @test_broken g5i ≈ _gradient(x -> real(@tullio y := abs2(x[i] + 2 + im)), x0)[1]

        g6 = _gradient(x -> real(sum(abs.(x.^3))), x0)[1]
        g6i = _gradient(x -> imag(sum(abs.(x.^3))), x0)[1] # zero!
        @test_broken g6 ≈ _gradient(x -> real(@tullio y := abs(x[i]^3)), x0)[1]
        @test_broken g6i ≈ _gradient(x -> real(@tullio y := abs(x[i]^3)), x0)[1]

    end
end

@info @sprintf("Zygote tests took %.1f seconds", time()-t5)

#===== ReverseDiff =====#
#=
t6 = time()
using ReverseDiff

GRAD = :ReverseDiff
_gradient(x...) = ReverseDiff.gradient(x...) # ??

@tullio grad=Base
@testset "gradients: ReverseDiff + DiffRules" begin include("gradients.jl") end

@tullio grad=Dual
@testset "gradients: ReverseDiff + ForwardDiff" begin include("gradients.jl") end

@info @sprintf("ReverseDiff tests took %.1f seconds", time()-t6)
=#

#===== Yota =====#
#=
t7 = time()
using Yota

GRAD = :Yota
_gradient(x...) = Yota.grad(x...)[2]

@tullio grad=Base
@testset "gradients: Yota + DiffRules" begin include("gradients.jl") end

@tullio grad=Dual
@testset "gradients: Yota + ForwardDiff" begin include("gradients.jl") end

@info @sprintf("Yota tests took %.1f seconds", time()-t7)
=#

#===== done! =====#
