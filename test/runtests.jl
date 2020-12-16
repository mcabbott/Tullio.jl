
using Test, Printf

t1 = @elapsed using Tullio
@info @sprintf("Loading Tullio took %.1f seconds", t1)

import InteractiveUtils
InteractiveUtils.versioninfo(; verbose = true)

@info "Testing with $(Threads.nthreads()) threads"
if Threads.nthreads() > 1 # use threading even on small arrays
    Tullio.BLOCK[] = 32
    Tullio.TILE[] = 32
end

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

#===== KernelAbstractions =====#

t4 = time()
using KernelAbstractions

@testset "KernelAbstractions + gradients" begin
    A = (rand(3,4));
    B = (rand(4,5));
    @tullio C[i,k] := A[i,j] * B[j,k]  threads=false  # verbose=2
    @test C ≈ A * B

    @tullio threads=false # else KernelAbstractions CPU kernels not used
    include("gradients.jl")
    @tullio threads=true

    for sy in Tullio.SYMBOLS
        @test !isdefined(@__MODULE__, sy)
    end
end

using CUDA

if CUDA.has_cuda_gpu()
    @info "===== found a GPU, starting CUDA tests ====="
    @testset "===== CUDA tests on GPU =====" begin
        include("cuda.jl")
    end
end

@info @sprintf("KernelAbstractions tests took %.1f seconds", time()-t4)

@tullio cuda=false

#===== Zygote =====#

if VERSION < v"1.6-" # Zygote isn't working on 1.6

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

        # with finaliser
        g7 = _gradient(x -> real(sum(sqrt.(sum(exp.(x), dims=2)))), x0 .+ x0')[1]
        g7i = _gradient(x -> imag(sum(sqrt.(sum(exp.(x), dims=2)))), x0 .+ x0')[1]
        @test g7 ≈ _gradient(x -> real(sum(@tullio y[i] := sqrt <| exp(x[i,j]) )), x0 .+ x0')[1]
        @test g7i ≈ _gradient(x -> imag(sum(@tullio y[i] := sqrt <| exp(x[i,j]) )), x0 .+ x0')[1]

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

end # VERSION

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

#===== LoopVectorization =====#

t8 = time()
using LoopVectorization
using VectorizationBase

@static if Base.VERSION >= v"1.5"
    const Vec = VectorizationBase.Vec
else
    const Vec = VectorizationBase.SVec
end

@testset "LoopVectorization onlyone" begin
    ms = mask(Val(8), 2) # Mask{8,Bool}<1, 1, 0, 0, 0, 0, 0, 0>
    sv = Vec{4,Int}(1,2,3,4) # Vec{4,Int64}<1, 2, 3, 4>

    # preliminaries:
    @test Tullio.allzero(sv) === false
    @test Tullio.allzero(zero(sv)) === true

    @test Tullio.anyone(ms) === true

    # the main function:
    @test Tullio.onlyone(false, 0) === false
    @test Tullio.onlyone(true, 0) === true
    @test Tullio.onlyone(true, 1) === false

    # @test Tullio.onlyone(ms, 0) === Mask{UInt8}(0x02)
    @test Tullio.onlyone(ms, 0).u == 0x02
    # @test Tullio.onlyone(ms, sv) === Mask{UInt8}(0x00)
    @test Tullio.onlyone(ms, sv).u == 0x00
    # @test Tullio.onlyone(ms, zero(sv)) === Mask{UInt8}(0x02)
    @test Tullio.onlyone(ms, zero(sv)).u == 0x02
end

@testset "parsing + LoopVectorization" begin include("parsing.jl") end

using Tracker
GRAD = :Tracker
_gradient(x...) = Tracker.gradient(x...)

@tullio grad=Base
@testset "gradients: Tracker + DiffRules + LoopVectorization" begin include("gradients.jl") end

@tullio grad=Dual
@testset "gradients: Tracker + ForwardDiff + LoopVectorization" begin include("gradients.jl") end

@info @sprintf("LoopVectorization tests took %.1f seconds", time()-t8)

@tullio avx=false

#===== TensorOperations =====#

t9 = time()
using TensorOperations

using Tracker
GRAD = :Tracker
_gradient(x...) = Tracker.gradient(x...)

@tullio grad=Base
@testset "gradients: Tracker + TensorOperations" begin include("gradients.jl") end

if VERSION < v"1.6-" # Zygote isn't working on 1.6

using Zygote
GRAD = :Zygote
_gradient(x...) = Zygote.gradient(x...)

@tullio grad=Base
@testset "gradients: Zygote + TensorOperations" begin include("gradients.jl") end

@testset "complex gradients with TensorOperations" begin

    x0 = [1 2; 3 4] .+ [5im 0; 7im -8im]

    @testset "analytic" begin

        g1 = _gradient(x -> real(sum(x * x)), x0)[1]
        g1i = _gradient(x -> imag(sum(x * x)), x0)[1]
        @test g1 ≈ _gradient(x -> real(sum(@tullio y[i,j] := x[i,k] * x[k,j])), x0)[1]
        @test g1i ≈ _gradient(x -> imag(sum(@tullio y[i,j] := x[i,k] * x[k,j])), x0)[1]

    end
    @testset "non-analytic" begin

        g2 = _gradient(x -> real(sum(x * x')), x0)[1]
        g2i = _gradient(x -> imag(sum(x * x')), x0)[1] # zero
        @test_broken g2 ≈ _gradient(x -> real(sum(@tullio y[i,j] := x[i,k] * conj(x[j,k]))), x0)[1]
        @test_broken g2i ≈ _gradient(x -> imag(sum(@tullio y[i,j] := x[i,k] * conj(x[j,k]))), x0)[1]

    end
end

end # VERSION

@testset "parsing + TensorOperations" begin include("parsing.jl") end # testing correct fallback

@info @sprintf("TensorOperations tests took %.1f seconds", time()-t9)

#===== done! =====#
