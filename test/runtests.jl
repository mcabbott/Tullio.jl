using Test, Printf

t1 = @elapsed using Tullio
@info @sprintf("Loading Tullio took %.1f seconds", t1)

@info "Testing with $(Threads.nthreads()) threads"
if Threads.nthreads() > 1 # use threading even on small arrays
    Tullio.BLOCK[] = 32
    Tullio.TILE[] = 32
end

is_buildkite = parse(Bool, get(ENV, "BUILDKITE", "false"))
if is_buildkite
    test_group = "2" # if this is Buildkite, we only run group 2
else
    test_group = get(ENV, "TULLIO_TEST_GROUP", "all")
end
@info "" test_group is_buildkite

if test_group in ["all", "1"]
    include("group-1.jl")
end
if test_group in ["all", "2"]
    include("group-2.jl")
end
if test_group in ["all", "3"]
    include("group-3.jl")
end
<<<<<<< HEAD
=======

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

if isdefined(LoopVectorization, :SVec) # version 0.8, for Julia ⩽1.5
    using LoopVectorization.VectorizationBase: SVec, Mask
else # version 0.9, supports Julia 1.6
    using LoopVectorization.VectorizationBase: Vec, Mask
    SVec{N,T} = Vec{N,T}
end

@testset "LoopVectorization onlyone" begin
    ms = Mask{4,UInt8}(0x03); # Mask{4,Bool}<1, 1, 0, 0>
    sv = SVec{4,Int}(1,2,3,4) # SVec{4,Int64}<1, 2, 3, 4>

    # preliminaries:
    @test Tullio.allzero(sv) === false
    @test Tullio.allzero(zero(sv)) === true

    @test Tullio.anyone(ms) === true

    # the main function:
    @test Tullio.onlyone(false, 0) === false
    @test Tullio.onlyone(true, 0) === true
    @test Tullio.onlyone(true, 1) === false

    @test Tullio.onlyone(ms, 0) === Mask{4}(0x02)
    # @test Tullio.onlyone(ms, 0).u == 0x02
    @test Tullio.onlyone(ms, sv) === Mask{4}(0x00)
    # @test Tullio.onlyone(ms, sv).u == 0x00
    @test Tullio.onlyone(ms, zero(sv)) === Mask{4}(0x02)
    # @test Tullio.onlyone(ms, zero(sv)).u == 0x02
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
>>>>>>> tweak tests of onlyone(Mask ,...)
