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
