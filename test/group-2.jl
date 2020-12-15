#===== KernelAbstractions =====#

t4 = time()
using KernelAbstractions

using Tracker

GRAD = :Tracker
_gradient(x...) = Tracker.gradient(x...)

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

if is_buildkite
    # If we are on Buildkite, we should assert that we have a CUDA GPU available
    @test CUDA.has_cuda_gpu()
end

if CUDA.has_cuda_gpu()
    @info "===== found a GPU, starting CUDA tests ====="
    @testset "===== CUDA tests on GPU =====" begin
        include("cuda.jl")
    end
end

@info @sprintf("KernelAbstractions tests took %.1f seconds", time()-t4)

@tullio cuda=false
