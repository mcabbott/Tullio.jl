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
