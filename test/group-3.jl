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
