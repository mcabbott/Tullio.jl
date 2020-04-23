
using Test, Printf

@info "Testing with $(Threads.nthreads()) threads"

t0 = @elapsed using Tullio
@info @sprintf("Loading Tullio took %.1f seconds", t0)

Tullio.BLOCK[] = 32 # use threading even on small arrays
Tullio.MINIBLOCK[] = 32

#===== stuff =====#

t1 = time()

@testset "parsing all the things" begin include("parsing.jl") end

@testset "tests from Einsum.jl" begin include("einsum.jl") end

@info @sprintf("Basic tests took %.1f seconds", time()-t1)

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

t3 = @elapsed using Tracker
@info @sprintf("Loading Tracker took %.1f seconds", t3)

unfill(x) = x  # gradient of sum returns a FillArrays.Fill
unfill(x::TrackedArray) = Tracker.track(unfill, x)
Tracker.@grad unfill(x) = unfill(Tracker.data(x)), dx -> (collect(dx),)

_gradient(x...) = Tracker.gradient(x...)
@testset "backward gradients: Tracker" begin include("gradients.jl") end

#===== Yota =====#
#=
t4 = @elapsed using Yota
@info @sprintf("Loading Yota took %.1f seconds", t4)
# Yota.@diffrule unfill(x) x collect(ds)

_gradient(x...) = Yota.grad(x...)[2]
@testset "backward gradients: Yota" begin include("gradients.jl") end
=#
#===== Zygote =====#

t5 = @elapsed using Zygote
@info @sprintf("Loading Zygote took %.1f seconds", t5)

Zygote.@adjoint unfill(x) = x, dx -> (collect(dx),)

_gradient(x...) = Zygote.gradient(x...)
@testset "backward gradients: Zygote" begin include("gradients.jl") end

#===== ReverseDiff =====#

t6 = @elapsed using ReverseDiff
@info @sprintf("Loading ReverseDiff took %.1f seconds", t6)

_gradient(f, xs...) = ReverseDiff.gradient(f, xs)
@testset "backward gradients: ReverseDiff" begin include("gradients.jl") end

#===== done! =====#

#=
using Yota
g(x) = x^2
dg(dy, x) = (@show dy x; x + dy + im)
Yota.@diffrule g(x) x dg(dy, x)
Yota.grad(g, 1)[2][1]

f(x) = x^3
df(dy, x) = (@show dy x; x + dy + im)
Yota.@diffrule f(args...) x df(dy, args...)
Yota.grad(f, 1)[2][1]



=#
