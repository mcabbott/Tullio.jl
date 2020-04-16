
using Test, Printf

t0 = @elapsed using Tullio
@info @sprintf("Loading Tullio took %.1f seconds", t0)

Tullio.BLOCK[] = 20 # use threading even on very small arrays

#===== stuff =====#

t1 = time()

@testset "parsing all the things" begin include("parsing.jl") end

@testset "tests from Einsum.jl" begin include("einsum.jl") end

@info @sprintf("Basic tests took %.1f seconds", time()-t1)

@testset "internal pieces" begin include("utils.jl") end

#===== Tracker =====#

t2 = @elapsed using Tracker
@info @sprintf("Loading Tracker took %.1f seconds", t2)

unfill(x) = x  # gradient of sum returns a FillArrays.Fill
unfill(x::TrackedArray) = Tracker.track(unfill, x)
Tracker.@grad unfill(x) = unfill(Tracker.data(x)), dx -> (collect(dx),)

_gradient(x...) = Tracker.gradient(x...)
@testset "backward gradients: Tracker" begin include("gradients.jl") end

#===== Yota =====#
#=
t3 = @elapsed using Yota
@info @sprintf("Loading Yota took %.1f seconds", t3)
# Yota.@diffrule unfill(x) x collect(ds)

_gradient(x...) = Yota.grad(x...)[2]
@testset "backward gradients: Yota" begin include("gradients.jl") end
=#
#===== Zygote =====#

# @info "now loading Zygote..."
# t0 = time()
t5 = @elapsed using Zygote
@info @sprintf("Loading Zygote took %.1f seconds", t5)
# @info @sprintf("Loading Zygote took %.1f or perhaps %.1f seconds", t5, time()-t0)
# @info @sprintf("  ... done! Only took %.1f seconds!", t5)

Zygote.@adjoint unfill(x) = x, dx -> (collect(dx),)

_gradient(x...) = Zygote.gradient(x...)
@testset "backward gradients: Zygote" begin include("gradients.jl") end

#===== ReverseDiff =====#

t4 = @elapsed using ReverseDiff
@info @sprintf("Loading ReverseDiff took %.1f seconds", t4)

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
