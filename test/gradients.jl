
using Tullio, Test

# using Tullio: LoopVectorization.@avx, storage_type
#=
using Tracker, ForwardDiff
unfill(x) = x  # gradient of sum returns a FillArrays.Fill
unfill(x::TrackedArray) = Tracker.track(unfill, x)
Tracker.@grad unfill(x) = unfill(Tracker.data(x)), dx -> (collect(dx),)
_gradient = Tracker.gradient
=#

@tullio grad=Base

@testset "symbolic" begin

    # simple
    f(x) = @tullio y[i] := 2*x[i]
    @test _gradient(sum∘unfill∘f, rand(3))[1] == [2,2,2]

    f(x) = @tullio y[i] := 2*x[i] + i # integer promotion
    @test _gradient(sum∘unfill∘f, rand(3)) == ([2,2,2],)

    # two contributions
    g(x) = @tullio y[i, j] := 1 * x[i] + 1000 * x[j]
    mat = [1 1 3; 1 1 5; 7 7 7]
    g_fd = ForwardDiff.gradient(x -> sum(mat .* g(x)), rand(3))
    @test g_fd ≈ _gradient(x -> sum(mat .* g(x)), rand(3))[1]

    # two arrays
    h(x,y) = @tullio z[i] := x[i,j] + y[j,i]
    @test _gradient(sum∘unfill∘h, rand(2,3), rand(3,2)) == (ones(2,3), ones(3,2))

    # nontrivial function
    f(x,y) = @tullio z[i] := log(x[i,j]) / y[j,i]
    r_x, r_y = rand(2,3), rand(3,2)
    fx = ForwardDiff.gradient(x -> sum(f(x, r_y)), r_x)
    fy = ForwardDiff.gradient(y -> sum(f(r_x, y)), r_y)
    @test fx ≈ _gradient(sum∘unfill∘f, r_x, r_y)[1]
    @test fy ≈ _gradient(sum∘unfill∘f, r_x, r_y)[2]

end

using ForwardDiff
using ForwardDiff: partials # scope issue?

@testset "dual" begin

    # simple
    f(x) = @tullio grad=Dual y[i] := 2*x[i]
    @test _gradient(sum∘unfill∘f, rand(3))[1] == [2,2,2]

    f(x) = @tullio grad=Dual y[i] := 2*x[i] + i # integer promotion
    @test_skip _gradient(sum∘unfill∘f, rand(3)) == ([2,2,2],) # no method matching vconvert(::Type{NTuple{4,VecElement{ForwardDiff.Dual{Nothing,Any,1}}}}, ::Int64)

    # two contributions
    g(x) = @tullio grad=Dual y[i, j] := 1 * x[i] + 1000 * x[j]
    mat = [1 1 3; 1 1 5; 7 7 7]
    g_fd = ForwardDiff.gradient(x -> sum(mat .* g(x)), rand(3))
    @test_skip g_fd ≈ _gradient(x -> sum(mat .* g(x)), rand(3))[1] # no method matching zero(::Type{Any})

    # two arrays
    h(x,y) = @tullio grad=Dual z[i] := x[i,j] + y[j,i]
    @test _gradient(sum∘unfill∘h, rand(2,3), rand(3,2)) == (ones(2,3), ones(3,2))

    # nontrivial function
    f(x,y) = @tullio grad=Dual z[i] := log(x[i,j]) / y[j,i]
    r_x, r_y = rand(2,3), rand(3,2)
    fx = ForwardDiff.gradient(x -> sum(f(x, r_y)), r_x)
    fy = ForwardDiff.gradient(y -> sum(f(r_x, y)), r_y)
    @test fx ≈ _gradient(sum∘unfill∘f, r_x, r_y)[1]
    @test fy ≈ _gradient(sum∘unfill∘f, r_x, r_y)[2]

end
