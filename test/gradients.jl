
using Tullio, Test, ForwardDiff
@info "now loading Zygote..."
t1 = time()
using Zygote
t2 = round(time()-t1, digits=1)
@info "... done! Only took $t2 seconds"

# using Tullio: LoopVectorization.@avx, storage_type

Tullio.VERBOSE[] = true
Tullio.VERBOSE[] = false

unfill(x) = x  # gradient of sum returns a FillArrays.Fill
Zygote.@adjoint function unfill(x)
    back(dx::Zygote.FillArrays.Fill) = (collect(dx),)
    back(dx) = (dx,)
    x, back
end
#=
using Tracker
unfill(x::TrackedArray) = Tracker.track(unfill, x)
Tracker.@grad unfill(x) = unfill(Tracker.data(x)), dx -> (collect(dx),)
=#

Tullio.GRAD[] = :Base

using ForwardDiff: partials # scope issue?

# for avx in [false, true]
#     Tullio.AVX[] = avx # this doesn't work in tests! wtf?
# @testset "zygote, AVX[] = $avx" begin
#     @info "AVX[] = $avx"
@testset "zygote" begin

    # simple
    f(x) = @tullio y[i] := 2*x[i]
    @test Zygote.gradient(sum∘unfill∘f, rand(3)) == ([2,2,2],)

    f(x) = @tullio y[i] := 2*x[i] + i # integer promotion
    @test Zygote.gradient(sum∘unfill∘f, rand(3)) == ([2,2,2],)

    # two contributions
    g(x) = @tullio y[i, j] := 1 * x[i] + 1000 * x[j]
    mat = [1 1 3; 1 1 5; 7 7 7]
    g_fd = ForwardDiff.gradient(x -> sum(mat .* g(x)), rand(3))
    @test g_fd ≈ Zygote.gradient(x -> sum(mat .* g(x)), rand(3))[1]

    # two arrays
    h(x,y) = @tullio z[i] := x[i,j] + y[j,i]
    @test Zygote.gradient(sum∘unfill∘h, rand(2,3), rand(3,2)) == (ones(2,3), ones(3,2))

    # nontrivial function
    f(x,y) = @tullio z[i] := log(x[i,j]) / y[j,i]
    r_x, r_y = rand(2,3), rand(3,2)
    fx = ForwardDiff.gradient(x -> sum(f(x, r_y)), r_x)
    fy = ForwardDiff.gradient(y -> sum(f(r_x, y)), r_y)
    @test fx ≈ Zygote.gradient(sum∘unfill∘f, r_x, r_y)[1]
    @test fy ≈ Zygote.gradient(sum∘unfill∘f, r_x, r_y)[2]

end
# end

