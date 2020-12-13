#=
This file is run several times
* with grad=Base vs grad=Dual
* with Tracker, Zygote
* using KernelAbstractions, LoopVectorization, TensorCast
=#

using Tullio, Test, ForwardDiff, Random
# using Tracker; _gradient(x...) = Tracker.gradient(x...); GRAD = :Tracker

function gradtest(f, dims)
    x = randn(dims...)
    grad = ForwardDiff.gradient(x -> sum(sin, f(x)), x)
    grad ≈ _gradient(x -> sum(sin, f(x)), x)[1]
end

@testset "simple" begin

if Tullio._GRAD[] != :Dual || VERSION >= v"1.5" # These 3 give errors on Julia 1.4, LV 0.8, I have no idea why.

    @test _gradient(x -> sum(@tullio y[i] := 2*x[i]), rand(3))[1] == [2,2,2]
    @test _gradient(x -> sum(@tullio y[i] := 2*x[i] + i), rand(3))[1] == [2,2,2]

    # two contributions
    g2(x) = @tullio y[i, j] := 1 * x[i] + 1000 * x[j]
    mat = [1 1 3; 1 1 5; 7 7 7]
    g_fd = ForwardDiff.gradient(x -> sum(mat .* g2(x)), rand(3))
    @test g_fd ≈ _gradient(x -> sum(mat .* g2(x)), rand(3))[1]

    # larger array, no shared indices -- https://github.com/mcabbott/Tullio.jl/issues/14
    r100 = randn(100)
    g_fd = ForwardDiff.gradient(x -> sum(sin, g2(x)), r100)
    @test g_fd ≈ _gradient(x -> sum(sin, g2(x)), r100)[1]

end
    r100 = randn(100)

    # scalar output
    s2(x) = @tullio s := exp(x[i]) / x[j]
    @test _gradient(s2, r100)[1] ≈ ForwardDiff.gradient(s2, r100)

    # two arrays, and a sum
    h2(x,y) = @tullio z[i] := x[i,j] + y[j,i]
    @test _gradient(sum∘h2, rand(2,3), rand(3,2)) == (ones(2,3), ones(3,2))

    # nontrivial function
    flog(x,y) = @tullio z[i] := log(x[i,j]) / y[j,i]
    r_x, r_y = rand(2,3), rand(3,2)
    fx = ForwardDiff.gradient(x -> sum(flog(x, r_y)), r_x)
    fy = ForwardDiff.gradient(y -> sum(flog(r_x, y)), r_y)
    @test fx ≈ _gradient(sum∘flog, r_x, r_y)[1]
    @test fy ≈ _gradient(sum∘flog, r_x, r_y)[2]

    # classic
    mm(x,y) = @tullio z[i,j] := 2 * x[i,k] * y[k,j]
    x1 = rand(3,4);
    y1 = rand(4,5);
    z1 = x1 * y1
    dx, dy = _gradient(sum∘mm, x1, y1)
    @test dx ≈ 2 * ones(3,5) * y1'
    @test dy ≈ 2 * x1' * ones(3,5)

    # abs, abs2
    va = [1,-2,3,-4,5]
    abs_grad = ForwardDiff.gradient(v -> sum(abs, 1 .+ v.^2), va)
    @test abs_grad ≈ _gradient(v -> (@tullio s := abs(1 + v[i]^2)), va)[1]
    abs2_grad = ForwardDiff.gradient(v -> sum(abs2, 1 .+ v.^2), va)
    @test abs2_grad ≈ _gradient(v -> (@tullio s := abs2(1 + v[i]^2)), va)[1]

end
@testset "zero-arrays" begin

    # Using zero-dim arrays fails on ReverseDiff & Tracker
    # Tracker.gradient(x -> x[], fill(1.0))
    # ReverseDiff.gradient(x -> x[], fill(1.0)) # is ambiguous
    if GRAD in [:Tracker, :ReverseDiff]
        @test_skip _gradient(x -> sum(@tullio y[] := log(x[i])), collect(1:3.0))[1] == 1 ./ (1:3)
    else
        @test _gradient(x -> sum(@tullio y[] := log(x[i])), collect(1:3.0))[1] == 1 ./ (1:3)
    end
    # one-element vectors are fine:
    @test _gradient(x -> sum(@tullio y[1] := log(x[i])), collect(1:3.0))[1] == 1 ./ (1:3)
    # which is what's now used for this:
    @test _gradient(x -> (@tullio y := log(x[i])), collect(1:3.0))[1] == 1 ./ (1:3)

end
@testset "gather/scatter" begin

    inds = vcat(1:3, 1:2)
    @test _gradient(x -> sum(@tullio y[i] := x[inds[i]]), rand(3))[1] == [2,2,1]

    _gradient(x -> sum(@tullio y[inds[i]] := x[i]), rand(5))[1] == [1,1,1,1,1]
    ForwardDiff.gradient(x -> sum(@tullio y[inds[i]] := x[i]), rand(5)) == [0,0,1,1,1]
    # This difference may be another edge case like multiple maxima?

    ind2 = rand(1:10, 1024) # many repeats
    dx2 = ForwardDiff.gradient(x -> sum(@tullio y[i] := x[ind2[i]] + x[i]), rand(1024))
    @test dx2 ≈ _gradient(x -> sum(@tullio y[i] := x[ind2[i]] + x[i]), rand(1024))[1]

    ind3 = vcat(unique(rand(2:1024, 10)), 1) # many missing, no repeats, but always includes 1
    g3 = ForwardDiff.gradient(x -> sum(@tullio y[ind3[i]] := i^2 * x[i]), ones(size(ind3)))
    @test g3 ≈ _gradient(x -> sum(@tullio y[ind3[i]] := i^2 * x[i]), ones(size(ind3)))[1]
    # You get weird errors here if indices of y don't start at 1.

    # 1.6 failure on CI, with rand(1:1024, 10)
    # [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 0.0, 64.0, 81.0, 100.0, 121.0] ≈ [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0]

end
@testset "shifts, etc" begin

    c1(N,K) = @tullio M[x,y,c] := N[x+i-1, y+j-1,c] * K[i,j]
    m1 = rand(10,10,2)
    k1 = rand(3,3)
    g_m = ForwardDiff.gradient(N -> sum(sin, c1(N, k1)), m1)
    g_k = ForwardDiff.gradient(K -> sum(sin, c1(m1, K)), k1)
    @test_skip g_m ≈ _gradient(N -> sum(sin, c1(N, k1)), m1)[1]  atol=0.01 # works at repl, fails in tests
    @test g_k ≈ _gradient(K -> sum(sin, c1(m1, K)), k1)[1]  atol=0.01

    c2(mat, kern) = @tullio out[x,y,n] := begin
            i = mod(x+a, axes(mat,1))
            j = mod(y+b, axes(mat,2))
            @inbounds mat[i,j,n] * abs(kern[a,b])
        end (x in axes(mat,1), y in axes(mat,2)) grad=Dual

    if Tullio._GRAD[] == :Dual
        g_m = ForwardDiff.gradient(N -> sum(sin, c2(N, k1)), m1)
        g_k = ForwardDiff.gradient(K -> sum(sin, c2(m1, K)), k1)
        @test g_m ≈ _gradient(N -> sum(sin, c2(N, k1)), m1)[1]  atol=0.01
        @test g_k ≈ _gradient(K -> sum(sin, c2(m1, K)), k1)[1]  atol=0.01
    end

end
@testset "mod, clamp, pad" begin

    fmod(x) = @tullio y[i] := x[mod(i)]  i in 1:5
    fclamp(x) = @tullio y[i] := x[clamp(i)]  i in 1:5
    fpad(x) = @tullio y[i] := x[pad(i-2,2)]
    @test _gradient(sum∘fmod, ones(3))[1] == [2,2,1]
    @test _gradient(sum∘fclamp, ones(3))[1] == [1,1,3]
    @test _gradient(sum∘fpad, ones(3))[1] == [1,1,1]

end
@testset "@inferred" begin

    h2(x,y) = @tullio z[i] := x[i,j] + y[j,i]  # as above
    flog(x,y) = @tullio z[i] := log(x[i,j]) / y[j,i]

    mat = rand(3,3)
    @test @inferred(h2(mat, mat)) ≈ vec(sum(mat .+ mat', dims=2))
    @test @inferred(flog(mat, mat)) isa Vector

    if GRAD == :Zygote
        @test_broken @inferred(_gradient(sum∘h2, rand(2,3), rand(3,2))) isa Tuple
        @test_broken @inferred(_gradient(sum∘flog, mat, mat)) isa Tuple
    else
        @test @inferred(_gradient(sum∘h2, rand(2,3), rand(3,2))) isa Tuple
        @test @inferred(_gradient(sum∘flog, mat, mat)) isa Tuple
    end

end
@testset "from TensorTrace" begin
    # These can all be handled using TensorOperations

    triv1(x) = @tullio A[i,j] := 2 * x[i,j]
    @test gradtest(triv1, (2,3))

    r32 = randn(3,2);
    r312 = randn(3,1,2);

    ## trace!
    tr1(x) = @tullio T[k] := 22 * x[i,i,k]
    @test gradtest(tr1, (3,3,4))

    tr2(x) = @tullio T[k] := 22 * x[i,i,k,j,j]
    @test gradtest(tr2, (3,3,4,7,7))

    ## contract! A
    con1(x) = @tullio C[i,j] := 5 * x[i,k] * r32[k,j]
    @test gradtest(con1, (2,3))

    r22 = rand(2,2);

    con3(x) = @tullio C[i,j,m,n] := x[i,j,k] * r312[k,m,n]
    @test gradtest(con3, (1,2,3))

    con4(x) = @tullio C[i,m] := x[i,kk,k] * r312[k,m,kk]
    @test gradtest(con4, (1,2,3))

    con5(x) = @tullio C[j,i,n,m] := 44 * x[i,j,k] * r312[k,m,n]
    @test gradtest(con5, (1,2,3))

    r392 = randn(3,9,2);
    con6(x) = @tullio C[n,i,m,j] := x[i,j,k] * r392[k,m,n]
    @test gradtest(con6, (9,2,3))

    con7(x) = @tullio C[m,n,j,i] := 44 * x[i,j,k] * r392[k,m,n]
    @test gradtest(con7, (9,2,3))

    ## contract! B
    con8b(x) = @tullio K[i,j] := 5 * r32[i,k] * x[k,j]
    @test gradtest(con8b, (2,3))

    con9b(x) = @tullio K[i,j,m,n] := r312[i,j,k] * x[m,k,n]
    @test gradtest(con9b, (1,2,3))

    con10b(x) = @tullio K[n,j,m,i] := r392[i,j,k] * x[m,k,n]
    @test gradtest(con10b, (9,2,3))

    r3399 = randn(3,3,9,9);

    con13(x) = @tullio K[i,j] := r3399[s,s,j,k] * x[t,t,k,i]
    @test gradtest(con13, (3,3,9,9))

    r33 = rand(3,3);
    con14(x) = @tullio K[i,j] := r3399[a,b,j,k] * x[b,c,k,i] * r33[a,c]
    @test gradtest(con14, (3,3,9,9))

    ## scalar -- one with :=, one without
    sc1(x) = @tullio s = r22[b,β] * x[a,b,c] * r312[c,a,β]
    @test gradtest(sc1, (1,2,3))

    sc2(x) = @tullio s := x[γ,c] * r3399[c,γ,i,i]
    @test gradtest(sc2, (3,3))

end

if Tullio._GRAD[] != :Dual
#=
    @testset "products" begin

        p1(x) = @tullio (*) z = x[i]
        @test _gradient(p1, 1:4)[1] == ForwardDiff.gradient(p1, 1:4)
        @test _gradient(p1, -1:3)[1] == ForwardDiff.gradient(p1, -1:3) # one zero
        @test _gradient(p1, [1,0,2,0])[1] == ForwardDiff.gradient(p1, [1,0,2,0])

        p2(m,v) = @tullio (*) y[i] := (m[i,j] + 3*v[j])^2 # / sqrt(v[i])
        p2(m,v) = @tullio (*) y[i] := m[i,j] * v[j]
        m1 = rand(4,4) .+ 1
        v1 = rand(4) .+ 1
        dm = ForwardDiff.gradient(m -> sum(p2(m,v1)), m1)
        @test dm ≈ _gradient(sum∘p2, m1, v1)[1]
        dv = ForwardDiff.gradient(v -> sum(p2(m1,v)), v1)
        @test_broken dv ≈ _gradient(sum∘p2, m1, v1)[2]

        m1[2,3] = 0
        p3(m) = @tullio (*) y[i] := 4 * m[i,j]
        @test _gradient(sum∘p3, m1)[1] ≈ ForwardDiff.gradient(sum∘p3, m1)
        m1[3,4] = -1
        p4(m) = @tullio (*) y[i] := sin(1 + m[i,j])
        @test _gradient(sum∘p4, m1)[1] ≈ ForwardDiff.gradient(sum∘p4, m1)

    end
=#
    @testset "min/max" begin

        f1(x) = @tullio (max) z = x[i]
        f2(x) = @tullio (min) z = x[i] # avx=false

        @test _gradient(f1, 1:4)[1] == ForwardDiff.gradient(f1, 1:4)
        @test _gradient(f2, 1:4)[1] == ForwardDiff.gradient(f2, 1:4)

        @test _gradient(f1, [2,2,3,3])[1] in ([0,0,1,0], [0,0,0,1]) # changes with @avx
        ForwardDiff.gradient(f1, [2,2,3,3]) == [0,0,0,1] # different sub-gradient, OK
        @test _gradient(f2, [2,2,3,3])[1] == [1,0,0,0]

        m4 = reshape(shuffle(1:3*4*5*2), 3,4,5,2);
        m2 = reshape(shuffle(1:16), 4,4);
        v2 = shuffle(1:4)

        f3(x) = @tullio (max) y[i,k,l] := x[i,j,k,l]

        @test all(==(1), sum(_gradient(sum∘f3, m4)[1], dims=2))
        @test _gradient(sum∘f3, m4)[1] ≈ ForwardDiff.gradient(sum∘f3, m4)

        f4(x) = @tullio (min) y[j] := x[i,j,k,l]

        @test all(==(1), sum(_gradient(sum∘f4, m4)[1], dims=(1,3,4)))
        @test _gradient(sum∘f4, m4)[1] ≈ ForwardDiff.gradient(sum∘f4, m4)

        f5(x,y) = @tullio (max) z[i] := x[i,j] + 0.01*y[i]

        dm = ForwardDiff.gradient(m -> sum(f5(m,v2)), m2)
        @test dm ≈_gradient(sum∘f5, m2, v2)[1]
        dv = ForwardDiff.gradient(v -> sum(f5(m2,v)), v2)
        @test dv ≈_gradient(sum∘f5, m2, v2)[2]

        f6(x,y) = @tullio (max) z[i] := x[i,j] + 0.01*y[j] # max is now along y, not perp

        dm = ForwardDiff.gradient(m -> sum(f6(m,v2)), m2)
        @test dm ≈ _gradient(sum∘f6, m2, v2)[1]
        dv = ForwardDiff.gradient(v -> sum(f6(m2,v)), v2)
        @test dv ≈ _gradient(sum∘f6, m2, v2)[2]

        f7(x,y) = @tullio (max) z[i] := x[i,j]^2 / sqrt(y[i]) + exp(y[j])  avx=false

        dm = ForwardDiff.gradient(m -> sum(f7(m,v2)), m2)
        @test dm ≈ _gradient(sum∘f7, m2, v2)[1]  # avx: broken in tests, Julia 1.4
        dm .- _gradient(sum∘f7, m2, v2)[1]
        dv = ForwardDiff.gradient(v -> sum(f7(m2,v)), v2)
        @test dv ≈ _gradient(sum∘f7, m2, v2)[2]

        f8(x,y) = @tullio (max) z[i,l] := log(x[i,j,k,l]) / y[j]^1/3  avx=false
        f9(x,y) = @tullio (min) z[i,j] := log(x[i,j,k,l]) / y[j]^1/3  avx=false
        @tullio z89[i,j,k,l] := log(m4[i,j,k,l]) / v2[j]^1/3
        length(z89), length(unique(z89))

        dm = ForwardDiff.gradient(m -> sum(f8(m,v2)), m4)
        @test dm ≈ _gradient(sum∘f8, m4, v2)[1]  # avx: OK with 0.8, broken with 0.9
        dm .- _gradient(sum∘f8, m4, v2)[1]       # at exactly one element
        dv = ForwardDiff.gradient(v -> sum(f8(m4,v)), v2)
        @test dv ≈ _gradient(sum∘f8, m4, v2)[2]

        dm = ForwardDiff.gradient(m -> sum(f9(m,v2)), m4)
        @test dm ≈_gradient(sum∘f9, m4, v2)[1]  # avx: broken with 0.8 and 0.9
        dm .- _gradient(sum∘f9, m4, v2)[1]
        dv = ForwardDiff.gradient(v -> sum(f9(m4,v)), v2)
        @test dv ≈ _gradient(sum∘f9, m4, v2)[2]  # avx: broken with 0.8 and 0.9
        dv .- _gradient(sum∘f9, m4, v2)[2]       # but broken in different elements
        # I suspect that @avx is re-ordering loops, which makes onlyone() incorrect.

    end
    @testset "finalisers" begin

        norm2(m) = @tullio n[i] := m[i,j]^2 |> sqrt

        gradtest(norm2, (3,4))
        mat = rand(3,3)
        @test _gradient(sum∘norm2, mat)[1] ≈ ForwardDiff.gradient(sum∘norm2, mat)
        @test gradtest(norm2, (3,4))

        layer(x) = @tullio y[i,k] := mat[i,j] * x[j,k] |> tanh
        @test gradtest(layer, (3,4))

        lse1(mat) = @tullio lse[j] := log <| exp(mat[i,j])
        @test gradtest(lse1, (3,4))

        # relu(x) = max(x, zero(x))
        # lay2(x) = @tullio y[i,k] := mat[i,j] * x[j,k] |> relu

        mx3(x) = @tullio (max) r[i] := x[i,j]^3 |> cbrt
        mx3(mat) # hmmm what is this?
        _gradient(sum∘mx3, mat)[1] # zero

    end
end

if GRAD == :Zygote
    @testset "nograd keyword" begin

        f2(x,y) = @tullio out[i,j] := x[i] + y[j]  nograd=y threads=false
        @test _gradient(sum∘f2, rand(2), rand(2)) == ([2,2], nothing)

        f3(x,y,z) = @tullio out[i,j] := x[i] + y[j] * z[k]  nograd=(x,z) threads=false
        @test _gradient(sum∘f3, rand(2), rand(2), ones(2)) == (nothing, [4,4], nothing)

        f0(x,y) = @tullio out[i,j] := x[i]/y[j]  nograd=(y,x) threads=false
        @test _gradient(sum∘f0, rand(2), rand(2)) == (nothing, nothing)

    end
end
