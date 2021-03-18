
using Tullio, Test, ForwardDiff
# using Tracker; _gradient(x...) = Tracker.gradient(x...); GRAD = :Tracker

function gradtest(f, dims)
    x = randn(dims...)
    grad = ForwardDiff.gradient(x -> sum(sin, f(x)), x)
    grad ≈ _gradient(x -> sum(sin, f(x)), x)[1]
end

@testset "from TensorTrace" begin
    # These can all be handled using TensorOperations

    triv1(x) = Tullio.@tensor A[i,j] := 2 * x[i,j]
    @test gradtest(triv1, (2,3))

    r32 = randn(3,2);
    r312 = randn(3,1,2);

    ## trace!
    tr1(x) = Tullio.@tensor T[k] := 22 * x[i,i,k]
    @test gradtest(tr1, (3,3,4))

    tr2(x) = Tullio.@tensor T[k] := 22 * x[i,i,k,j,j]
    @test gradtest(tr2, (3,3,4,7,7))

    ## contract! A
    con1(x) = Tullio.@tensor C[i,j] := 5 * x[i,k] * r32[k,j]
    @test gradtest(con1, (2,3))

    r22 = rand(2,2);

    con3(x) = Tullio.@tensor C[i,j,m,n] := x[i,j,k] * r312[k,m,n]
    @test gradtest(con3, (1,2,3))

    con4(x) = Tullio.@tensor C[i,m] := x[i,kk,k] * r312[k,m,kk]
    @test gradtest(con4, (1,2,3))

    con5(x) = Tullio.@tensor C[j,i,n,m] := 44 * x[i,j,k] * r312[k,m,n]
    @test gradtest(con5, (1,2,3))

    r392 = randn(3,9,2);
    con6(x) = Tullio.@tensor C[n,i,m,j] := x[i,j,k] * r392[k,m,n]
    @test gradtest(con6, (9,2,3))

    con7(x) = Tullio.@tensor C[m,n,j,i] := 44 * x[i,j,k] * r392[k,m,n]
    @test gradtest(con7, (9,2,3))

    ## contract! B
    con8b(x) = Tullio.@tensor K[i,j] := 5 * r32[i,k] * x[k,j]
    @test gradtest(con8b, (2,3))

    con9b(x) = Tullio.@tensor K[i,j,m,n] := r312[i,j,k] * x[m,k,n]
    @test gradtest(con9b, (1,2,3))

    con10b(x) = Tullio.@tensor K[n,j,m,i] := r392[i,j,k] * x[m,k,n]
    @test gradtest(con10b, (9,2,3))

    r3399 = randn(3,3,9,9);

    con13(x) = Tullio.@tensor K[i,j] := r3399[s,s,j,k] * x[t,t,k,i]
    @test gradtest(con13, (3,3,9,9))

    r33 = rand(3,3);
    con14(x) = Tullio.@tensor K[i,j] := r3399[a,b,j,k] * x[b,c,k,i] * r33[a,c]
    @test gradtest(con14, (3,3,9,9))

    ## scalar -- one with :=, one without
    sc1(x) = Tullio.@tensor s = r22[b,β] * x[a,b,c] * r312[c,a,β]
    @test gradtest(sc1, (1,2,3))

    sc2(x) = Tullio.@tensor s := x[γ,c] * r3399[c,γ,i,i]
    @test gradtest(sc2, (3,3))

end

@testset "errors" begin
    @test_throws LoadError @eval Tullio.@tensor C[k] := A[i,i,k] + B[k]  # two terms
    @test_throws LoadError @eval Tullio.@tensor B[k] := conj(A[k])  # functions
    @test_throws LoadError @eval Tullio.@tensor C[k] := A[i, i+k]  # not a contraction
end

