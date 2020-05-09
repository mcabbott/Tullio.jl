
using Tullio, Test, LinearAlgebra

@testset "new arrays" begin

    # functions
    @tullio A[i] := (1:10)[i]^2
    @test A == [i^2 for i in 1:10]

    @tullio A[i] := (1:10)[i] * i
    @test A == [i^2 for i in 1:10]

    # diagonals
    @tullio D[i,i] := trunc(Int, sqrt(A[i]))
    @test D == Diagonal(sqrt.(A))

    # arrays of arrays
    C = [fill(i,3) for i=1:5]
    @tullio M[i,j] := C[i][j]
    @test M == (1:5) .* [1 1 1]

    # fields
    E = [(a=i, b=i^2, c=[i,2i,3i]) for i in 1:10]
    @tullio O[i] := A[i]//E[i].b # gets :noavx flag from E[i].b
    @test O == ones(10)

    @tullio F[i,j] := E[i].c[j] # also :noavx
    @test F == (1:10) .* [1 2 3]

    # arrays of tuples
    Y = [(i,i^2,i^3) for i in 1:10]
    @tullio W[i,j] := Y[i][j]
    @test W[9,3] == 9^3

    # scalar
    @tullio S := A[i]/2
    @tullio S′ = A[i]/2 # here = is equivalent
    @test S ≈ S′ ≈ sum(A)/2

    @tullio Z[] := A[i] + A[j]
    @test Z isa Array{Int,0}
    @tullio Z′[1,1] := A[i] + A[j]
    @test size(Z′) == (1,1)

    # scalar update
    @tullio S += A[i]/2
    @test S ≈ sum(A)

    # inner
    J = repeat(1:3, 4);
    @tullio G[i,k] := F[i,J[k]]
    @test G[3,1] == G[3,4] == G[3,7]

    # fixed
    @tullio F[i] := D[i,5]
    @test F[5] == 5

    j = 6
    @tullio G[i] := D[i,$j]
    @test G[6] == 6

    @tullio H[i] := D[i,:] # storage_type(H, D) == Array, this avoids @avx
    @test H[5] == F

    # trivial dimensions
    @tullio J[1,1,i] := A[i]
    @test size(J) == (1,1,10)

    @tullio J[_,i] := A[i]
    @test J == A'

    # non-unique arrays
    @tullio A2[i] := A[i] + A[i]
    @test A2 == 2 .* A

    # broadcasting
    @tullio S[i] := sqrt.(M[:,i]) # dot sets :noavx & :nograd
    @tullio T[i] := A[i] .+ A[j]  # dot does nothing, except set :noavx & :nograd

    # scope
    f(x,k) = @tullio y[i] := x[i] + i + $k
    @test f(ones(3),j) == 1 .+ (1:3) .+ j

    g(x) = @tullio y := sqrt(x[i])
    @test g(fill(4,5)) == 10

    # ranges
    @tullio K[i] := i^2  (i ∈ 1:3)
    @test K == (1:3).^2
    @test axes(K,1) === Base.OneTo(3) # literal 1:3

    @tullio N[i,j] := A[i]/j  (j in axes(K,1))  (i in axes(A,1)) # K not an argument
    @test N ≈ A ./ (1:3)'

    # primes
    @test A == @tullio P[i′] := A[i']
    @test A == @tullio P[i'] := A[i′]
    @test [1,4,9] == @tullio Q[i'] := (i′)^2  (i' in 1:3)

    # non-numeric array
    @tullio Y[i] := (ind=i, val=A[i])
    @test Y[2] === (ind = 2, val = 4)

    # no name given
    Z = @tullio [i] := A[i] + 1
    @test Z == A .+ 1

    # internal name leaks
    @test !isdefined(@__MODULE__, Tullio.ZED)
    @test !isdefined(@__MODULE__, Symbol(Tullio.AXIS, :i))

end

@testset "in-place" begin

    A = [i^2 for i in 1:10]
    D = similar(A, 10, 10)

    @tullio D[i,j] = A[i] + 100
    @test D[3,7] == A[3] + 100

    # sum and +=
    B = copy(A);
    D .= 3;
    @tullio B[i] += D[i,j]
    @test B[1] == A[1] + 30

    # writing back into same
    B = copy(A)
    @tullio B[i] += B[i] + 10^3
    @test B[6] == 2 * A[6] + 10^3

    @tullio A[i] = A[i] + 100
    @test A[1] == 101

    # indices in expression
    @tullio A[i] = 100*i
    @test A[7] == 700

    # fixed on left
    j = 3
    @tullio D[$j,i] = 99
    @test D[j,j] == 99
    @test D[1,1] != 0

    # diagonal & ==, from https://github.com/ahwillia/Einsum.jl/pull/14
    B = [1 2 3; 4 5 6; 7 8 9]
    @tullio W[i, j, i, n] := B[n, j]  i in 1:2
    @test size(W) == (2,3,2,3)
    @test W[1,2,1,3] == B[3,2]

    W2 = zero(W);
    @tullio W2[i, j, m, n] = (i == m) * B[n, j]
    @test W2 == W

    @test_throws Exception Tullio._tullio(:( [i,j] = A[i] + 100 ))

    # internal name leaks
    @test !isdefined(@__MODULE__, Tullio.ZED)
    @test !isdefined(@__MODULE__, Symbol(Tullio.AXIS, :i))

end

@testset "broadcasting" begin

    f1(A) = @tullio C[i, ..] := A[i, ..] + 1
    @test f1(ones(3)) == ones(3) .+ 1
    @test f1(ones(3,4)) == ones(3,4) .+ 1
    @test f1(ones(3,4,5)) == ones(3,4,5) .+ 1

    f2(A) = @tullio C[i, ..] := A[i, k, ..]
    @test f2(ones(3,4)) == fill(4.0, 3)
    A3 = rand(3,4,5)
    @test f2(A3) ≈ dropdims(sum(A3, dims=2), dims=2)

    f3(A, B) = @tullio C[i,j, ..] := A[i, k, ..] * B[j, k, ..]
    A2 = rand(3,3);
    B2 = rand(3,3);
    @test f3(A2, B2) ≈ A2 * B2'
    A3 = rand(3,3,2);
    B3 = rand(3,3,2);
    C3 = f3(A3, B3)
    @test C3[:,:,1] ≈ A3[:,:,1] * B3[:,:,1]'
    @test C3[:,:,2] ≈ A3[:,:,2] * B3[:,:,2]'

    C4 = f3(A3, B2)
    @test C4[:,:,1] ≈ A3[:,:,1] * B2[:,:]'
    @test C4[:,:,2] ≈ A3[:,:,2] * B2[:,:]'

end

@testset "without packages" begin

    A = [i^2 for i in 1:10]

    # without OffsetArrays
    @test axes(@tullio B[i] := A[2i+1] + A[i]) === (Base.OneTo(4),)
    @test_throws Exception @tullio C[i] := A[2i+5]

    # without NamedDims
    @test_throws Exception @tullio M[row=i, col=j, i=1] := (1:3)[i] // (1:7)[j]

end

using OffsetArrays

@testset "index shifts" begin

    A = [i^2 for i in 1:10]

    @tullio L[i,j] := A[i]//j  (j ∈ 2:3, i in 1:10) # no shift, just needs OffsetArrays
    @test axes(L) == (1:10, 2:3)

    # shifts
    @tullio B[i] := A[2i+1] + A[i]
    @test axes(B,1) == 1:4 # would be OneTo(4) without OffsetArrays

    @tullio C[i] := A[2i+5]
    @test axes(C,1) == -2:2 # error without OffsetArrays

    j = 7 # interpolation
    @tullio C[i] := A[2i+$j]
    @test axes(C,1) == -3:1

    cee(A) = @tullio C[i] := A[2i+$j] # closure over j
    @test axes(cee(A),1) == -3:1

    @test_throws Exception @tullio D[i] := A[i] + B[i]
    @tullio D[i] := A[i] + B[i+0] # switches to intersection
    @test axes(D,1) == 1:4

    @test_throws Exception @tullio M[i,j] := A[i+0]/A[j]  (i ∈ 2:5, j ∈ 2:5) # intersection for i but not j

    @tullio L[i] := A[i+j+1]  (j ∈ -1:1)
    @test axes(L,1) == 1:8

    # negative
    @test eachindex(@tullio F[i] := A[-1i]) == -10:-1
    @test eachindex(@tullio F[i] := A[-i]) == -10:-1
    @test eachindex(@tullio F[i] := A[-i+0]) == -10:-1
    @test eachindex(@tullio F[i] := A[0-i]) == -10:-1

    # non-constant
    @test axes(@tullio I[i,j] := A[i+j] + 0 * B[j]) == (0:6, 1:4)
    @test axes(@tullio I[i,j] := A[j+i+0] + 0 * B[j]) == (0:6, 1:4)
    @test axes(@tullio I[i,j] := A[(j+i)*1] + 0 * B[j]) == (0:6, 1:4)
    @test axes(@tullio I[i,j] := A[2i+j] + 0 * B[j]) == (0:3, 1:4)
    @test axes(@tullio I[i,j] := A[1j+2i] + 0 * B[j]) == (0:3, 1:4)
    @test axes(@tullio I[i,j] := A[i+2j] + 0 * B[j]) == (-1:2, 1:4)
    @test axes(@tullio I[i,j] := A[2i+2j] + 0 * B[j]) == (0:1, 1:4)
    @test axes(@tullio I[i,j] := A[2(i+j)] + 0 * B[j]) == (0:1, 1:4)
    @test axes(@tullio I[i,j] := A[2i-1+2j] + 0 * B[j]) == (0:1, 1:4)
    @test axes(@tullio I[i,j] := A[2i+2j+5] + 0 * B[j]) == (-3:-2, 1:4)
    @test axes(@tullio I[i,j] := A[2i+2j-5] + 0 * B[j]) == (2:3, 1:4)
    @test axes(@tullio I[i,j] := A[2i+2(j-2)-1] + 0 * B[j]) == (2:3, 1:4)
    @test axes(@tullio I[i,j] := A[2(0+i)+(2j-4)-1] + 0 * B[j]) == (2:3, 1:4)

    @test axes(@tullio J[i,j] := A[i-j] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-(-i+j)] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-(j-i)] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-1*(j-i)] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[i+(-j)] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-j+i] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-1j+i] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-1j-(-i)] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[i-2j] + 0 * B[j]) == (9:12, 1:4)
    @test axes(@tullio J[i,j] := A[-2j+i] + 0 * B[j]) == (9:12, 1:4)
    @test axes(@tullio J[i,j] := A[2i-2j] + 0 * B[j]) == (5:6, 1:4)

    @test_throws LoadError @eval @tullio I[i,j] := A[i+j] # under-specified

    # in-place
    E = zero(A)
    @tullio E[i] = A[i+5] + 100
    @test E == vcat(A[6:end] .+ 100, zeros(Int,5))

    M = fill(pi/2, 10, 10)
    @tullio M[i,i] = A[i-2]
    @test M[3,3] == A[1]
    @test M[1,1] == pi/2 # was not set to zero

    # shifts on left
    E = zero(A)
    @tullio E[2i+1] = A[i]
    @test E[2+1] == A[1]
    @test E[2*4+1] == A[4]

    # non-constant
    @tullio I[i,j] := 0 * A[i+j] + 0 * B[j]
    @test axes(@tullio I[i,j] = A[i+j] + B[j]) == (0:6, 1:4) # over-specified
    @test axes(@tullio I[i,j] = A[i+j]) == (0:6, 1:4) # needs range from LHS

end

@testset "named dimensions" begin

    using NamedDims

    # reading
    N = NamedDimsArray(rand(Int8,3,10), (:r, :c))

    @tullio A[i,j] := N[i, j] + 100 * (1:10)[j]

    @tullio C[j,i] := N[c=j, r=i] + 100 * (1:10)[j]
    @test_broken A == C'
    @test_broken dimnames(C) == (:_, :_)

    # writing
    @tullio M[row=i, col=j, i=1] := (1:3)[i] // (1:7)[j]
    @test dimnames(M) == (:row, :col, :i)

end
