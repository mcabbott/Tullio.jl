
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
    @tullio O[i] := A[i]//E[i].b # avx disabled by try/catch
    @test O == ones(10)

    @tullio F[i,j] := E[i].c[j]
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

    # fixed
    @tullio F[i] := D[i,5]
    @test F[5] == 5

    j = 6
    @tullio G[i] := D[i,$j]
    @test G[6] == 6

    @test_throws LoadError @eval @tullio D[i,$j] := A[i]

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
    @tullio S[i] := sqrt.(M[:,i]) # avx & grad now disabled by try/catch
    # @tullio T[i] := A[i] .+ A[j]  # dot does nothing, fails with LoopVectorization loaded

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

    @test_throws String @tullio A[i] := i^2 (i in 1+10) # not a range

    # repeated scalar arg
    tri = Base.OneTo(3) # with 1:3, this fails without OffsetArrays,
    # as it only converts shifted indices to OneTo
    @tullio M[i,j] := (r=i, c=j)  (i in tri, j in tri)
    @test M[3,3] == (r=3, c=3)

    # indexing by an array, "gather"...
    J = repeat(1:3, 4);
    @tullio G[i,k] := M[i,J[k]]
    @test G[3,1] == G[3,4] == G[3,7]

    inds = vcat(1:3, 1:3)
    @tullio AI[i] := A[inds[i]]
    @test AI == A[inds]
    jnds = -5:5
    @test_throws String @tullio AJ[j] := A[jnds[j]]
    @test_throws BoundsError A[jnds]
    knds = 1:3.0
    @test_throws String @tullio AK[j] := A[knds[j]]
    @test_throws ArgumentError A[knds]

    # ... and "scatter"
    M = rand(1:99, 4,5)
    J = [3,1,2,3]
    @tullio H[J[i],k] := M[i,k] # i is not marked unsafe, may be threaded
    @test size(H) == (3,5)
    @test H[1,:] == M[2,:] # but H[3,:] gets written into twice.

    J′ = [1,2,10]
    @tullio H′[J′[i'],k] := A[k]
    @test size(H′) == (10, length(A))
    @test H′[2,:] == A
    @test H′[3,4] == 0 # zeroed before being written into

    inds = vcat(1:3, 1:3)
    @test_throws String @tullio H[inds[i],k] := M[i,k] # range of index i

    # masking
    @tullio M[i,j] := A[i] * A[j] * (i<=j)
    @test M == UpperTriangular(A .* A')

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

    # multi-line
    @tullio B[i,j] := begin
        x = (1:10)[i] + 3
        y = (1:3)[j]
        x // y
    end
    @test B == (4:13) .// (1:3)'

    # wrong ndims
    @test_throws Any @tullio Z[i] := B[i] # Any as TensorOperations throws ErrorException

    # internal name leaks
    for sy in Tullio.SYMBOLS
        @test !isdefined(@__MODULE__, sy)
    end

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

    @test_throws LoadError @eval @tullio [i,j] = A[i] + 100

    # zero off-diagonal? not now, but maybe it should?
    @tullio D[i,i] = A[i]

    # scatter operation
    D = similar(A, 10, 10) .= 999
    inds = [2,3,5,2]
    @tullio D[inds[i],j] = A[j]
    @test D[2,:] == A
    @test D[4,4] == 0 # zeroed before writing.

    @tullio D[inds[i],j] += A[j]
    @test D[2,:] == 3 .* A # was not re-zeroed for +=

    kinds = [1,2,13,4]
    @test_throws String @tullio D[kinds[i],j] = A[j]

    # assignment: no loop over j
    B = zero(A);
    @tullio B[i] = begin
        j = mod(i^4, 1:10)
        A[j]
    end
    @test_skip B == A[[mod(i^4, 1:10) for i in 1:10]]
    # on travis 1.3 multi-threaded, B == [500, 600, 100, 600, 500, 600, 100, 600, 100, 1000]
    # and on 1.4 multi-threaded,    B == [100, 600, 100, 600, 100, 600, 100, 600, 100, 1000]

    # wrong ndims
    @test ndims(B)==1 && ndims(D)==2
    @test_throws Any @tullio B[i] = D[i]^2
    @test_throws Any @tullio D[i] = B[i]+2
    @test_throws Any @tullio B[i,j] = D[i,j]

    # internal name leaks
    for sy in Tullio.SYMBOLS
        @test !isdefined(@__MODULE__, sy)
    end

end

if !@isdefined OffsetArray
    @testset "without packages" begin

        A = [i^2 for i in 1:10]

        # without OffsetArrays
        @test axes(@tullio B[i] := A[2i+1] + A[i]) === (Base.OneTo(4),)
        @test_throws String @tullio C[i] := A[2i+5]

        J = [3,5,7] # doesn't start at 1
        @test_throws String @tullio G[J[i],k] := A[k]

        # without NamedDims
        @test_throws UndefVarError @tullio M[row=i, col=j, i=1] := (1:3)[i] // (1:7)[j]

    end
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

    @test_throws String @tullio D[i] := A[i] + B[i]
    @tullio D[i] := A[i] + B[i+0] # switches to intersection
    @test axes(D,1) == 1:4

    @test_throws String @tullio M[i,j] := A[i+0]/A[j]  (i ∈ 2:5, j ∈ 2:5) # intersection for i but not j

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

    # indexing by an array
    inds = [-1,0,0,0,1]
    @tullio K[i,j] := A[inds[i]+j]
    @test K[2,3] == K[3,3] == K[4,3]
    @test axes(K) == (1:5, 2:9)

    @tullio K2[i,j] := A[j+2inds[i]]
    @test axes(K2) == (1:5, 3:8)

    j = 7
    @test_skip @tullio K3[i,j] := A[j+2inds[i]+$j]
    @test_broken vec(K2) == vec(K3)

    # scatter with shift not allowed
    @test_throws LoadError @eval @tullio G[inds[i]+1, j] := A[j]
    @test_throws LoadError @eval @tullio G[2inds[i], j] := A[j]

    # multiplication not implemented
    @test_throws LoadError @eval @tullio C[i] = A[i*j] + A[i]
    @test_throws LoadError @eval @tullio C[i] = A[i⊗j] + A[i]
    @test_throws LoadError @eval @tullio C[i] = A[(i,j)] + A[i]

    # magic shift
    @test axes(@tullio Z[i+_] := A[2i+10]) === (Base.OneTo(5),)

    @test_throws LoadError @eval @tullio Z[_+i] := A[2i+10] # wrong notation
    @test_throws LoadError @eval @tullio Z[J[i]+_] := A[2i+10] # with scatter
    @test_throws LoadError @eval @tullio Z[i+_] = A[2i+10] # in-place
end

@testset "modulo & clamped indices" begin

    A = [i^2 for i in 1:10]
    B = 1:5

    @test vcat(B,B) == @tullio C[i] := B[mod(i)]  i in 1:10
    @test vcat(B, fill(B[end],5)) == @tullio D[i] := min(A[i], B[clamp(i)])

    @test [4,16,36,64,100,4] == @tullio E[i] := A[mod(2i)]  i in 1:6
    @test [9,25,49,81,1,9] == @tullio E[i] := A[mod(2i+1, 1:10)]  i in 1:6

    # unable to infer range
    @test_throws LoadError @eval @tullio F[i] := A[mod(i+1)]
    # can't use index mod(i) on LHS
    @test_throws LoadError @eval @tullio G[mod(i)] := A[i]

end

@testset "other reductions" begin

    A = [i^2 for i in 1:10]

    # basics
    @test [prod(A)] == @tullio (*) P[_] := float(A[i])
    @test maximum(A) == @tullio (max) m := float(A[i])
    @test minimum(A) == @tullio (min) m := float(A[i]) # fails with @avx

    @test true == @tullio (&) p := A[i] > 0
    @test true === @tullio (&) p := A[i] > 0 # sum([true]) isa Int
    @test true == @tullio (|) q := A[i] > 50

    # in-place
    C = copy(A)
    @test cumprod(A) == @tullio (*) C[k] = ifelse(i<=k, A[i], 1)
    @test cumprod(A).^2 == @tullio (*) C[k] *= i<=k ? A[i] : 1

    M = rand(1:9, 4,5)
    @test vec(prod(M,dims=2)) == @tullio (*) B[i] := M[i,j]

    # ^= generalises +=, *=
    C = copy(A)
    @tullio (max) C[i] ^= 5i
    @test C == max.(5:5:50, A)
    @test_throws LoadError @eval @tullio A[i] ^= A[i]
    @test_throws LoadError @eval @tullio (*) A[i] ^= A[i]

    # initialisation
    @test 200 == @tullio (max) m := A[i] init=200
    @tullio (max) C[i] := i^2   (i in 1:10, j in 1:1)  init=33.3 # widens type
    @test C == max.(33.3, A)
    @tullio C[i] := 0   (i in 1:10, j in 1:1)  init=randn() tensor=false # runs once
    @test C == fill(C[1], 10)

    # more dimensions
    Q = rand(1:10^3, 4,5,6)
    @test vec(maximum(Q,dims=(2,3))) == @tullio (max) R[i] := Q[i,j,k]
    @test vec(minimum(Q,dims=(1,3))).+2 == @tullio (min) P[j] := Q[i,j,k]+2
    @test dropdims(maximum(Q, dims=2), dims=2) == @tullio (max) S[i,k] := Q[i,j,k]

    # indexing
    ind = vcat(1:3, 1:3)
    V = 1:6
    @tullio (*) Z[j] := M[ind[k],j] * exp(-V[k]) # product over k
    @test Z ≈ vec(prod(M[ind,:] .* exp.(.-V), dims=1))

    # scalar update ("plusequals" internally)
    s = 1.0
    @tullio (*) s *= float(A[i])
    @test s == prod(A)
    @tullio s *= float(A[i]) # works without specifying (*), is this a good idea?
    @test s == float(prod(A))^2

    @test_throws LoadError @eval @tullio s += (*) A[i]
    @test_throws LoadError @eval @tullio s *= (max) A[i]

    # scalar + threading
    L = randn(100 * Tullio.TILE[]);
    @tullio (max) m := L[i]
    @test m == maximum(L)

    # no reduction means no redfun, and no init:
    @test_throws LoadError @eval @tullio (max) A2[i] := A[i]^2
    @test_throws LoadError @eval @tullio A2[i] := A[i]^2 init=0.0

end

@testset "finalisers" begin

    A = [i^2 for i in 1:10]
    @tullio n2 = A[i]^2 |> sqrt
    @test n2 ≈ norm(A,2)
    @tullio n3 := cbrt <| A[i]^3
    @test n3 ≈ norm(A,3)

    @tullio B[i,j] := A[i] + A[k] // A[j]

    @tullio B2[_,j] := (B[i,j] + B[j,i])^2 |> sqrt
    @test B2 ≈ mapslices(norm, B + B', dims=1)

    # trivial use, no reduction -- now forbidden
    @test_throws LoadError @eval @tullio A2[i] := A[i]^2 |> sqrt
    @test_throws LoadError @eval @tullio (*) A2[i] := A[i]^2 |> sqrt

    # larger size, to trigger threads & tiles
    C = randn(10^6) # > Tullio.BLOCK[]
    @tullio n2 = C[i]^2 |> sqrt
    @test n2 ≈ norm(C,2)

    D = rand(1000, 1000) # > Tullio.TILE[]
    @tullio D2[_,j] := D[i,j]^2 |> sqrt
    @test D2 ≈ mapslices(norm, D, dims=1)

    # functions with underscores
    @tullio n2′ = A[i]^2 |> (_)^0.5
    @test n2′ ≈ norm(A,2)

    @tullio (max) E[i] := float(B[i,j]) |> atan(_, A[i]) # i is not reduced over
    @test E ≈ vec(atan.(maximum(B, dims=2), A))

    j = 2
    @tullio G[i'] := float(B[i',j]) |> atan(_, B[i',$j])
    @test G ≈ vec(atan.(sum(B, dims=2), B[:,j]))

    @test_throws LoadError @eval @tullio F[i] := B[i,j] |> (_ / A[j]) # wrong index
    @test_throws String @tullio F[i] := B[i,j] |> (_ / C[i]) # wrong length

end

@testset "named dimensions" begin

    using NamedDims

    # reading
    N = NamedDimsArray(rand(Int8,3,10), (:r, :c))

    @tullio A[i,j] := N[i, j] + 100 * (1:10)[j]
    @test A == N .+ 100 .* (1:10)'

    @tullio B[i] := N[r=i, c=1]
    @test B == N[:,1]

    @tullio C[j,i] := N[c=j, r=i] + 100 * (1:10)[j]
    @test A == C'
    @test dimnames(C) == (:_, :_) # similar(parent(A)) avoids a bug

    # writing
    @tullio M[row=i, col=j, i=1] := (1:3)[i] // (1:7)[j]
    @test dimnames(M) == (:row, :col, :i)

end

@testset "options" begin

    # keyword threads accepts false or a positive integer
    @tullio A[i] := (1:10)[i]^2  threads=false
    @tullio A[i] := (1:10)[i]^2  threads=2^2
    # when using KernelAbstractions, something leaks from the 1st leading 2nd to error
    block = 64
    @tullio A[i] := (1:10)[i]^2  threads=block # Symbol
    @test_throws LoadError @macroexpand1 @tullio A[i] := (1:10)[i]^2  threads=:maybe

    # keyword verbose accepts values [true, false, 2, 3]
    @tullio A[i] := (1:10)[i]^2  verbose=1
    @tullio A[i] := (1:10)[i]^2  verbose=false
    @test_throws LoadError @macroexpand1 @tullio A[i] := (1:10)[i]^2  verbose=4

    # keyword grad accepts values [false, Base, Dual]
    @tullio A[i] := (1:10)[i]^2  grad=false
    @tullio A[i] := (1:10)[i]^2  grad=Base
    @test_throws LoadError @macroexpand1 @tullio A[i] := (1:10)[i]^2  grad=true

    # recognised keywords are [:threads, :verbose, :avx, :cuda, :grad]
    @test_throws LoadError @macroexpand1 @tullio A[i] := (1:10)[i]^2  key=nothing

end

@testset "bugs" begin

    # https://github.com/mcabbott/Tullio.jl/issues/10
    arr = [1 2; 3 4]
    function f10(arr)
        @tullio res1 = arr[i, k] - arr[i - 1, k]
        @tullio res2 = arr[i, k] - arr[i, k + 1]
        return res1 + res2
    end
    @test f10(arr) == 2

    let
        B = rand(3,3)
        @tullio tot = B[i, k] - B[i - 1, k]
        @test_throws UndefVarError 𝒜𝒸𝓉! isa Function
    end

end
