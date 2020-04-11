
using Tullio, Test, LinearAlgebra
# using Tullio: LoopVectorization.@avx, storage_type

Tullio.VERBOSE[] = true
Tullio.VERBOSE[] = false

Tullio.AVX[] = false
Tullio.AVX[] = true

Tullio.GRAD[] = nothing
Tullio.GRAD[] = :ForwardDiff
Tullio.GRAD[] = :Base

@testset "new arrays" begin

    # functions
    @tullio A[i] := (1:10)[i]^2
    @test A == [i^2 for i in 1:10]

    @tullio A[i] := (1:10)[i] * i
    @test A == [i^2 for i in 1:10]

    # shifts
    @tullio B[i] := A[i+5]
    @test axes(B,1) == -4:5

    # diagonals
    @tullio D[i,i] := trunc(Int, sqrt(A[i]))
    @test D == Diagonal(sqrt.(A))

    # arrays of arrays
    C = [fill(i,3) for i=1:5]
    @tullio M[i,j] := C[i][j]
    @test M == (1:5) .* [1 1 1]

    # fields
    E = [(a=i, b=i^2, c=[i,2i,3i]) for i in 1:10]
    @tullio O[i] := A[i]//E[i].b # gets "noavx" flag from E[i].b
    @test O == ones(10)

    @tullio F[i,j] := E[i].c[j] # also "noavx"
    @test F == (1:10) .* [1 2 3]

    # scalar
    @tullio S := A[i]/2
    @test S ≈ sum(A)/2

    @tullio Z[] := A[i] + A[j]
    @test Z isa Array{Int,0}

    # fixed
    @tullio F[i] := D[i,5]
    @test F[5] == 5

    j = 6
    @tullio G[i] := D[i,$j]
    @test G[6] == 6

    @tullio H[i] := D[i,:] # storage_type(H, D) == Array, this avoids @avx
    @test H[5] == F

    # non-unique
    @tullio A2[i] := A[i] + A[i]
    @test A2 == 2 .* A

    # scope
    f(x) = @tullio y[i] := x[i] + i
    @test f(ones(3)) == 1 .+ (1:3)

    g(x) = @tullio y := sqrt(x[i])
    @test g(fill(4,5)) == 10

end

@testset "in-place" begin

    A = [i^2 for i in 1:10]
    D = similar(A, 10, 10)

    # in-place
    @tullio D[i,j] = A[i] + 100
    @test D[3,7] == A[3] + 100

    B = zero(A)
    @tullio B[i] = A[i+5] + 100  # broken!
    @test B == vcat(A[6:end] .+ 100, zeros(Int,5))

    # writing back into same
    @tullio B[i] += B[i] + 10^3
    @test B[6] == 10^3

    @tullio A[i] = A[i] + 100
    @test A[1] == 101

    # indices in expression
    @tullio A[i] = 100*i  # broken!
    @test A[7] == 700

end

# To fix:
# whether you can add *= etc easily, for compat
# demand ranges start at 1, unless Offs. loaded
# named creation

@testset "named dimensions" begin

    using NamedDims

    N = NamedDimsArray(rand(Int8,3,10), (:r, :c))

    @tullio A[i,j] := N[i, j] + 100 * (1:10)[j]

    @tullio C[j,i] := N[c=j, r=i] + 100 * (1:10)[j]
    @test_broken A == C'
    @test_broken dimnames(C) == (:_, :_)

end
