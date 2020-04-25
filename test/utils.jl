using Test

using Tullio: storage_type, promote_storage
using ForwardDiff, FillArrays

@testset "storage_type" begin

    @test storage_type(rand(2), rand(2,3)) == Array{Float64,N} where N
    @test storage_type(rand(2), rand(Float32, 2)) == Vector{Float64}
    @test storage_type(rand(2), rand(Float32, 2,2)) == Array{Float64,N} where N

    Base.promote_type(Matrix{Int}, Vector{Int}) == Array{Int64,N} where N
    Base.promote_type(Matrix{Int}, Matrix{Int32}) == Matrix{Int64}
    Base.promote_type(Matrix{Int}, Vector{Int32}) == Array # != Array{Int64,N} where N
    promote_storage(Matrix{Int}, Vector{Int32}) == Array{Int64,N} where N

    @test storage_type(rand(2), 1:3) == Vector{Float64}
    @test storage_type(rand(Int,2), 1:3) == Vector{Int}
    @test storage_type(1:3.0, 1:3) <: AbstractRange{Float64}

    @test storage_type(rand(2), fill(ForwardDiff.Dual(1,0),2)) == Vector{ForwardDiff.Dual{Nothing,Float64,1}}
    @test storage_type(rand(2), fill(ForwardDiff.Dual(1,0),2,3)) == Array{ForwardDiff.Dual{Nothing,Float64,1}}
    @test storage_type(rand(2), FillArrays.Fill(1.0, 2,2)) == AbstractArray{Float64,N} where N
    @test storage_type(rand(2), FillArrays.Fill(true, 2,2)) == AbstractArray

end

using Tullio: range_expr_walk, divrange, minusrange, subranges, addranges

@testset "range_expr_walk" begin

    for r in [Base.OneTo(10), 0:10, 0:11, 0:12, -1:13]
        for (f, ex) in [
            # +
            (i -> i+1, :(i+1)),
            (i -> i+2, :(i+2)),
            (i -> 3+i, :(3+i)),
            # -
            (i -> -i, :(-i)),
            (i -> i-1, :(i-1)),
            (i -> 1-i, :(1-i)),
            (i -> 2-i, :(2-i)),
            (i -> 1+(-i), :(1+(-i))),
            (i -> -i+1, :(-i+1)),
            (i -> -i-1, :(-i-1)),
            (i -> 1-(2-i), :(1-(2-i))),
            (i -> 1-(-i+2), :(1-(-i+2))),
            # *
            (i -> 2i, :(2i)),
            (i -> 2i+1, :(2i+1)),
            (i -> -1+2i, :(-1+2i)),
            (i -> 1-3i, :(1-3i)),
            (i -> 1-3(i+4), :(1-3(i+4))),
            # ÷
            (i -> i÷2, :(i÷2)),
            (i -> 1+i÷3, :(1+i÷3)),
            # triple...
            (i -> i+1+2, :(i+1+2)),
            (i -> 1+2+i, :(1+2+i)),
            (i -> 2i+3+4, :(2i+3+4)),
            (i -> 1+2+3i+4, :(1+2+3i+4)),
            (i -> 1+2+3+4(-i), :(1+2+3+4(-i))),
            # evil
            (i -> (2i+1)*3+4, :((2i+1)*3+4)),
            (i -> 3-(-i)÷2, :(3-(-i)÷2)), # needs divrange_minus
            ]
            rex, i = range_expr_walk(:($r .+ 0), ex)
            @test issubset(sort(f.(eval(rex))), r)
        end
        @test minusrange(r) == divrange(r, -1)

        @test issubset(subranges(r, 1:3) .+ 1, r)
        @test issubset(subranges(r, 1:3) .+ 3, r)
        @test union(subranges(r, 1:3) .+ 1, subranges(r, 1:3) .+ 3) == r

        @test issubset(addranges(r, 1:3) .- 1, r)
        @test issubset(addranges(r, 1:3) .- 3, r)
        @test sort(union(addranges(r, 1:3) .- 1, addranges(r, 1:3) .- 3)) == r
    end
end

using Tullio: cleave, quarter, productlength

@testset "threading" begin
    @test cleave((1:10, 1:4, 7:8)) == ((1:4, 1:4, 7:8), (5:10, 1:4, 7:8))
    @test cleave((7:8, 9:9)) == ((7:7, 9:9), (8:8, 9:9))

    @test quarter((1:10, 11:20)) == ((1:4, 11:14), (1:4, 15:20), (5:10, 11:14), (5:10, 15:20))
    @test sum(productlength, quarter((1:10, 11:20))) == 100

    for r1 in [1:10, 3:4, 5:5], r2 in [11:21, -3:-2, 0:0], r3 in [2:7, 8:9, 0:0]
        tup = (r1,r2,r3)
        len = productlength(tup)
        @test len == sum(productlength, cleave(tup))
        @test len == sum(productlength, quarter(tup))
    end
end

using Tullio: @capture_

@testset "capture_ macro" begin
    EXS  = [:(A[i,j,k]),  :(B{i,2,:}),  :(C.dee), :(fun(5)),   :(g := h+i),        :(k[3] += l[4]), :([m,n,0]) ]
    PATS = [:(A_[ijk__]), :(B_{ind__}), :(C_.d_), :(f_(arg_)), :(left_ := right_), :(a_ += b_),     :([emm__]) ]
    # @test length(EXS) == length(PATS)
    @testset "ex = $(EXS[i])" for i in eachindex(EXS)
        for j in eachindex(PATS)
        @eval res = @capture_($EXS[$i], $(PATS[j]))
        if i != j
            @test res == false
        else
            @test res == true
            if i==1
                @test A == :A
                @test ijk == [:i, :j, :k]
            elseif i==3
                @test C == :C
                @test d == :dee
            elseif i==5
                @test left == :g
                @test right == :(h+i)
            elseif i==7
                @test emm == [:m, :n, 0]
            end
        end
        end
    end

end

using Tullio: leibnitz
using ForwardDiff

@testset "symbolic gradients" begin

    @testset "ex = $ex" for ex in [
        :(x*y + 1/z),
        :(x*y + z*x*z^2),
        :((x/y)^2 + inv(z)),
        :((x+2y)^z),
        :(1/(x+1) + 33/(22y) -4/(z/4)),
        :(inv(x+y) + z^(-2)),

        :(sqrt(x) + 1/sqrt(y+2z)),
        :(inv(sqrt(x)*sqrt(y)) + sqrt(2*inv(z))),
        :(x * z / sqrt(y * z)),

        :(log(x/y) - log(z+2)),
        :(log(x*y*z) - 33y),

        :(2exp(x*y*z)),
        :(exp((x-y)^2/2)/z),
    ]

        dfdx = leibnitz(ex, :x)
        dfdy = leibnitz(ex, :y)
        dfdz = leibnitz(ex, :z)

        @eval f_(x,y,z) = $ex
        @eval f_x(x,y,z) = $dfdx
        @eval f_y(x,y,z) = $dfdy
        @eval f_z(x,y,z) = $dfdz

        xyz = rand(Float32, 3)

        # check correctness
        gx, gy, gz = ForwardDiff.gradient(xyz -> f_(xyz...), xyz)
        @test f_x(xyz...) ≈ gx
        @test f_y(xyz...) ≈ gy
        @test f_z(xyz...) ≈ gz

        # don't accidentally make Float64
        @test 0f0 + f_x(xyz...) isa Float32
        @test 0f0 + f_y(xyz...) isa Float32
        @test 0f0 + f_z(xyz...) isa Float32

    end

end

