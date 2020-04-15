using Test

using Tullio: storage_type, storage_typejoin
using Zygote, ForwardDiff

@testset "storage_type" begin

    @test storage_type(rand(2), rand(2,3)) == Array{Float64,N} where N
    @test storage_type(rand(2), rand(Float32, 2)) == Array{Float64,1}
    @test_broken storage_type(rand(2), rand(Float32, 2,2)) == Array{Float64,N} where N # not good!

    Base.promote_type(Matrix{Int}, Vector{Int}) == Array{Int64,N} where N
    Base.promote_type(Matrix{Int}, Matrix{Int32}) == Matrix{Int64}
    Base.promote_type(Matrix{Int}, Vector{Int32}) == Array # != Array{Int64,N} where N

    @test storage_type(rand(2), 1:3) == AbstractArray{T,1} where T
    @test storage_type(rand(2), fill(ForwardDiff.Dual(1,0),2)) == Array{T,1} where T
    @test storage_type(rand(2), fill(ForwardDiff.Dual(1,0),2,3)) == Array
    @test storage_type(rand(2), Zygote.FillArrays.Fill(1.0, 2,2)) == AbstractArray{Float64,N} where N


    storage_typejoin(rand(2), rand(2,3)) == Array{Float64,N} where N
    storage_typejoin(rand(2), rand(Float32, 2)) == Array{T,1} where T
    storage_typejoin(rand(2), rand(Float32, 2,2)) == Array

    storage_typejoin(rand(2), 1:3) == AbstractArray{T,1} where T
    storage_typejoin(rand(2), fill(ForwardDiff.Dual(1,0),2)) == Array{T,1} where T
    storage_typejoin(rand(2), fill(ForwardDiff.Dual(1,0),2,3)) == Array
    storage_typejoin(rand(2), Zygote.FillArrays.Fill(1.0, 2,2)) == AbstractArray{Float64,N} where N

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

#=

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

=#
