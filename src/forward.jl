
#========== backward gradient using ForwardDiff ==========#

function insert_forward_gradient(create, apply!, store)

    store.epsilonright[] = MacroTools.postwalk(epsilonwalk(store), store.right[])

        # # Version of right with (A[i,j] + 𝜀A′) etc, with dict[:𝜀A′] = A[i,j]
        # epsilonright = Ref{ExprSym}(),
        # epsilondict = Dict{Symbol,Expr}(),

    dZ = Symbol(DEL, ZED)
    worker! = Symbol(:∇, apply!)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)

    loopind = vcat(store.leftind, store.redind)
    shared = map(i -> Symbol(AXIS, i), store.sharedind)
    nonshared = map(i -> Symbol(AXIS, i), setdiff(loopind, store.sharedind))

    # defineepsilons = map(enumerate(store.epsilondict)) do (d, (Aepsilon,_))
    #     tup = ntuple(i -> i==d ? 1 : 0, length(store.epsilondict))
    #    :($Aepsilon = $ForwardDiff.Dual(0, $tup))
    # end
    # readepsilons = map(enumerate(store.epsilondict)) do (d, (_,Aex))
    #     :($Aex += $ForwardDiff.partials($ZED, $d) * $dZ[$(store.leftraw...)])
    # end
    defineepsilons, readepsilons = [], []
    for (d, (Aepsilon, Aex)) in enumerate(store.epsilondict) # order isn't consistent? so do it once
        basis = [i==d ? :(one($TYP)) : :(zero($TYP)) for i in 1:length(store.epsilondict)]
        push!(defineepsilons, :($Aepsilon = $ForwardDiff.Dual(zero($TYP), ($(basis...),))))
        push!(readepsilons, :($Aex = $Aex + $ForwardDiff.partials($ZED, $d) * $dZ[$(store.leftraw...)]))
        # push!(readepsilons, :($Aex = $Aex + $ZED.partials.values[$d] * $dZ[$(store.leftraw...)])) # doesn't work with avx
    end

    ex = :($ZED = $(store.epsilonright[]); $(readepsilons...))
    loopex = recurseloops(ex, (loop = loopind, store...)) # do this in two steps, to keep $dZ[$(store.leftraw...)] around?

    push!(store.outex, quote
        function $worker!($(gradarrays...), ::Type, $dZ::AbstractArray{$TYP}, $(store.arrays...), $(store.scalars...), $(shared...), $(nonshared...), ) where {$TYP}
            $(defineepsilons...)
            @fastmath @inbounds $loopex
        end
    end)

    if AVX[] && !(:noavx in store.flags)
        LoopVecTypes = Union{Float64,Float32,Int64,Int32,Int8}
        push!(store.outex, quote
            function $worker!($(gradarrays...), ::Type{<:Array{<:$LoopVecTypes}}, $dZ::AbstractArray{$TYP}, $(store.arrays...), $(store.scalars...), $(shared...), $(nonshared...), ) where {$TYP}
                $(defineepsilons...)
                $LoopVectorization.@avx $loopex
            end
        end)
    end
    # to special-case dZ::FillArray, you'd need to build a different readepsilons ... loopex
    # Or you'd have to edit it:
    # fillarrayloop = MacroTools.postwalk(loopex) do ex
    #     x == :($dZ[$(store.leftraw...)]) ? :($dZ.val) : ex  # ??
    # end
    # And you'd have to make storage_type not trip on this.

end


epsilonwalk(store) = ex -> begin
        @capture(ex, A_[inds__]) || return ex
        return arrayplusepsilon(A, inds, store)
    end

arrayplusepsilon(A::Symbol, inds, store) = begin # the same array may occur twice!
    Aepsilon = Symbol(EPS, A)
    while haskey(store.epsilondict, Aepsilon)
        Aepsilon = Symbol(Aepsilon, "′")
    end
    store.epsilondict[Aepsilon] = :( $(Symbol(DEL, A))[$(inds...)] )
    :(( $A[$(inds...)] + $Aepsilon ))
end
arrayplusepsilon(A, inds, store) = begin
    push!(store.flags, :nograd)
    @debug "expression ", string(A), " is the problem"
    :🐳
end


#========== making ForwardDiff work with LoopVectorization ==========#

# using Tullio.LoopVectorization: LoopVectorization, SVec, vconvert, SIMDPirates
using LoopVectorization
using LoopVectorization: SVec, vconvert, SIMDPirates
using Core: VecElement

s1 = SVec{2,Float64}(5.5, 6.6) # SVec{2,Float64}<5.5, 6.6>
# dump(s1)
# SVec{2,Float64}
#   data: Tuple{VecElement{Float64},VecElement{Float64}}
#     1: VecElement{Float64}
#       value: Float64 5.5
#     2: VecElement{Float64}
#       value: Float64 6.6
s1[2]

@inline svec(tup::NTuple{N,T}) where {N,T} = SVec{N,T}(tup...)

# Base.inv(sv::SVec{N,<:Integer}) where {N} = svec(ntuple(n -> inv(sv[n]), N))
# Base.sqrt(sv::SVec{N,<:Integer}) where {N} = svec(ntuple(n -> sqrt(sv[n]), N))
# Base.trunc(T::Type, sv::SVec{N}) where {N} = svec(ntuple(n -> trunc(T, sv[n]), N))
@inline Base.inv(sv::SVec{N,<:Integer}) where {N} = svec(ntuple(n -> inv(sv[n]), N))
@inline Base.sqrt(sv::SVec{N,<:Integer}) where {N} = svec(ntuple(n -> sqrt(sv[n]), N))
@inline Base.trunc(T::Type, sv::SVec{N}) where {N} = svec(ntuple(n -> trunc(T, sv[n]), N))

using ForwardDiff
using ForwardDiff: Dual, Partials, partials

d1 = Dual(1.23, (4,0,0))
typeof(d1) # Dual{Nothing,Float64,3}
# dump(d1)
# Dual{Nothing,Float64,2}
#   value: Float64 1.23
#   partials: Partials{2,Float64}
#     values: Tuple{Float64,Float64}
#       1: Float64 4.0
#       2: Float64 0.0
#       3: Float64 0.0
d1.partials # Partials{3,Float64}
d1.partials[1]

partials(d1, 1)
@inline val(d::Dual) = d.value
# @inline vals(d::Partials) = d.values

ForwardDiff.can_dual(::Type{<:SVec}) = true

#=

# TAKE ONE -- these explicitly go via scalar Duals, thinking to exploit existing scalar defns.
for f in [:+, :*, :-, :/]
    @eval begin

        # dual(number) * svec

        function Base.$f(x::Dual{Z,T,D}, sv::SVec{N}) where {Z,T<:Number,D,N}
            duals = ntuple(n -> $f(x, sv[n]), N)
            Dual(svec(val.(duals)), ntuple(d -> svec(partials.(duals, d)), D))
        end
        function Base.$f(sv::SVec{N}, x::Dual{Z,T,D}, ) where {Z,T<:Number,D,N}
            duals = ntuple(n -> $f(sv[n], x), N)
            Dual(svec(val.(duals)), ntuple(d -> svec(partials.(duals, d)), D))
        end

        # dual(svec) * svec

        function Base.$f(x::Dual{Z,SVec{N,T},D}, sv::SVec{N}) where {Z,T,D,N}
            duals = ntuple(N) do n
                xn = Dual(val(x)[n], ntuple(d -> partials(x,d)[n] ,D))
                res = $f(xn, sv[n])
            end
            Dual(svec(val.(duals)), ntuple(d -> svec(partials.(duals,d)), D))
        end
        function Base.$f(sv::SVec{N}, x::Dual{Z,SVec{N,T},D}, ) where {Z,T,D,N}
            duals = ntuple(N) do n
                xn = Dual(val(x)[n], ntuple(d -> partials(x,d)[n] ,D))
                res = $f(sv[n], xn)
            end
            Dual(svec(val.(duals)), ntuple(d -> svec(partials.(duals,d)), D))
        end

        # partials(svec) * svec

        function Base.$f(p::Partials{D,SVec{N,T}}, sv::SVec{N}) where {Z,T,D,N}
            ps = ntuple(N) do n
                pn = Partials(ntuple(d -> p[d][n] ,D))
                res = $f(pn, sv[n])
            end
            Partials(ntuple(d -> svec(getindex.(ps,d)), D))
        end
        function Base.$f(sv::SVec{N}, p::Partials{D,SVec{N,T}}) where {Z,T,D,N}
            ps = ntuple(N) do n
                pn = Partials(ntuple(d -> p[d][n] ,D))
                res = $f(sv[n], pn)
            end
            Partials(ntuple(d -> svec(getindex.(ps,d)), D))
        end

    end
end

=#
#=
# TAKE TWO -- stay in the land of the SVecs

Base.:+(x::Dual{Z,T,D}, sv::SVec{N}) where {Z,T,D,N} =
    Dual(val(x) + sv, ntuple(d -> partials(x,d) + zero(sv), D)) # is * zero(sv) wasteful?
Base.:+(sv::SVec{N}, x::Dual{Z,T,D}) where {Z,T,D,N} =
    Dual(val(x) + sv, ntuple(d -> partials(x,d) + zero(sv), D))

Base.:*(x::Dual{Z,T,D}, sv::SVec{N}) where {Z,T,D,N} =
    Dual(val(x) * sv, ntuple(d -> partials(x,d) * sv, D))
Base.:*(sv::SVec{N}, x::Dual{Z,T,D}) where {Z,T,D,N} =
    Dual(sv * val(x), ntuple(d -> sv * partials(x,d), D))

Base.:*(p::Partials{D,SVec{N,T}}, sv::SVec{N}) where {Z,T,D,N} =
    Partials(ntuple(d -> p.values[d] * sv, D))
Base.:*(sv::SVec{N}, p::Partials{D,SVec{N,T}}) where {Z,T,D,N} =
    Partials(ntuple(d -> sv * p.values[d], D))

=#
#=
# TAKE THREE -- more stable?

function Base.:+(x::Dual{Z,T,D}, sv::SVec{N,S}) where {Z,T,D,N,S}
    y = x.value + sv # scalar or svec + svec
    ps = ntuple(d -> partials(x,d) + zero(sv), Val(D))
    Dual{Z,typeof(y),D}(y, Partials{D,typeof(y)}(ps))
end
function Base.:+(sv::SVec{N,S}, x::Dual{Z,T,D}) where {Z,T,D,N,S}
    y = x.value + sv
    ps = ntuple(d -> partials(x,d) + zero(sv), Val(D))
    Dual{Z,typeof(y),D}(y, Partials{D,typeof(y)}(ps))
end

function Base.:*(x::Dual{Z,T,D}, sv::SVec{N,S}) where {Z,T,D,N,S}
    y = x.value * sv
    ps = ntuple(d -> partials(x,d) * sv, Val(D))
    Dual{Z,typeof(y),D}(y, Partials{D,typeof(y)}(ps))
end
function Base.:*(sv::SVec{N,S}, x::Dual{Z,T,D}) where {Z,T,D,N,S}
    y = sv * x.value
    ps = ntuple(d -> sv * partials(x,d), Val(D))
    Dual{Z,typeof(y),D}(y, Partials{D,typeof(y)}(ps))
end

function Base.:*(p::Partials{D,SVec{N,T}}, sv::SVec{N,S}) where {Z,T,D,N,S}
    TS = SVec{N,promote_type(T,S)}
    Partials{D,TS}(ntuple(d -> p.values[d] * sv, Val(D)))
end
function Base.:*(sv::SVec{N,S}, p::Partials{D,SVec{N,T}}) where {Z,T,D,N,S}
    TS = SVec{N,promote_type(T,S)}
    Partials{D,TS}(ntuple(d -> sv * p.values[d], Val(D)))
end

=#
# TAKE FOUR -- avoid typeof(y), these don't cover as many cases, but enough?

@inline function Base.:+(x::Dual{Z,T,D}, sv::SVec{N,S}) where {Z,T<:Number,D,N,S}
    y = x.value + sv
    ps = ntuple(d -> x.partials.values[d] + zero(sv), Val(D)) # avoiding partials(x.d) didn't help
    TS = SVec{N,promote_type(T,S)}
    Dual{Z,TS,D}(y, Partials{D,TS}(ps))
end
@inline function Base.:+(sv::SVec{N,S}, x::Dual{Z,T,D}) where {Z,T<:Number,D,N,S}
    y = x.value + sv
    ps = ntuple(d -> x.partials.values[d] + zero(sv), Val(D))
    TS = SVec{N,promote_type(T,S)}
    Dual{Z,TS,D}(y, Partials{D,TS}(ps))
end

@inline function Base.:*(x::Dual{Z,SVec{N,T},D}, sv::SVec{N,S}) where {Z,T,D,N,S}
    y = x.value * sv
    ps = ntuple(d -> x.partials.values[d] * sv, Val(D))
    TS = SVec{N,promote_type(T,S)}
    Dual{Z,typeof(y),D}(y, Partials{D,typeof(y)}(ps))
end
@inline function Base.:*(sv::SVec{N,S}, x::Dual{Z,SVec{N,T},D}) where {Z,T,D,N,S}
    y = sv * x.value
    ps = ntuple(d -> sv * x.partials.values[d], Val(D))
    TS = SVec{N,promote_type(T,S)}
    Dual{Z,TS,D}(y, Partials{D,TS}(ps))
end

@inline function Base.:*(p::Partials{D,SVec{N,T}}, sv::SVec{N,S}) where {Z,T,D,N,S}
    TS = SVec{N,promote_type(T,S)}
    Partials{D,TS}(ntuple(d -> p.values[d] * sv, Val(D)))
end
@inline function Base.:*(sv::SVec{N,S}, p::Partials{D,SVec{N,T}}) where {Z,T,D,N,S}
    TS = SVec{N,promote_type(T,S)}
    Partials{D,TS}(ntuple(d -> sv * p.values[d], Val(D)))
end

#========== the end ==========#

#=

function f1(A, B)
    d = Dual(0, (1, 0))
    LoopVectorization.@avx for i in eachindex(B)
        res = (A[i] + d) * (A[i] + d) # + 1 this breaks it
        B[i] = partials(res, 1)
    end
    B
end
f1(collect(1:4), rand(1:99, 4)) # ok!
f1(collect(1:4), rand(4)) # ok!
f1(collect(1:4.0), rand(4)) # ok!

function f2(A, B)
    d = Dual(0, (1, 0))
    LoopVectorization.@avx for i in eachindex(B)
        res = log(A[i] + d) * A[i]
        B[i] = partials(res, 1)
    end
    B
end
f2(collect(1:4.0), rand(4)) # ok!
f2(collect(1:4), rand(4)) # ok!


function f3(A, B, C)
    d = Dual(0, (1, 0))
    d2 = Dual(0, (0, 1))
    LoopVectorization.@avx for i in eachindex(1)
        res = log((A[i] + d)/(B[i] + d2)) * (A[i] + d)
        C[i] = partials(res, 1) + partials(res, 2) * B[i]
    end
    C
end
f3(collect(1:4.0), ones(4), rand(4))

=#

#=

function plus1(x::Dual{Z,T,D}, sv::SVec{N}) where {Z,T<:Number,D,N} # take I above
    duals = ntuple(n -> x + sv[n], N)
    Dual(svec(val.(duals)), ntuple(d -> svec(partials.(duals, d)), D))
end

plus2(x::Dual{Z,T,D}, sv::SVec{N}) where {Z,T,D,N} = # take II above
    Dual(val(x) + sv, ntuple(d -> partials(x,d) + zero(sv), D))

function plus3(x::Dual{Z,T,D}, sv::SVec{N,S}) where {Z,T<:Number,D,N,S}
    TS = SVec{N,promote_type(T,S)}
    Dual{Z,TS,D}(val(x) + sv, Partials{D,TS}(ntuple(d -> partials(x,d) + zero(sv), Val(D))))
end

# function plus4(x::Dual{Z,T,D}, sv::SVec{N,T}) where {Z,T,D,N}
#     Dual{Z,typeof(sv),D}(val(x) + sv, Partials{D,typeof(sv)}(ntuple(d -> partials(x,d) + zero(sv), D)))
# end

function plus5(x::Dual{Z,T,D}, sv::SVec{N,S}) where {Z,T,D,N,S}
    res = val(x) + sv
    Dual{Z,typeof(res),D}(res, Partials{D,typeof(res)}(ntuple(d -> partials(x,d) + zero(sv), Val(D))))
end

plus1(d1, s1)
plus2(d1, s1)
plus3(d1, s1)
plus5(d1, s1)

@code_warntype plus1(d1, s1) # ok!
@code_warntype plus2(d1, s1) # Body::Dual{Nothing,SVec{2,Float64},_A} where _A
@code_warntype plus3(d1, s1) # ok!
@code_warntype plus5(d1, s1) # ok!

# "plus5" became take III above, check that all are indeed stable:

@code_warntype d1 + s1
@code_warntype s1 + d1

@code_warntype (d1 + s1) * s1
@code_warntype s1 * (d1 + s1)

p1 = (d1 + s1).partials
@code_warntype p1 * s1
@code_warntype s1 * p1

=#
