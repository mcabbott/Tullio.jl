
#========== making ForwardDiff work with LoopVectorization ==========#

# using Tullio.LoopVectorization: LoopVectorization, SVec, vconvert, SIMDPirates
using .LoopVectorization
using .LoopVectorization: SVec, vconvert, SIMDPirates
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
s1 |> typeof |> parentmodule # VectorizationBase

# @inline svec(tup::NTuple{N,T}) where {N,T} = SVec{N,T}(tup...)

using .ForwardDiff
using .ForwardDiff: Dual, Partials, partials

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
# @inline val(d::Dual) = d.value

ForwardDiff.can_dual(::Type{<:SVec}) = true

@inline function Base.:+(x::Dual{Z,T,D}, sv::SVec{N,S}) where {Z,T<:Number,D,N,S}
    y = x.value + sv
    ps = ntuple(d -> x.partials.values[d] + zero(sv), Val(D))
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

@inline function Base.:*(p::Partials{D,SVec{N,T}}, sv::SVec{N,S}) where {T,D,N,S}
    TS = SVec{N,promote_type(T,S)}
    Partials{D,TS}(ntuple(d -> p.values[d] * sv, Val(D)))
end
@inline function Base.:*(sv::SVec{N,S}, p::Partials{D,SVec{N,T}}) where {T,D,N,S}
    TS = SVec{N,promote_type(T,S)}
    Partials{D,TS}(ntuple(d -> sv * p.values[d], Val(D)))
end

#========== the end ==========#

