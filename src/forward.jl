
#========== backward gradient using ForwardDiff ==========#

function insert_forward_gradient(create, apply!, store)
    store.verbose && @info "using ForwardDiff for $create ~ $(store.right[])"

    store.epsilonright[] = MacroTools.postwalk(epsilonwalk(store), store.right[])

        # # Version of right with (A[i,j] + 𝜀A′) etc, with dict[:𝜀A′] = A[i,j]
        # epsilonright = Ref{ExprSym}(),
        # epsilondict = Dict{Symbol,Expr}(),

    dZ = Symbol(DEL, ZED)
    ∇apply! = Symbol(:∇, apply!)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)

    # loopind = vcat(store.leftind, store.redind)
    # shared = map(i -> Symbol(AXIS, i), store.sharedind)
    # nonshared = map(i -> Symbol(AXIS, i), setdiff(loopind, store.sharedind))
    nonshared = setdiff(vcat(store.leftind, store.redind), store.sharedind)
    axislist = map(i -> Symbol(AXIS, i), vcat(store.sharedind, nonshared))

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

    ex_iter = :($ZED = $(store.epsilonright[]); $(readepsilons...))

    make_many_workers(∇apply!,
        vcat(gradarrays, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        :(($(defineepsilons...);)), store.sharedind, nothing, nonshared, ex_iter, nothing, store)

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
