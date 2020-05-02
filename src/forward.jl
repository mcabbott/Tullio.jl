
#========== backward gradient using ForwardDiff ==========#

function insert_forward_gradient(act!, store)
    store.verbose && @info "using ForwardDiff for $act! ~ $(store.right[])"

    dZ = Symbol(DEL, ZED)
    ∇act! = Symbol(:∇, act!)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    # gradscalars = map(A -> Symbol(DEL, A), store.scalars)
    nonshared = setdiff(vcat(store.leftind, store.redind), store.sharedind)
    axislist = map(i -> Symbol(AXIS, i), vcat(store.sharedind, nonshared))

    epsilondict = Dict{Symbol,Expr}()

    epsilonright = MacroTools_postwalk(epsilonwalk(epsilondict), store.right)
    # epsilonright = MacroTools_postwalk(epsilonwalk(epsilondict, store.scalars), store.right)

    defineepsilons, readepsilons = [], []
    for (d, (Aepsilon, Aex)) in enumerate(epsilondict)
        basis = [i==d ? :(one($TYP)) : :(zero($TYP)) for i in 1:length(epsilondict)]
        push!(defineepsilons, :($Aepsilon = ForwardDiff.Dual(zero($TYP), ($(basis...),))))
        push!(readepsilons, :($Aex = $Aex + ForwardDiff.partials($ZED, $d) * $dZ[$(store.leftraw...)]))
    end

    ex_iter = :($ZED = $(epsilonright); $(readepsilons...))

    make_many_actors(∇act!,
        vcat(gradarrays, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        # vcat(gradarrays, gradscalars, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        :(($(defineepsilons...);)), store.sharedind, nothing, nonshared, ex_iter, nothing, store)

    if isdefined(store.mod, :Zygote)
        ex_iter2 = fillarrayreplace(ex_iter, dZ)
        ex_value = :($(Symbol(dZ, :_value)) = $dZ.value)

        make_many_actors(∇act!,
            vcat(gradarrays, :($dZ::Zygote.Fill{$TYP}), store.arrays, store.scalars, axislist),
            :(($(defineepsilons...); $ex_value)), store.sharedind, nothing, nonshared, ex_iter2, nothing, store)

        push!(store.outeval, quote
            Tullio.promote_storage(T::Type, ::Type{<:Zygote.Fill}) = T
            Tullio.promote_storage(::Type{<:Zygote.Fill}, T::Type) = T
        end)
    end

end

epsilonwalk(dict) = ex -> begin
# epsilonwalk(dict, scalars) = ex -> begin
#         ex isa Symbol && ex in scalars && return scalarplusepsilon(ex, dict)
        @capture_(ex, A_[inds__]) || return ex
        return arrayplusepsilon(A, inds, dict)
    end

arrayplusepsilon(A::Symbol, inds, dict) = begin # the same array may occur twice!
    Aepsilon = Symbol(EPS, A)
    while haskey(dict, Aepsilon)
        Aepsilon = Symbol(Aepsilon, "′")
    end
    dict[Aepsilon] = :( $(Symbol(DEL, A))[$(inds...)] )
    :(( $A[$(inds...)] + $Aepsilon ))
end
arrayplusepsilon(A, inds, dict) = begin
    @debug "expression ", string(A), " is why you can't use ForwardDiff here"
    :🐳
end

# scalarplusepsilon(A::Symbol, dict) = begin
#     Aepsilon = Symbol(EPS, A)
#     dict[Aepsilon] = Symbol(DEL, A)
#     :(( $A + $Aepsilon ))
# end

#========== the end ==========#
