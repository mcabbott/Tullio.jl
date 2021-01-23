
#========== backward gradient using ForwardDiff ==========#

function insert_forward_gradient(axislist, store)
    store.finaliser == :identity || error("can't use grad=Dual with |> finaliser")

    dZ = Symbol(DEL, ZED)
    ∇act! = Symbol(:∇, ACT!)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    # gradscalars = map(A -> Symbol(DEL, A), store.scalars)
    nonshared = setdiff(vcat(store.leftind, store.redind), store.sharedind)

    epsilondict = Dict{Symbol,Expr}()

    epsilonright = MacroTools_postwalk(epsilonwalk(epsilondict, store), store.right)
    # epsilonright = MacroTools_postwalk(epsilonwalk(epsilondict, store.scalars), store.right)

    defineepsilons, readepsilons = [], []
    for (d, (Aepsilon, Aex)) in enumerate(epsilondict)
        basis = [i==d ? :($one($TYP)) : :($zero($TYP)) for i in 1:length(epsilondict)]
        push!(defineepsilons, :($Aepsilon = ForwardDiff.Dual($zero($TYP), ($(basis...),))))
        push!(readepsilons, :($Aex = $Aex + ForwardDiff.partials($ZED, $d) * $dZ[$(store.leftraw...)]))
    end

    if isempty(defineepsilons) # short-circuit
        push!(store.outpre, :(local @inline $∇act!(::Type, args...) = nothing ))
        store.verbose > 0 && @info "no gradient to calculate"
        return nothing
    end

    ex_iter = :($ZED = $(epsilonright); $(readepsilons...))

    make_many_actors(∇act!,
        vcat(gradarrays, :($dZ::AbstractArray{$TYP}), ZED, store.arrays, store.scalars, axislist),
        # vcat(gradarrays, gradscalars, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        :(($(defineepsilons...);)), store.sharedind, nothing, nonshared, ex_iter, nothing, store, "(gradient using ForwardDiff)")

    if isdefined(store.mod, :Zygote) && !(store.scalar)
        ex_iter2 = fillarrayreplace(ex_iter, dZ)
        ex_value = :($(Symbol(dZ, :_value)) = $dZ.value)

        make_many_actors(∇act!,
            vcat(gradarrays, :($dZ::Zygote.Fill{$TYP}), ZED, store.arrays, store.scalars, axislist),
            :(($(defineepsilons...); $ex_value)), store.sharedind, nothing, nonshared, ex_iter2, nothing, store, "(gradient method for FillArrays)")

        # push!(store.outeval, quote
        #     Tullio.promote_storage(T::Type, ::Type{<:Zygote.Fill}) = T
        #     Tullio.promote_storage(::Type{<:Zygote.Fill}, T::Type) = T
        # end)
    end

end

epsilonwalk(dict, store) = ex -> begin
# epsilonwalk(dict, scalars) = ex -> begin
#         ex isa Symbol && ex in scalars && return scalarplusepsilon(ex, dict)
        @capture_(ex, A_[inds__]) || return ex
        A in store.nograd && return ex
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
    @debug string("expression ", A, " is why you can't use ForwardDiff here")
    :🐳
end

# scalarplusepsilon(A::Symbol, dict) = begin
#     Aepsilon = Symbol(EPS, A)
#     dict[Aepsilon] = Symbol(DEL, A)
#     :(( $A + $Aepsilon ))
# end

#========== the end ==========#
