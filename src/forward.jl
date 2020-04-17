
#========== backward gradient using ForwardDiff ==========#

function insert_forward_gradient(create, apply!, store)
    store.verbose && @info "using ForwardDiff for $create ~ $(store.right[])"

    store.epsilonright[] = MacroTools_postwalk(epsilonwalk(store), store.right[])

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
    #    :($Aepsilon = ForwardDiff.Dual(0, $tup))
    # end
    # readepsilons = map(enumerate(store.epsilondict)) do (d, (_,Aex))
    #     :($Aex += ForwardDiff.partials($ZED, $d) * $dZ[$(store.leftraw...)])
    # end
    defineepsilons, readepsilons = [], []
    for (d, (Aepsilon, Aex)) in enumerate(store.epsilondict) # order isn't consistent? so do it once
        basis = [i==d ? :(one($TYP)) : :(zero($TYP)) for i in 1:length(store.epsilondict)]
        push!(defineepsilons, :($Aepsilon = ForwardDiff.Dual(zero($TYP), ($(basis...),))))
        push!(readepsilons, :($Aex = $Aex + ForwardDiff.partials($ZED, $d) * $dZ[$(store.leftraw...)]))
        # push!(readepsilons, :($Aex = $Aex + $ZED.partials.values[$d] * $dZ[$(store.leftraw...)])) # doesn't work with avx
    end

    ex_iter = :($ZED = $(store.epsilonright[]); $(readepsilons...))

    make_many_workers(∇apply!,
        vcat(gradarrays, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        :(($(defineepsilons...);)), store.sharedind, nothing, nonshared, ex_iter, nothing, store)

    # to special-case dZ::FillArray, you'd need to build a different readepsilons ... loopex
    # Or you'd have to edit it:
    # fillarrayloop = MacroTools_postwalk(loopex) do ex
    #     x == :($dZ[$(store.leftraw...)]) ? :($dZ.val) : ex  # ??
    # end
    # And you'd have to make storage_type not trip on this.

end


epsilonwalk(store) = ex -> begin
        @capture_(ex, A_[inds__]) || return ex
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

#========== the end ==========#
