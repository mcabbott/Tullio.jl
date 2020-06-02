
#========== use TensorOperations when you can ==========#
# This seems to always be faster, when applicable.
# When not, it will return nothing, and we go back the the loops.

function try_tensor(expr, ranges, store)
    outex = []
    try
        tex = macroexpand(store.mod, :(TensorOperations.@tensor $expr))

        if @capture_(expr, left_ := right_)

            #===== new array =====#

            @capture_(left, Z_[leftind__]) || error("expected A[...] := ...")

            arrays, indices, scalars = [], [], []
            MacroTools_postwalk(right) do ex
                ex isa Expr || return ex
                # Check that it only has one term -- else our conventions disagree
                if ex.head == :call && ex.args[1] in [:+, :-] && length(ex.args)>=3
                    error("@tullio can only use @tensor on expresions with one term")

                # Save array and scalar arguments
                elseif @capture_(ex, A_[ijk__])
                    push!(arrays, arrayonly(A))
                    push!(indices, ijk)
                elseif ex.head == :call && ex.args[1] == :*
                    foreach(ex.args[2:end]) do a
                        a isa Symbol && push!(scalars, a)
                    end
                end
                ex
            end

            args = unique(vcat(arrays, scalars))
            push!(outex, quote
                function $MAKE($(args...),)
                    $tex
                end
            end)

            if store.grad != false

                #===== gradients =====#

                ∇make, backdef = tensor_grad(right, arrays, indices, scalars, leftind)
                push!(outex, backdef)

                push!(outex, :( $Z = $Eval($MAKE, $∇make)($(args...)) ))
            else
                push!(outex, :( $Z = $Eval($MAKE, $nothing)($(args...)) ))
            end

        else
            #===== in-place =====#

            MacroTools_postwalk(expr) do ex
                # Check that it only has one term
                if ex isa Expr && ex.head == :call && ex.args[1] in [:+, :-] && length(ex.args)>=3
                    error("@tullio can only use @tensor on expresions with one term")
                end
                ex
            end

            push!(outex, tex)
        end
        store.verbose == 2 && verbose_tensor(outex)
        return outex

    catch err
        store.verbose > 0 && @error "TensorOperations failed" err
        return nothing
    end
end

verbose_tensor(outex) = begin
    @info "using TensorOperations"
    printstyled("    outex =\n", color=:blue)
    foreach(ex -> printstyled(Base.remove_linenums!(ex) , "\n", color=:green), outex)
end


#========== symbolic gradient ==========#
# Originally TensorGrad.jl (an unregistered package),
# all terms are again @tensor expressions.

function tensor_grad(right, arrays, indices, scalars, leftind)
    dZ = Symbol(DEL, ZED)
    ∇make = Symbol(:∇, MAKE)
    backsteps, backseen = [], []

    for (B, Binds) in zip(arrays, indices)
        deltaB = Symbol(DEL, B)

        newright, extra, ijk = replace_B_with_Δ(B, Binds, right, leftind)

        append!(backsteps, extra)

        if B in backseen
            addon = :( @tensor $deltaB[$(ijk...)] = $deltaB[$(ijk...)] + $newright )
            push!(backsteps, addon)
        else
            push!(backseen, B)
            symB = Symbol(DEL, B, '_', join(ijk))
            create = :( @tensor( $deltaB[$(ijk...)] := $newright ) )
            push!(backsteps, create)
        end
    end

    args = unique(vcat(arrays, scalars))
    backtuple = vcat(map(B -> Symbol(DEL, B), unique(arrays)), map(_ -> nothing, unique(scalars)))

    out = :(
        function $∇make($dZ, $(args...))
            $(backsteps...)
            return ($(backtuple...),)
        end
    )
    # Todo: make a version for Zygote.Fill

    ∇make, out
end


function replace_B_with_Δ(B, Bijk, right, leftind)
    dZ = Symbol(DEL, ZED)

    # If B[ijk] occurs twice this will be wrong:
    countB = 0

    # Construct the new RHS
    out = MacroTools_postwalk(right) do x
        if @capture_(x, A_[ijk__]) && A==B && ijk == Bijk
            countB += 1
            return :( $dZ[$(leftind...)] )
        else
            return x
        end
    end

    # Deal with partial traces -- repeated indices on same array
    extra, deltas = [], []
    newijk = copy(Bijk)
    if !allunique(Bijk)
        for n in 1:length(Bijk)
            i = newijk[n]
            m = findfirst(isequal(i), newijk[n+1:end])
            if m != nothing
                j = Symbol('_',i,'′')
                newijk[n] = j
                delta = Symbol("_δ_",i,j)

                # This definition is added up front:
                push!(extra, quote
                    local $delta = LinearAlgebra.Diagonal(fill!(similar($B, real(eltype($B)), size($B,$n)),true))
                end)
                # This factor is included in the new RHS:
                push!(deltas, :( $delta[$i,$j] ))
            end
        end
    end
    if length(extra) > 0
        out = :( *($out, $(deltas...)) )
    end

    # I said:
    # Gradient has indices appearing only on LHS... so you need * ones()[i,j]?

    countB > 1 && error("can't handle case of $B appearing twice with same indices")
    # Could also multiply by countB, and replace just once, would that be safe?

    return out, extra, newijk
end

#========== the end ==========#
