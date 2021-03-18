
#========== use TensorOperations when you can ==========#

"""
    Tullio.@tensor C[i,j] := A[i,k] * B[k,j]

This is a way to run `TensorOperations.@tensor`, whose only advantage is 
that it provides gradient definitions. (This code is part of Tullio.jl a bit by accident.)

This is less flexible than `TensorOperations.@tensor`, and in particular it accepts 
only a single term on the RHS. It is much less flexible that `@tullio`, but will 
sometimes be faster.
"""
macro tensor(exs...)
    opts, ranges, ex = parse_options(exs...)
    
    isempty(ranges) || throw("@tensor does not accept explicit index ranges")
    opts.redfun == :+ || throw("@tensor only reduces over +")
    opts.initkeyword == TYP || throw("@tensor does not accept init keyword")

    res = try_tensor(ex, ranges, DotDict(; mod = __module__, opts...,
        newarray = false, scalar = false,
        arrays = Symbol[], indices = [], scalars = Symbol[]))

    Expr(:block, res...) |> esc
end

function try_tensor(expr, ranges, store)

    fail = nothing
    if isexpr(expr, [:(:=), :(=), :(+=)])
    else
        fail = "@tensor expected left := right etc"
    end
    if @capture_(expr.args[1], Z_[leftind__]) && all(a -> a isa Symbol, leftind)
        if expr.head == :(:=)
            store.newarray = true
        end
    elseif expr.args[1] isa Symbol # scalar output
        store.scalar = true # not used?
        Z, leftind = expr.args[1], nothing
        if expr.head == :(:=)
            store.newarray = true
            expr.head = :(=) # mutate it as @tensor doesn't accept scalar :=
        elseif expr.head == :(=)
            store.newarray = true
        end # for scalars, only += case isn't :newarray
    else
        fail = "@tensor expected A[i,j,k] := ..."
    end
    MacroTools_postwalk(expr.args[2]) do ex
        ex isa Expr || return ex
        if ex.head == :call && ex.args[1] == :* && all(a -> a isa Expr || a isa Number, ex.args[2:end])
            # Todo: allow A[i] * $c
        elseif ex.head == :ref && all(a -> a isa Symbol, ex.args)
        elseif ex.head == :call && ex.args[1] in [:+, :-] && length(ex.args)==2
            # Allows -A[i]. Could allow conj() too, but gradient would be wrong.
        else
            # Disallows anything containing +, since A[i] + B[i,k,k] has differing meanings.
            fail = "Tullio.@tensor can't handle $(ex)"
        end
        ex
    end
    fail != nothing && throw(fail)

    outex = [] # you could simplify, only one expression really
    # try
        tex = macroexpand(store.mod, :(TensorOperations.@tensor $expr))

        if store.newarray
            left, right = expr.args
            #===== new array =====#

            MacroTools_postwalk(right) do ex
                ex isa Expr || return ex
                # Save array and scalar arguments
                if @capture_(ex, A_[ijk__])
                    A1 = arrayonly(A)
                    push!(store.arrays, A1)
                    push!(store.indices, ijk)
                    n = length(ijk)
                    str = "expected a $n-array $A1"
                    push!(outex, :( $ndims($A1) == $n || $error($str) ))
                elseif ex.head == :call && ex.args[1] == :*
                    foreach(ex.args[2:end]) do a
                        a isa Symbol && push!(store.scalars, a)
                    end
                end
                ex
            end

            if store.grad == false
                push!(outex, tex)
            else
                args = unique(vcat(store.arrays, store.scalars))
                push!(outex, quote
                    local function $MAKE($(args...),)
                        local $Z
                        $tex
                    end
                end)

                ∇make, backdefs = tensor_grad(right, leftind, store)
                append!(outex, backdefs)
                outex = [:($Z = let
                    $(outex...)
                    $Eval($MAKE, $∇make)($(args...))
                end)]
            end
        else
            #===== in-place =====#
            push!(outex, tex)
        end

        # @tensor may return "throw(TensorOperations.IndexError("non-matching indices ..."
        for line in outex
            MacroTools_postwalk(line) do ex
                isexpr(ex, :call) && ex.args[1] == :throw && error(string(ex.args[2]))
                ex
            end
        end
        store.verbose>1 && verbose_tensor(outex, store)
        return outex

    # catch err
    #     store.verbose>0 && @warn "TensorOperations failed" err
    #     return nothing
    # end
end

verbose_tensor(outex, store) = begin
    printstyled("TensorOperations outex =\n", color=:blue)
    foreach(ex -> printstyled(Base.remove_linenums!(ex) , "\n", color=:green), outex)
    verboseprint(store)
end



#========== symbolic gradient ==========#
# Originally TensorGrad.jl (an unregistered package),
# all terms are again @tensor expressions.

function tensor_grad(right, leftind, store)
    dZ = Symbol(DEL, ZED)
    ∇make = Symbol(:∇, MAKE)
    backsteps, backseen = [], []

    for (B, Binds) in zip(store.arrays, store.indices)
        deltaB = Symbol(DEL, B)

        newright, extra, ijk = replace_B_with_Δ(B, Binds, right, leftind)

        append!(backsteps, extra)

        if B in backseen
            addon = macroexpand(store.mod, :( @tensor $deltaB[$(ijk...)] = $deltaB[$(ijk...)] + $newright ))
            push!(backsteps, addon)
            store.verbose>0 && @info "gradient @tensor $deltaB[$(join(ijk,','))] += $newright"
        else
            push!(backseen, B)
            symB = Symbol(DEL, B, '_', join(ijk))
            create = macroexpand(store.mod, :( @tensor( $deltaB[$(ijk...)] := $newright ) ))
            push!(backsteps, create)
            store.verbose>0 && @info "gradient @tensor $deltaB[$(join(ijk,','))] := $newright"
        end
    end

    args = unique(vcat(store.arrays, store.scalars))
    backtuple = vcat(
        map(B -> Symbol(DEL, B), unique(store.arrays)),
        map(_ -> nothing, unique(store.scalars)),
        )

    outex = [:(
        local function $∇make($dZ, $ZED, $(args...))
            $(backsteps...)
            return ($(backtuple...),)
        end
    )]

    if !isnothing(leftind) && isdefined(store.mod, :Zygote) # special case for FillArrays
        # backsteps_fill = fillarrayreplace(backsteps, dZ)
        # ex_value = :($(Symbol(dZ, :_value)) = $dZ.value)
        push!(outex, :(
            local $∇make($dZ::Zygote.Fill, $ZED, $(args...)) = $∇make(collect($dZ), $ZED, $(args...))
            # Todo: make this work without collect! Ideally it would write simpler @tensor expr.
            # local function $∇make($dZ::Zygote.Fill, $ZED, $(args...))
            #     $ex_value
            #     $(backsteps_fill...)
            #     return ($(backtuple...),)
            # end
        ))
    end

    ∇make, outex
end

using LinearAlgebra

function replace_B_with_Δ(B, Bijk, right, leftind)
    dZ = Symbol(DEL, ZED)

    # If B[ijk] occurs twice this will be wrong:
    countB = 0

    # Construct the new RHS
    out = MacroTools_postwalk(right) do x
        if @capture_(x, A_[ijk__]) && A==B && ijk == Bijk
            countB += 1
            if isnothing(leftind)
                return :(conj($dZ)) # scalar case
            else
                return :( conj($dZ[$(leftind...)]) )
            end
        else
            return x
        end
    end
    out = :(conj($out))

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
                    local $delta = $Diagonal(fill!(similar($B, real(eltype($B)), size($B,$n)),true))
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

    countB > 1 && throw("can't handle case of $B appearing twice with same indices")
    # Could also multiply by countB, and replace just once, would that be safe?

    return out, extra, newijk
end

#========== the end ==========#
