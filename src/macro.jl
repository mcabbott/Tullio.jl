
#========== the macro! ==========#

"""
    @tullio C[i,k] := A[i,j] * B[j,k]
    @tullio F[i,k] := \$α * D[i].field[j] * E[col=k, row=j] + \$β

This is a replacement for `@einsum` which understands a bit more syntax.
The expression on the right is summed over all possible valued of the free index `k`,
and `:=` makes a new array `C`, while `=` and `+=` would write into an existing one.
Scalar arguments should have a dollar sign, like `\$α` or `A[i,\$γ]`.

    @tullio G[i,j] := A[i+x+1, j+y+1] * K[x,y]
    @tullio H[i,j] := A[2i+x, 2j+y]  (x in -1:1, y in -1:1)

Shifts and scaling of indices are allowed, including shifts by other indices.
Ranges can be provided as shown, for under-constrained indices.
If they are over-constrained, shifted indices run over the intersection allowed by all constraints,
while un-shifted indices demand agreement between them (e.g. `axes(A,2) == axes(B,1)` above).

    @tullio (*) L[i] := A[J[k]+2, i] / B[k]^2

This is a product instead of a sum, which could also enabled by writing `L[i] *= ...` (in-place).
You can use any reduction function such as `@tullio (max) M[i,j] := ...`.
Indexing by `J[k]+2` here demands `issubset(J, axes(A,1) .- 2)`.

    @tullio N[j] := sqrt <| M[i,j]^2

Pipe operators `|>` and `<|` apply a function after the sum, here `N ≈ map(norm, eachcol(M))`.
Underscores create functions, e.g. `|> sqrt(_ / V[i])` where clearly `i` must not have been summed.

See the readme for further options.
"""
macro tullio(exs...)
    _tullio(exs...; mod=__module__)
end

function _tullio(exs...; mod=Main)

    opts, ranges, ex = parse_options(exs...)
    if isnothing(ex) # then we simply updated global settings
        return (verbose=_VERBOSE[], fastmath=_FASTMATH[], threads=_THREADS[], grad=_GRAD[], avx=_AVX[], cuda=_CUDA[], tensor=_TENSOR[])
    end

    if opts.tensor && opts.redfun == :+ && isdefined(mod, :TensorOperations) && opts.grad != :Dual
        res = try_tensor(ex, ranges, DotDict(;mod = mod, opts...,
            arrays = Symbol[], indices = [], scalars = Symbol[]))
        if res != nothing # then forward & backward both handled by try_tensor
            return Expr(:block, res...) |> esc
        end
    end

    store = DotDict(; mod = mod, opts...,
        flags = Set{Symbol}(), # set while parsing input
    # Reduction
        redind = Symbol[],
    # Everything writes into leftarray[leftraw...], sometimes with a generated name
        leftraw = [],
        leftind = Symbol[],    # vcat(leftind, redind) is the complete list of loop indices
        leftarray = nothing,
        leftscalar = nothing, # only defined for scalar reduction
        leftnames = Symbol[],  # for NamedDims
    # Whole RHS, without finaliser, plus things extracted:
        right = nothing,
        finaliser = nothing,
        rightind = Symbol[],
        sharedind = Symbol[], # indices appearing on every RHS array, safe for ∇thread
        unsafeleft = Symbol[], # k in A[J[k]] never written to by different threads
        unsaferight = Symbol[], # same for gradient
        arrays = Symbol[],
        scalars = Symbol[],
        cost = 1,
    # Index ranges: first save all known constraints
        constraints = Dict{Symbol,Vector}(), # :k => [:(axes(A,2)), :(axes(B,1))] etc.
        notfree = Symbol[], # indices assigned values i = clamp(j, 1,3) within RHS
        shiftedind = Symbol[],
        pairconstraints = Tuple[], # (:i, :j, entangled range_i, range_j) from A[i+j] etc.
        axisdefs = Expr[],
    # Expressions:
        outpre = Expr[],  # preliminary steps
        outex = Expr[],   # the rest!
    )

    parse_input(ex, store)

    parse_ranges(ranges, store)

    index_ranges(store)

    output_array(store)

    ex = action_functions(store)

    opts.verbose == 2 && verboseprint(store)

    ex |> esc
end

#========== options, etc ==========#

OPTS = Dict(
    :verbose => Any[true, false, 2],
    :fastmath => [true, false],
    :threads => Integer,
    :grad => [false, :Base, :Dual],
    :avx => Integer,
    :cuda => Integer,
    :tensor => [true, false],
    )

_VERBOSE = Ref{Any}(false)
_FASTMATH = Ref(true)
_THREADS = Ref{Any}(true)
_GRAD = Ref{Any}(:Base)
_AVX = Ref{Any}(true)
_CUDA = Ref{Any}(256)
_TENSOR = Ref(true)

function parse_options(exs...)
    opts = Dict{Symbol,Any}(
        :redfun => :+,
        :verbose => _VERBOSE[],
        :fastmath => _FASTMATH[],
        :threads => _THREADS[],
        :grad => _GRAD[],
        :avx => _AVX[],
        :cuda => _CUDA[],
        :tensor => _TENSOR[],
        )
    expr = nothing
    nograd = Symbol[]
    ranges = Tuple[]
    for ex in exs
        # Actual options:
        if ex isa Expr && ex.head == :(=) && haskey(OPTS, ex.args[1])
            checklegal(ex.args[1], ex.args[2])
            opts[ex.args[1]] = ex.args[2]

        # Nograd keyword
        elseif ex isa Expr && ex.head == :(=) && ex.args[1] == :nograd
            if ex.args[2] isa Symbol
                push!(nograd, ex.args[2])
            elseif ex.args[2] isa Expr && ex.args[2].head == :tuple
                append!(nograd, ex.args[2].args)
            else
                error("this accepts nograd=A or nograd=(A,B,C)")
            end

        # Ranges specified outside:
        elseif ex isa Expr && ex.head == :call && ex.args[1] in [:in, :∈]
            push!(ranges, (ex.args[2], ex.args[3]))
        elseif ex isa Expr && ex.head == :tuple && ex.args[1] isa Expr && ex.args[1].args[1] in [:in, :∈]
            for el in ex.args
                el isa Expr && el.head == :call && el.args[1] in [:in, :∈] || error("expected (i ∈ 1:3) but got $el")
                push!(ranges, (el.args[2], el.args[3]))
            end

        # Reduction function
        elseif ex isa Symbol
            opts[:redfun] = ex

        # The main course!
        elseif ex isa Expr
            isnothing(expr) || error("too many expressions! recognised keywords are $(vcat(:nograd, keys(opts)...))")
            expr = ex
        else
            error("not sure what to do with input $ex")
        end
    end
    if isnothing(expr) # if run with no expression, it updates global options
        _VERBOSE[] = opts[:verbose]
        _FASTMATH[] = opts[:fastmath]
        _THREADS[] = opts[:threads]
        _GRAD[] = opts[:grad]
        _AVX[] = opts[:avx]
        _CUDA[] = opts[:cuda]
        _TENSOR[] = opts[:tensor]
    end
    (redfun=opts[:redfun],
        verbose=opts[:verbose],
        fastmath=opts[:fastmath],
        threads=opts[:threads],
        grad=opts[:grad],
        avx=opts[:avx],
        cuda=opts[:cuda],
        tensor=opts[:tensor],
        nograd=nograd,
    ), ranges, expr
end

checklegal(opt, val) =
    if OPTS[opt] isa Vector
        val in OPTS[opt] || error("keyword $opt accepts values [" * join(OPTS[opt], ", ") * "]")
    elseif val isa Expr || val isa Symbol
        # allows threads=64^3 to work
    elseif OPTS[opt] == Integer
        val isa Integer && val >= 0 || error("keyword $opt accepts false or a positive integer")
    end

verboseprint(store) =
    foreach(propertynames(store)) do k
        r = getproperty(store, k) # startswith(string(k), "out") fails?
        k ∉ [:outpre, :outex] && return printstyled("    $k = ", repr(r), "\n", color=:blue)
        printstyled("    $k =\n", color=:blue)
        foreach(ex -> printstyled(verbosetidy(ex) , "\n", color=:green), r)
    end

#========== symbols ==========#

# These only need not to clash with symbols in the input:
RHS, AXIS = :𝓇𝒽𝓈, :𝒶𝓍
ZED, TYP, ACC, KEEP, FINAL = :ℛ, :𝒯, :𝒜𝒸𝒸, :♻️, :💀
EPS, DEL, EXPR = :𝜀, :𝛥, :ℰ𝓍
MAKE, ACT! = :ℳ𝒶𝓀ℯ, :𝒜𝒸𝓉!

# @gensym RHS MAKE ACT!
# @gensym AXIS ZED TYP ACC KEEP FINAL
# @gensym EPS DEL EXPR

SYMBOLS = [
    RHS, MAKE, ACT!, AXIS, ZED, TYP, ACC, KEEP, EPS, DEL, EXPR,
    Symbol(:∇, MAKE), Symbol(:∇, ACT!), Symbol(DEL, ZED), Symbol(AXIS, :i),
    ] # to test for leaks

#========== input parsing ==========#

function parse_input(expr, store)

    # Equals sign & friends:
    if @capture_(expr, left_ := right_ )
        push!(store.flags, :newarray)
    elseif @capture_(expr, left_ = right_ )
    elseif @capture_(expr, left_ += right_ )
        push!(store.flags, :plusequals)
        store.redfun == :+ || error("can't use += with reduction $(store.redfun)")
    elseif @capture_(expr, left_ *= right_ )
        push!(store.flags, :plusequals) # slightly abusing the name of the flag!
        if store.redfun == :+ # default, then we change it?
            store.verbose>0 && @info "inferring reduction by *, because of lhs *= rhs"
            store.redfun = :*
        elseif store.redfun == :*
        else
            error("can't use *= with reduction $(store.redfun)")
        end
    else error("can't understand input, expected A[] := B[] (or with =, +=, or *=) got $expr")
    end

    # Left hand side:
    if @capture_(left, Z_[leftraw__] ) || @capture_(left, [leftraw__] )
    elseif left isa Symbol # complete reduction, by writing into a new 1-el array
        push!(store.flags, :newarray, :scalar)
        store.leftscalar = left # because store.leftarray will be the array
        leftraw = [1,] # make a 1D array, not zero
        expr.head == :(+=) && push!(store.scalars, left)
    else
        error("can't understand LHS, expected A[i,j,k], got $left")
    end
    leftraw2 = tidyleftraw(leftraw, store)
    store.leftind = filter(i -> i isa Symbol, leftraw2) # this gives correct outer loop order

    isnothing(Z) && !(:newarray in store.flags) && error("can't write into an array whose name isn't given!")
    Zed = isnothing(Z) ? ZED : Z
    store.leftarray = Zed

    store.leftraw = finishleftraw(leftraw2, store)
    if :newarray in store.flags
        !allunique(store.leftind) && push!(store.flags, :zero) # making diagonals, etc.
        Zed in store.arrays && error("can't create a new array $Zed when this also appears on the right")
    else
        saveconstraints(Zed, leftraw, store, false) # this adds to leftind, e.g. A[2i+1] = ..., is that bad??
        detectunsafe(left, store.unsafeleft, store)
    end

    # Right hand side
    detectunsafe(right, store.unsaferight, store)
    right2 = MacroTools_postwalk(rightwalk(store), right)

    if right2 isa Expr && right2.head == :call && right2.args[1] in (:|>, :<|)
        if right2.args[1] == :|>
            store.finaliser = makefinaliser(right2.args[3], store)
            store.right = MacroTools_postwalk(dollarwalk(store), right2.args[2])
        elseif right.args[1] == :<|
            store.finaliser = makefinaliser(right2.args[2], store)
            store.right = MacroTools_postwalk(dollarwalk(store), right2.args[3])
        end
        if :scalar in store.flags
            # scalar threaded reduction won't work with nontrivial finalisers
            store.threads = false
        end
    else
        store.right = MacroTools_postwalk(dollarwalk(store), right2)
        store.finaliser = :identity
    end

    unique!(store.scalars)
    unique!(store.arrays)
    unique!(store.leftind)
    store.sharedind = unique!(setdiff(store.sharedind, store.notfree))
    store.rightind = unique!(setdiff(store.rightind, store.notfree))
    unique!(store.outpre) # kill mutiple assertions, and evaluate any f(A) only once

end

rightwalk(store) = ex -> begin
        @nospecialize ex
        # First, this will detect any assignment before it is used:
        if ex isa Expr && ex.head == :(=)
            if ex.args[1] isa Symbol
                push!(store.notfree, ex.args[1])
            elseif ex.args[1] isa Expr && ex.args[1].head == :tuple
                for i in ex.args[1].args
                    i isa Symbol && push!(store.notfree, i)
                end
            end
        end
        ex isa Expr && ex.head == :return && error("can't use return inside body")

        # Second, alter indexing expr. to pull out functions of arrays:
        @capture_(ex, A_[inds__]) || return ex

        if isnothing(arrayonly(A))
            Anew = Symbol(string("≪", A, "≫"))
            push!(store.outpre, :(local $Anew = $A))
            A = Anew
        end

        # Third, save letter A, and what axes(A) says about indices:
        push!(store.arrays, arrayonly(A))
        inds = primeindices(inds)
        saveconstraints(A, inds, store, true)

        # Re-assemble RHS with new A, and primes on indices taken care of.
        return :( $A[$(inds...)] )
    end # A1[i][k] should be seen later, with corrected A

arrayonly(A::Symbol) = A   # this is for RHS(i,j,k, A,B,C)
arrayonly(A::Expr) =
    if @capture_(A, B_[inds__]) || @capture_(A, B_.field_)
        return arrayonly(B)
    end # returns nothing from :(f(A)), signal to pull function out.

saveconstraints(A, inds, store, right=true) = begin
    A1 = arrayfirst(A)
    is = Symbol[]
    foreach(enumerate(inds)) do (d,ex)
        is_const(ex) && return
        containsany(ex, store.notfree) && return
        axis_i = length(inds)==1 ? :(eachindex($A1)) : :(axes($A1,$d))
        range_i, i = range_expr_walk(axis_i, ex)
        if i isa Symbol
            push!(is, i)
            ex isa Symbol || push!(store.shiftedind, i)
            v = get!(store.constraints, i, [])
            push!(v, dollarstrip(range_i))
        elseif i isa Tuple # from things like A[i+j]
            push!(is, filter(!isnothing, collect(i))...) # collect for Julia ⩽ 1.3
            push!(store.shiftedind, filter(!isnothing, collect(i))...)
            push!(store.pairconstraints, (i..., dollarstrip.(range_i)...))
        elseif isnothing(i) # from A[J[k]], but A[J[k]+i] goes via store.pairconstraints
            str = "extrema of index $ex must fit within $A1"
            push!(store.outpre, :(issubset($range_i, $axis_i) || error($str)))
        end
    end
    if right
        append!(store.rightind, is)
        if A1 in store.nograd # then don't care whether it sharesindices
        elseif isassigned(store.sharedind)
            shared = intersect(is, store.sharedind)
            empty!(store.sharedind)
            append!(store.sharedind, shared)
        else
            append!(store.sharedind, is)
        end
    else
        append!(store.leftind, is) # why can's this be the only path for store.leftind??
    end
    n = length(inds)
    if n==1
        str = "expected a 1-array $A1, or a tuple"
        push!(store.outpre, :( $A1 isa Tuple || ndims($A1) == 1 || error($str) ))
    else
        str = "expected a $n-array $A1" # already arrayfirst(A)
        push!(store.outpre, :( ndims($A1) == $n || error($str) ))
    end
end

arrayfirst(A::Symbol) = A  # this is for axes(A,d), axes(first(B),d), etc.
arrayfirst(A::Expr) =
    if (@capture_(A, Binds_.field_) && @capture_(Binds, B_[inds__]))
        return :( first($B).$field )
    elseif @capture_(A, B_[inds__])
        return :( first($B) )
    elseif @capture_(A, B_.field_)
        return A
    end

containsany(ex, list) = begin
    out = false
    MacroTools_postwalk(ex) do x
        if x in list
            out = true
        end
        x
    end
    out
end

primeindices(inds) = map(inds) do ex
    ex isa Expr && ex.head == Symbol("'") &&
        return Symbol(ex.args[1], "′") # normalise i''
    ex
end

dollarwalk(store) = ex -> begin
        @nospecialize ex
        ex isa Expr || return ex
        if ex.head == :call
            callcost(ex.args[1], store) # cost model for threading
        elseif ex.head == :$ # interpolation of $c things:
            ex.args[1] isa Symbol || error("you can only interpolate single symbols, not $ex")
            push!(store.scalars, ex.args[1])
            return ex.args[1]
        end
        ex
    end

dollarstrip(expr) = MacroTools_postwalk(expr) do @nospecialize ex
        ex isa Expr && ex.head == :$ && return ex.args[1]
        ex
    end

tidyleftraw(leftraw, store) = begin
    step1 = map(leftraw) do i
        if i isa Expr && i.head == :kw && :newarray in store.flags # then NamedDims wrapper is put on later
                push!(store.leftnames, i.args[1])
                return i.args[2]
        elseif i === :_ # underscores on left denote trivial dimensions
            return 1
        end
        i
    end
    primeindices(step1) # normalise i' to i′
end

finishleftraw(leftraw, store) = map(enumerate(leftraw)) do (d,i)
    if i isa Expr && i.head == :$
        :newarray in store.flags && error("can't fix indices on LHS when making a new array")
        i.args[1] isa Symbol || error("you can only interpolate single symbols, not $ex")
        push!(store.scalars, i.args[1])
        return i.args[1]

    elseif @capture_(i, J_[inds__]) # scatter operation, A[i,J[j,k]] := ...
        push!(store.nograd, J)
        rightwalk(store)(i) # array J viewed as part of RHS, and provides a range for j,k
        inds2 = filter(j->j isa Symbol, tidyleftraw(inds, store))
        append!(store.leftind, inds2) # but j,k aren't to be summed

        ex = :($J[$(tidyleftraw(inds, store)...)])
        if :newarray in store.flags
            ax_i = Symbol(AXIS, string("≪", ex, "≫")) # fake index name, to which to attach a size?
            push!(store.axisdefs, :(local $ax_i = $extremerange($J)))
            push!(store.flags, :zero)
        elseif !(:plusequals in store.flags) # A[i,J[j,k]] += ... doesn't zero
            push!(store.flags, :zero)
        end

        return ex # has primes dealt with
    end
    i
end

detectunsafe(expr, list, store) = MacroTools_postwalk(expr) do ex
        @capture_(ex, A_[inds__]) || return ex
        for i in inds
            MacroTools_postwalk(i) do x
                @capture_(x, B_[inner__]) || return x
                # Now we have found an array which indexes another one, mark its indices unsafe
                append!(list, filter(j -> j isa Symbol, inner))
                unique!(list)
                # and don't compute a gradient for the inner array
                B isa Symbol && push!(store.nograd, B)
                x
            end
        end
        ex
    end

makefinaliser(s::Symbol, store) = s
makefinaliser(expr::Expr, store) = begin
    underscore = false
    out = MacroTools_postwalk(expr) do ex
        if ex == :_
            underscore = true
            return RHS
        elseif @capture_(ex, A_[inds__])
            for i in inds
                i isa Symbol || continue
                i in store.leftind || error("index $i can't be used in finaliser")
            end
        end
        ex
    end
    if underscore
        return dollarstrip(:($RHS -> $out))
    else
        return dollarstrip(ex)
    end
end

function parse_ranges(ranges, store) # now runs after parse_input
    for (i,r) in ranges
        if i isa Expr && i.head == Symbol("'") # catch primes!
            i = Symbol(i.args[1], "′")
        end
        push!(store.rightind, i)
        v = get!(store.constraints, i, [])
        if r isa Expr && r.head == :call && r.args[1] == :(:) && length(r.args) == 3
            # for a literal range, write OneTo(10) or 0:9 directly into constraints
            if r.args[2] == 1 && r.args[3] isa Integer
                push!(v, :(Base.OneTo($(r.args[3]))))
                continue
            elseif r.args[2] isa Integer && r.args[3] isa Integer
                push!(v, r)
                continue
            end
        end
        # for axes(A,2) where A is already available, just save it
        if r isa Expr && r.head == :call && r.args[1] in (:axes, :eachindex) && r.args[2] in store.arrays
            push!(v, r)
            continue
        end
        # for anything else, treat it as a scalar argument
        if r isa Symbol
            push!(store.scalars, r)
            push!(v, r)
        else
            s = Symbol(string("≪", r, "≫"))
            push!(store.outpre, :($s = $r))
            push!(store.scalars, s)
            push!(v, s)
        end
    end
    unique!(store.rightind)
    unique!(store.scalars)
    store.redind = setdiff(store.rightind, store.leftind)
end

#========== index ranges ==========#

function index_ranges(store)

    todo = Set(vcat(store.leftind, store.redind))
    done = Dict{Symbol,Any}()

    for (i,j,r_i,r_j) in store.pairconstraints

        if isnothing(i) # case of A[j + I[k]]
            v = get!(store.constraints, j, [])
            push!(v, r_j)
        elseif isnothing(j)
            v = get!(store.constraints, i, [])
            push!(v, r_i)

        elseif haskey(store.constraints, i) && i in todo
            resolveintersect(i, store, done) # use existing knowledge to fix i's range
            pop!(todo, i)
            v = get!(store.constraints, j, []) # and then allow j's range to depend on that
            push!(v, r_j)
        elseif haskey(store.constraints, j) && j in todo
            resolveintersect(j, store, done)
            pop!(todo, j)
            v = get!(store.constraints, i, [])
            push!(v, r_i)
        end
    end

    for i in todo
        haskey(store.constraints, i) || error("unable to infer range of index $i")
        if i in store.shiftedind
            resolveintersect(i, store, done)
        else
            resolvestrict(i, store, done)
        end
    end

    append!(store.outex, store.axisdefs)

    if store.verbose > 0
        lex = map(i -> Expr(:(=), i, done[i]), store.leftind)
        push!(store.outex, :(@info "left index ranges" $(lex...)))
        if !isempty(store.redind)
            rex = map(i -> Expr(:(=), i, done[i]), store.redind)
            push!(store.outex, :(@info "reduction index ranges" $(rex...)))
        end
    end
end

resolvestrict(i, store, done) = begin
    res = first(store.constraints[i])
    ax_i = Symbol(AXIS, i)
    push!(store.axisdefs, :( local $ax_i = $res ))
    done[i] = res
    for alt in store.constraints[i][2:end] # in which case it shouldn't be a Set
        str = "range of index $i must agree"
        push!(store.axisdefs, :( $alt == $res || error($str) ))
    end
end

resolveintersect(i, store, done) = begin
    res = length(store.constraints[i])==1 ?
        first(store.constraints[i]) : # because intersect(1:3) isa Vector, wtf?
        :( intersect($(store.constraints[i]...)) )
    ax_i = Symbol(AXIS, i)
    push!(store.axisdefs, :( local $ax_i = $res ))
    done[i] = res
end

#========== output array + eltype ==========#

function output_array(store)
    if :newarray in store.flags

        push!(store.outex, :( local $RHS($(store.arrays...), $(store.rightind...)) = $(store.finaliser)($(store.right)) ))

        # Try inference first, usually fine, and avoids scalar evaluation on GPU
        allfirst = map(i -> :(first($(Symbol(AXIS, i)))), store.rightind)
        T0 = Symbol(TYP,0)
        warn = store.verbose>0 ? :(@warn "unable to infer eltype from RHS") : nothing
        push!(store.outex, quote
            local $T0 = Core.Compiler.return_type($RHS, typeof(($(store.arrays...), $(allfirst...))))
            local $TYP = if Base.isconcretetype($T0)
                $T0
            else
                $warn
                typeof($RHS($(store.arrays...), $(allfirst...)))
            end
        end)

        # This now checks for OffsetArrays, and allows A[i,1] := ...
        outaxes = map(store.leftraw) do i
            i isa Integer && i==1 && return :(Base.OneTo(1))
            i isa Symbol && return Symbol(AXIS, i)
            i isa Expr && @capture_(i, J_[inds__]) && return Symbol(AXIS, string("≪", i, "≫"))
            error("can't use index $i on LHS for a new array")
        end

        if !isdefined(store.mod, :OffsetArrays)
            outaxes = map(store.leftraw, outaxes) do i, ax
                ax == :(Base.OneTo(1)) && return ax
                i in store.shiftedind || @capture_(i, J_[inds__]) || return ax
                push!(store.outex, :( first($ax) == 1 || error("to allow indices not starting at 1, OffsetArrays must be visible in the caller's module")))
                return :(Base.OneTo($ax))
            end
        end

        simex = if isempty(store.arrays)
            # :( zeros($TYP, tuple($(outaxes...))) ) # Array{T} doesn't accept ranges... but zero() doesn't accept things like  @tullio [i,j] := (i,j)  i ∈ 2:3, j ∈ 4:5
            :( similar(1:0, $TYP, tuple($(outaxes...))) )
        else
            # parent() is a trick to avoid a NamedDims bug
            :( similar(parent($(store.arrays[1])), $TYP, tuple($(outaxes...),)) )
        end
        if isempty(store.leftnames)
            push!(store.outex, :( local $(store.leftarray) = $simex ))
        else
            nex = :(tuple($(QuoteNode.(store.leftnames)...)))
            push!(store.outex, :( local $(store.leftarray) = NamedDims.NamedDimsArray($simex, $nex) ))
        end

        # Deal with scalar += now: write into array, later read it out:
        if :scalar in store.flags && :plusequals in store.flags
            push!(store.outex, :($(store.leftarray)[$(store.leftraw...)] = $(store.leftscalar)))
        end
    end

    if :zero in store.flags
        push!(store.outex, :( $(store.leftarray) .= false )) # zero($TYP) won't work in-place
    end

end

#========== action functions ==========#

function action_functions(store)

    axisleft = map(i -> Symbol(AXIS, i), setdiff(store.leftind, store.unsafeleft))
    axisred = map(i -> Symbol(AXIS, i), union(store.redind, store.unsafeleft))
    axislist = vcat(axisleft, axisred)
    # Order of these is convenient for threader(), which divides axisleft up freely,
    # divides axisred up with re-starts.
    # This is independent of the grouping inner/outer for make_many_actors().

    #===== constructing loops =====#

    init = store.redfun == :* ? :(one($TYP)) :
        store.redfun == :max ? :(typemin($TYP)) :
        store.redfun == :min ? :(typemax($TYP)) :
        store.redfun == :& ? :(true) :
        store.redfun == :| ? :(false) :
        :(zero($TYP))

    # Right now this would allow *= only with reduction * too. Could separate them:
    # acc=0; acc = acc + rhs; Z[i] = ifelse(keep, acc, Z[i] * acc)
    # But then keep=true can't be used for blocking, which wants to continue the same as acc.

    ex_init = :( $ACC = ifelse(isnothing($KEEP), $init, $ZED[$(store.leftraw...)]) )
    # ex_init = :( $ACC = isnothing($KEEP) ? $init : $ZED[$(store.leftraw...)] ) # more allocations with @avx, not sure why

    ex_iter = :( $ACC = $(store.redfun)($ACC, $(store.right) ) )

    ex_write = if store.finaliser == :identity
        :( $ZED[$(store.leftraw...)] = $ACC )
    else
        # :( $ZED[$(store.leftraw...)] = isnothing($FINAL) ? $ACC : $(store.finaliser)($ACC) )
        :( $ZED[$(store.leftraw...)] = ifelse(isnothing($FINAL), $ACC, $(store.finaliser)($ACC)) )
    end

    ex_nored = if :plusequals in store.flags # meaning always keep = true and final = true, since there's no branch, no need to reduce :identity
        :( $ZED[$(store.leftraw...)] =  $(store.finaliser)($(store.redfun)($ZED[$(store.leftraw...)] ,$(store.right))) )
    else
        :( $ZED[$(store.leftraw...)] =  $(store.finaliser)($(store.right)) )
    end

    if isempty(store.redind)
        make_many_actors(ACT!,
            vcat(:($ZED::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
            nothing, store.leftind, nothing, Symbol[], ex_nored, nothing, store)
    else
        make_many_actors(ACT!,
            vcat(:($ZED::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
            nothing, store.leftind, ex_init, store.redind, ex_iter, ex_write, store)
    end

    ∇make = if :newarray in store.flags
        # make_many_actors and backward_definitions both push into store.outpre
        backward_definitions(store)
    else
        nothing
    end

    #===== action! =====#

    ST = :($storage_type($(store.leftarray), $(store.arrays...)))
    keep = (:plusequals in store.flags) ? :true : :nothing
    block = store.threads==false ? nothing :
        store.threads==true ? (BLOCK[] ÷ store.cost) :
        store.threads
    push!(store.outex, quote
        $threader($ACT!, $ST, $(store.leftarray),
            tuple($(store.arrays...), $(store.scalars...),),
            tuple($(axisleft...),), tuple($(axisred...),), $(store.redfun), $block, $keep)
        $(store.leftarray)
    end)
    store.verbose>0 && block != nothing && @info "threading threshold (from cost = $(store.cost))" block

    if :newarray in store.flags
        # then slurp up outex to make a function:
        ex = quote
            let $ACT! = $ACT!
                local @inline function $MAKE($(store.arrays...), $(store.scalars...), )
                    $(store.outex...)
                end
                $Eval($MAKE, $∇make)($(store.arrays...), $(store.scalars...), )
            end
        end

        # wrap pre and out in one let block so that ACT! doesn't escape:
        ex = :(let
            $(store.outpre...)
            $ex
        end)

        # and assign the result if necc:
        if store.leftarray != ZED
            push!(store.outex, :($(store.leftarray) = $ex ))
            return :($(store.leftarray) = $ex )
        elseif :scalar in store.flags
             push!(store.outex, :($(store.leftscalar) = $getonly($ex)))
             return :($(store.leftscalar) = $getonly($ex))
        else # case of [i,j] := ... with no name given
            # push!(store.outex, ex)
            return ex
        end

    else
        # in-place, no MAKE function, but still keep ACT! from escaping
        return :(let
            $(store.outpre...)
            $(store.outex...)
        end)
    end
end


"""
    make_many_actors(f!, args, ex1, [:i,], ex3, [:k,], ex5, ex6, store)

This makes several functions of this form,
decorated as necessary with `@inbouds` or `@avx` etc,
and with appropriate `storage_type` as the first argument.
```
f!(::Type, args..., keep=nothing) where {T}
    ex1
    ex2 = (for i in axis_i
        ex3
        ex4 = (for k in axis_k
            ex5
        end)
        ex6
    end)
end
```
"""
function make_many_actors(act!, args, ex1, outer::Vector, ex3, inner::Vector, ex5, ex6, store, note="")

    ex4 = recurseloops(ex5, inner)
    ex2 = recurseloops(:($ex3; $ex4; $ex6), outer)

    if store.fastmath && isempty(store.notfree)
        push!(store.outpre, quote
            local @inline function $act!(::Type, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                @inbounds @fastmath ($ex1; $ex2)
            end
        end)
    elseif isempty(store.notfree)
        push!(store.outpre, quote
            local @inline function $act!(::Type, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                @inbounds ($ex1; $ex2)
            end
        end)
    else
        push!(store.outpre, quote
            local @inline function $act!(::Type, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                ($ex1; $ex2)
            end
        end)
    end

    expre, exloop, expost = if isempty(outer)
        :($ex1; $ex3), ex4, ex6
    else
        ex1, ex2, nothing
    end

    # Disable @avx for scatter, https://github.com/chriselrod/LoopVectorization.jl/issues/145
    safe = if act! == ACT!
        isempty(store.unsafeleft)
    else # working on ∇act!
        isempty(store.unsaferight)
    end

    if safe && store.avx != false && isdefined(store.mod, :LoopVectorization)
        unroll = store.avx == true ? 0 : store.avx # unroll=0 is the default setting
        info1 = store.verbose>0 ? :(@info "running LoopVectorization actor $($note)") : nothing
        try lex = macroexpand(store.mod, quote

                local @inline function $act!(::Type{<:Array{<:Union{Base.HWReal, Bool}}}, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                    $expre
                    $info1
                    LoopVectorization.@avx unroll=$unroll $exloop
                    $expost
                end

            end) # macroexpand quote
            push!(store.outpre, lex)
            store.verbose == 2 && @info "success wtih LoopVectorization, unroll=$unroll $note"
        catch err
            store.verbose > 0 && @warn "LoopVectorization failed $note" err
        end
    end

    axouter = map(i -> Symbol(AXIS, i), outer)

    if store.cuda > 0 && isdefined(store.mod, :KernelAbstractions)

        kernel = gensym(:🇨🇺)
        asserts = map(ax -> :( first($ax)==1 || error("KernelAbstractions can't handle OffsetArrays here")), axouter)
        sizes = map(ax -> :(length($ax)), axouter)
        try
            if isempty(outer)
                store.verbose > 0 && @warn "using KernelAbstractions with no outer indices, this will be slow"
                outer = [Symbol(EPS, 1)] # fake index name, only appears in @index
                sizes = [:(one(Int))]    # iterate over 1:1
            end
            kex1 = macroexpand(store.mod, quote

                KernelAbstractions.@kernel function $kernel($(args...), $KEEP, $FINAL) where {$TYP}
                    ($(outer...),) = @index(Global, NTuple)
                    ($ex1; $ex3; $ex4; $ex6)
                end

            end)
            push!(store.outpre, kex1)
            if isdefined(store.mod, :CuArrays) && isdefined(store.mod, :CuArray) # old-style, CuArrays.jl
                info2 = store.verbose>0 ? :(@info "running KernelAbstractions + CuArrays actor $($note)") : nothing
                kex2 = quote

                    local @inline function $act!(::Type{<:CuArray}, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                        $info2
                        cu_kern! = $kernel(CUDA(), $(store.cuda))
                        $(asserts...)
                        $ACC = cu_kern!($(args...), $KEEP, $FINAL; ndrange=tuple($(sizes...)))
                        KernelAbstractions.wait($ACC)
                    end

                end
                push!(store.outpre, kex2)
            end
            if isdefined(store.mod, :CUDA) && isdefined(store.mod, :CuArray) # new-style, CUDA.jl, with CUDADevice()
                info2bis = store.verbose>0 ? :(@info "running KernelAbstractions + CUDA actor $($note)") : nothing
                kex2bis = quote

                    local @inline function $act!(::Type{<:CuArray}, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                        $info2bis
                        cu_kern! = $kernel(CUDADevice(), $(store.cuda))
                        $(asserts...)
                        $ACC = cu_kern!($(args...), $KEEP, $FINAL; ndrange=tuple($(sizes...)))
                        KernelAbstractions.wait($ACC)
                    end

                end
                push!(store.outpre, kex2bis)
            end
            info3 = store.verbose>0 ? :(@info "running KernelAbstractions CPU actor $($note)") : nothing
            kex3 = quote

                local @inline function $act!(::Type{<:Array}, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                    $info3
                    cpu_kern! = $kernel(CPU(), 4)
                    $(asserts...)
                    $ACC = cpu_kern!($(args...), $KEEP, $FINAL; ndrange=tuple($(sizes...)))
                    KernelAbstractions.wait($ACC)
                end

            end
            if store.threads==false
                # This CPU kernel can't be called by threader, and so threads=false
                # offers a way to control whether it gets used or not. By default, not.
                push!(store.outpre, kex3)
            end
            store.verbose == 2 && @info "success wtih KernelAbstractions $note"
        catch err
            store.verbose > 0 && @warn "KernelAbstractions failed $note" err
        end
    end

    if act! != ACT! && isempty(store.sharedind) && Threads.nthreads()>1
        store.verbose > 0 && @warn "can't parallelise this gradient, no shared indices $note"
    end
end


recurseloops(ex, list::Vector) =
    if isempty(list)
        return ex
    else
        i = first(list)
        r = Symbol(AXIS, i)
        ex = :(for $i in $r; $ex; end)
        return recurseloops(ex, list[2:end])
    end

#===== define gradient hooks =====#

function backward_definitions(store)
    store.grad == false && return nothing # no gradient wanted

    axisshared = map(i -> Symbol(AXIS, i), setdiff(store.sharedind, store.unsaferight)) # safe to multi-thread
    loopind = vcat(store.leftind, store.redind)
    axisnonshared = map(i -> Symbol(AXIS, i), union(setdiff(loopind, store.sharedind), store.unsaferight))

    axislist = vcat(axisshared, axisnonshared) # this defines the order of arguments of ∇act!

    ok = false
    if store.grad == :Dual && store.redfun == :+
        try
            insert_forward_gradient(axislist, store)
            ok = true
            store.verbose == 2 && @info "using ForwardDiff gradient"
        catch err
            store.verbose > 0 && @warn "ForwardDiff gradient failed" err
        end
    elseif store.grad == :Base
        try
            insert_symbolic_gradient(axislist, store)
            ok = true
            store.verbose == 2 && @info "success wtih Symbolic gradient"
        catch err
            store.verbose > 0 && @warn "Symbolic gradient failed" err
        end
    end

    ok == false && return nothing # failed to make a gradient

    dZ = Symbol(DEL, ZED)
    ∇make = Symbol(:∇, MAKE)
    ∇act! = Symbol(:∇, ACT!)

    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    # gradscalars = map(A -> Symbol(DEL, A), store.scalars)
    defineempties = map(store.arrays, gradarrays) do A, dA
        if A in store.nograd
            :(local $dA = nothing)
        else
            :( local $dA = fill!(similar($A, Base.promote_type(eltype($A), $TYP)), 0) )
        end
    end
    # append!(defineempties, map((x,dx) -> :($dx = zero(Base.promote_type(typeof($x), $TYP))), store.scalars, gradscalars))
    returns = vcat(gradarrays, map(_->:nothing, store.scalars)) # ?? needs a test!
    # returns = vcat(gradarrays, gradscalars)

    ST = :($storage_type($(gradarrays...), $(store.arrays...)))
    block = store.threads==false ? nothing :
        store.threads==true ? (BLOCK[] ÷ store.cost) :
        store.threads
    push!(store.outpre, quote

        local $∇make = let $∇act! = $∇act!
            local function $∇make($dZ::AbstractArray{$TYP}, $ZED, $(store.arrays...), $(store.scalars...), ) where {$TYP}
                $(defineempties...)
                $(store.axisdefs...)
                $∇threader($∇act!, $ST,
                    tuple($(gradarrays...), $dZ, $ZED, $(store.arrays...), $(store.scalars...),),
                    tuple($(axisshared...),), tuple($(axisnonshared...), ), $block)
                return ($(returns...),)
            end
        end

    end)

    return ∇make
end

fillarrayreplace(rhs, dZ) = MacroTools_postwalk(rhs) do @nospecialize ex
        @capture_(ex, A_[inds__]) && A==dZ || return ex
        return Symbol(dZ, :_value)
    end

#========== the end ==========#
