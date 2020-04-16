
mutable struct Store store::NamedTuple end
Base.parent(x::Store) = getfield(x, :store)
Base.getproperty(x::Store, y::Symbol) = getproperty(parent(x), y)
Base.merge(x::NamedTuple, y::Store) = Store(merge(x, parent(y)))

#========== the macro! ==========#

const ExprSym = Union{Expr, Symbol}

"""
    @tullio C[i,k] := A[i,j] * B[j,k]
    @tullio C[i,k] := A[i].field[j] * B[row=j, col=k]

This is a replacement for `@einsum` which understands a bit more syntax.
`:=` makes a new array, `=` and `+=` write into an existing one.

    @tullio  avx=false  threads=false  C[i,k] = A[i,j] * B[j,k]

By default it uses LoopVectorization.jl when it can, and `Threads.@spawn` for big enough arrays.
These options disables both. Option `avx=4` will instead use `@avx unroll=4 for i in ...` loops.

    @tullio  grad=false  C[i,k] := ...

If Zygote.jl/Tracker.jl/ReverseDiff.jl are loaded, then it will define gradient hooks for these,
unless disabled by `grad=false`. The gradient itself is calculated in one of two ways,
either by symbolic differentiation of the RHS (the default, `grad=Base`)
or by using dual numbers from ForwardDiff.jl (option `grad=Dual`).

    @tullio  verbose=true

This prints out everythinng the macro knows & generates. (You can't always use `@macroexpand1`
as the gradients need things `eval`uated at top level.)
Options given without an expression change the global defaults, instead of applying just once.
"""
macro tullio(exs...)
    _tullio(exs...; mod=__module__)
end

function _tullio(exs...; mod=Main)

    verbose, threads, grad, avx, cuda, ex = parse_options(exs...)
    isnothing(ex) && return

    store = Store((mod = mod,
        verbose = verbose,
        threads = threads,
        grad = grad,
        avx = avx,
        cuda = cuda,
        flags = Set{Symbol}(), # set while parsing input
    # Reduction
        upop = Ref{Symbol}(:(=)), # allow *=  for @einsum compat, not yet done
        redfun = Ref{Symbol}(:+), # reduce by * not + etc, also not done
        redind = Symbol[],
    # Everything writes into leftarray[leftraw...], sometimes with a generated name.
        leftraw = Any[],
        leftind = Symbol[], # vcat(leftind, redind) is the complete list of loop indices
        leftarray = Ref{ExprSym}(),
        leftscalar = Ref{Symbol}(), # only defined for scalar reduction
        leftnames = Symbol[], # for NamedDims
    # Whole RHS, untouched
        right = Ref{Expr}(),
        rightind = Symbol[],
        sharedind = Array{Symbol}(undef, 0), # indices appearing on every RHS array
        arrays = Symbol[],
        scalars = Symbol[],
        cost = Ref{Int}(1),
    # Index ranges: first save all known constraints
        constraints = Dict{Symbol,Vector}(), # :k => [:(axis(A,2)), :(axis(B,1))] etc.
        pairconstraints = Tuple[], # (:i, :j, entangled range_i, range_j) from A[i+j] etc.
        axisdefs = Expr[],
    # Version of right with (A[i,j] + 𝜀A′) etc, with dict[:𝜀A′] = :(A[i,j])
        epsilonright = Ref{ExprSym}(),
        epsilondict = Dict{Symbol,Expr}(),
    # Expressions: outex is the main one, sometimes wrapped innto functions.
        outpre = ExprSym[], # things never to be inside function
        outeval = ExprSym[], # things already @eval-ed at top level for gradient.
        outex = ExprSym[],
    ))

    parse_input(ex, (:+), store)

    index_ranges(store)

    output_array(store)

    action_functions(store)

    verbose && verboseprint(store)

    Expr(:block, store.outpre..., store.outex...) |> esc
end

#========== options ==========#

OPTS = Dict(
    :verbose => [true, false],
    :threads => [true, false],
    :grad => [false, :Base, :Dual],
    :avx => vcat(false, Vector{Any}(1:16)),
    :cuda => 0:2048,
    )

VERBOSE = Ref(false)
THREADS = Ref(true)
GRAD = Ref{Any}(:Base)
AVX = Ref{Any}(true)
CUDA = Ref{Any}(256)

parse_options(exs...) = begin
    opts = Dict(:expr => nothing,
        :verbose => VERBOSE[],
        :threads => THREADS[],
        :grad => GRAD[],
        :avx => AVX[],
        :cuda => CUDA[],
        )
    for ex in exs
        if ex isa Expr && ex.head == :(=) && haskey(OPTS, ex.args[1])
            ex.args[2] in OPTS[ex.args[1]] || error(string(
            "keyword $(ex.args[1]) accepts values [", join(OPTS[ex.args[1]], ", "), "]"))
            opts[ex.args[1]] = ex.args[2]
        elseif ex isa Expr
            opts[:expr] = ex
        else
            error("not sure what to do with input $ex")
        end
    end
    if isnothing(opts[:expr]) # if run with no expression, it updates global options
        VERBOSE[] = opts[:verbose]
        THREADS[] = opts[:threads]
        GRAD[] = opts[:grad]
        AVX[] = opts[:avx]
    end
    opts[:verbose], opts[:threads], opts[:grad], opts[:avx], opts[:cuda], opts[:expr]
end

verboseprint(store) = begin
    foreach(keys(parent(store))) do k
        r = getproperty(store, k) # startswith(string(k), "out") fails?
        k ∉ [:outpre, :outeval, :outex] && return printstyled("    $k = ", repr(r), "\n", color=:blue)
        printstyled("    $k =\n", color=:blue)
        foreach(ex -> printstyled(MacroTools.prettify(ex) , "\n", color=:green), r)
    end
end

#========== symbols ==========#

# these just need not to clash with input
# ᵗᵘˡˡⁱᵒ 𝒵ᵉᵈ, 𝒵ℰ𝒟, 𝜀ᶠʷᵈ, :𝛥ᵇᵏ, :ᵃˣⁱˢ📏, :𝒜ᶜᶜ, :🖐ˢⁱᵈᵉ, :𝒯ʸᵖᵉ

RHS, AXIS = :🖐, :📏
ZED, TYP, ACC = :ℛℰ𝒮, :𝒯, :𝒜
EPS, DEL = :𝜀, :𝛥

#========== input parsing ==========#

function parse_input(ex1, ex2, store)
    ex = @capture(ex1, left_ += right_ ) ? :($left = $left + $right) :
        ex1
    if !isnothing(ex2)
        ex2 isa Symbol ? (store.redfun[] = ex2) : error("can't understand $ex2 yet")
    end

    newarray = @capture(ex, left_ := right_ )
    newarray || @capture(ex, left_ = right_ ) ||
        error("expected A[] := B[] or A[] = B[], got $ex")
    newarray && push!(store.flags, :newarray)

    if @capture_(left, Z_[leftraw__] ) || @capture(left, [leftraw__] )
    elseif left isa Symbol
        store.leftscalar[] = left
        leftraw = []
    else
        error("can't understand LHS, expected A[i,j,k], got $left")
    end
    append!(store.leftraw, tidyleftraw(leftraw, store))

    append!(store.leftind, reverse(filter(i -> i isa Symbol, store.leftraw))) # outer loop order
    !allunique(store.leftind) && newarray && push!(store.flags, :zero)

    Zed = isnothing(Z) ? ZED : Z
    store.leftarray[] = Zed

    newarray || saveconstraints(Zed, leftraw, store, false)
    unique!(store.leftind)

    right1 = MacroTools_postwalk(rightwalk(store), right)
    store.right[] = MacroTools_postwalk(dollarwalk(store), right1)
    unique!(store.scalars)

    unique!(store.arrays)
    unique!(store.sharedind)
    unique!(store.rightind)
    append!(store.redind, setdiff(store.rightind, store.leftind)) # seemingly random order??

    unique!(store.outpre) # kill mutiple @assert, also some limited CSE if f(A) appears twice

    newarray && Zed in store.arrays && error("can't create a new array $Zed when this also appears on the right")
end

rightwalk(store) = ex -> begin
        # First, note if these are seen:
        if @capture(ex, A_[inds__].field_) || @capture(ex, A_[inds__][more__])
            push!(store.flags, :noavx)
            push!(store.flags, :nograd)
        end
        ex isa Expr && ex.head == :kw && push!(store.flags, :noavx)
        ex isa Expr && ex.head == :call && ex.args[1] in [:(==)] && push!(store.flags, :noavx)
        ex isa Expr && ex.head == Symbol(".") && push!(store.flags, :noavx, :nograd)
        ex isa Symbol && startswith(string(ex), ".") && push!(store.flags, :noavx, :nograd)

        # Second, alter indexing expr. to pull out functions of arrays:
        @capture_(ex, A_[inds__]) || return ex

        if isnothing(arrayonly(A))
            Anew = Symbol(string("≪", A, "≫"))
            push!(store.outpre, :($Anew = $A))
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
        isconst(ex) && return
        ex isa Symbol || push!(store.flags, :intersect) # ?? might not be right
        range_i, i = range_expr_walk(:(axes($A1,$d)), ex)
        if i isa Symbol
            push!(is, i)
            v = get!(store.constraints, i, Expr[])
            isnothing(range_i) || push!(v, range_i) # ?? is this ever nothing?
        elseif i isa Tuple # from things like A[i+j]
            push!(is, i...)
            push!(store.pairconstraints, (i..., range_i...))
        end
    end
    if right
        append!(store.rightind, is)
        if isassigned(store.sharedind)
            shared = intersect(is, store.sharedind) # ?? is this right for multiple indices?
            empty!(store.sharedind)
            append!(store.sharedind, shared)
        else
            append!(store.sharedind, is)
        end
    else
        append!(store.leftind, is)
    end
    n = length(inds)
    str = "expected a $n-array $A1" # already arrayfirst(A)
    push!(store.outpre, :( @assert ndims($A1) == $n $str ))
end

arrayfirst(A::Symbol) = A  # this is for axes(A,d), axes(first(B),d), etc.
arrayfirst(A::Expr) =
    if @capture(A, B_[inds__].field_)
        return :( first($B).$field )
    elseif @capture_(A, B_[inds__])
        return :( first($B) )
    elseif @capture_(A, B_.field_)
        return A
    end

primeindices(inds) = map(inds) do ex
    ex isa Expr && ex.head == Symbol("'") &&
        return Symbol(primeindices(ex.args[1]), "′") # normalise i''
    ex
end

dollarwalk(store) = ex -> begin
        ex isa Expr || return ex
        if ex.head == :call
            ex.args[1] == :* && ex.args[2] === Int(0) && return false # tidy up dummy arrays!
            callcost(ex.args[1], store) # cost model for threading
        elseif ex.head == :$ # interpolation of $c things:
            ex.args[1] isa Symbol || error("you can only interpolate single symbols, not $ex")
            push!(store.scalars, ex.args[1])
            return ex.args[1]
        end
        ex
    end

tidyleftraw(leftraw, store) = map(leftraw) do i
    if i isa Expr && i.head == :kw
        if :newarray in store.flags
            push!(store.leftnames, i.args[1])
            return i.args[2]
        else
            push!(store.flags, :noavx)
        end
    end
    i
end

#========== index ranges ==========#

function index_ranges(store)

    todo = Set(vcat(store.leftind, store.redind))

    for (i,j,r_i,r_j) in store.pairconstraints
        if haskey(store.constraints, i) # && i in todo ??
            resolveintersect(i, store) # use existing knowledge to fix i's range
            pop!(todo, i)
            v = get!(store.constraints, j, Expr[]) # and then allow j's range to depend on that
            push!(v, r_j)
        elseif haskey(store.constraints, j) # && j in todo
            resolveintersect(j, store)
            pop!(todo, j)
            v = get!(store.constraints, i, Expr[])
            push!(v, r_i)
        end
    end

    for i in todo
        haskey(store.constraints, i) || error("unable to infer range of index $i")
        # if i in store.sloppyindices # ?? maybe later
        if :intersect in store.flags
            resolveintersect(i, store)
        else
            resolvestrict(i, store)
        end
    end

    append!(store.outex, store.axisdefs)
end

resolvestrict(i, store) = begin
    res = first(store.constraints[i])
    r_i = Symbol(AXIS, i)
    push!(store.axisdefs, :( local $r_i = $res ))
    for alt in store.constraints[i][2:end] # in which case it shouldn't be a Set
        str = "range of index $i must agree"
        push!(store.axisdefs, :( @assert $alt == $res $str ))
    end
end

resolveintersect(i, store) = begin
    res = length(store.constraints[i])==1 ?
        first(store.constraints[i]) : # because intersect(1:3) isa Vector, wtf?
        :( intersect($(store.constraints[i]...)) )
    r_i = Symbol(AXIS, i)
    push!(store.axisdefs, :( local $r_i = $res ))
end

#========== output array + eltype ==========#

function output_array(store)
    if :newarray in store.flags

        funwithargs = :( $RHS($(store.arrays...), $(store.rightind...)) )
        push!(store.outex, :( $funwithargs = $(store.right[]) ))

        # Try inference first, usually fine, and avoids scalar evaluation on GPU
        allfirst = map(i -> :(first($(Symbol(AXIS, i)))), store.rightind)
        T0 = Symbol(TYP,0)
        push!(store.outex, quote
            $T0 = first(Base.return_types($RHS, typeof(($(store.arrays...), $(allfirst...)))))
            $TYP = if Base.isconcretetype($T0)
                $T0
            else
                typeof($RHS($(store.arrays...), $(allfirst...)))
            end
        end)

        # This now checks for OffsetArrays, and allows A[i,1] := ...
        outaxes = map(store.leftraw) do i
            # i === :_ && return :(Base.OneTo(1)) # not understood elsewhere
            i isa Integer && i==1 && return :(Base.OneTo(1))
            i isa Symbol && return Symbol(AXIS, i)
            error("can't use index $i on LHS for a new array")
        end

        if !isdefined(store.mod, :OffsetArrays) # && (:shift in store.flags) # turn off unless needed?
            for r in outaxes
                r == :(Base.OneTo(1)) && continue
                push!(store.outex, :(@assert first($r) == 1 "to allow indices not starting at 1, OffsetArrays must be visible in the caller's module"))
            end
            outaxes = map(r -> :(Base.OneTo($r)), outaxes)
        end

        simex = :( similar($(store.arrays[1]), $TYP, ($(outaxes...),)) )
        if isempty(store.leftnames)
            push!(store.outex, :( $(store.leftarray[]) = $simex ))
        else
            nex = :(tuple($(QuoteNode.(store.leftnames)...)))
            push!(store.outex, :( $(store.leftarray[]) = NamedDims.NamedDimsArray($simex, $nex) ))
        end
    end

    if :zero in store.flags
        push!(store.outex, :( $(store.leftarray[]) .= zero($TYP) ))
    end

end

#========== action functions ==========#

function action_functions(store)

    rn = abs(rand(Int16))
    apply!, create = Symbol(:💥, rn), Symbol(:💧, rn)
    # apply!, create = gensym(:💥), gensym(:💧)

    axisleft = map(i -> Symbol(AXIS, i), store.leftind)
    axisred = map(i -> Symbol(AXIS, i), store.redind)
    axislist = vcat(axisleft, axisred)

    #===== new array =====#
    if :newarray in store.flags
        sofar = Expr(:block, store.outex...)
        empty!(store.outex)
        ST = :($storage_type($(store.leftarray[]), $(store.arrays...)))
        if store.threads
            push!(store.outex, quote
                function $create($(store.arrays...), $(store.scalars...), )
                    $sofar
                    $threader($apply!, $ST, $(store.leftarray[]),
                        tuple($(store.arrays...), $(store.scalars...),),
                        tuple($(axisleft...),), tuple($(axisred...),),
                        $(BLOCK[] ÷ store.cost[]))
                    return $(store.leftarray[])
                end
            end)
        else # no threads
            push!(store.outex, quote
                function $create($(store.arrays...), $(store.scalars...), )
                    $sofar
                    $apply!($ST, $(store.leftarray[]), $(store.arrays...), $(store.scalars...), $(axislist...), )
                    return $(store.leftarray[])
                end
            end)
        end
    end

    #===== constructing loops =====#
    init = store.redfun[] == :* ? :(one($TYP)) :
            store.redfun[] == :max ? :(typemin($TYP)) :
            store.redfun[] == :min ? :(typemin($TYP)) :
            :(zero($TYP))

    ex_init = :( $ACC = $init )

    ex_iter = :( $ACC = $(store.redfun[])($ACC, $(store.right[]) ) )

    ex_write = :( $ZED[$(store.leftraw...)] = $ACC )

    ex_nored = :( $ZED[$(store.leftraw...)] = $(store.right[]) )

    if isempty(store.redind)
        make_many_workers(apply!,
            vcat(:($ZED::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
            nothing, store.leftind, nothing, Symbol[], ex_nored, nothing, store)
    else
        make_many_workers(apply!,
            vcat(:($ZED::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
            nothing, store.leftind, ex_init, store.redind, ex_iter, ex_write, store)
    end

    #===== gradient hooks =====#
    if store.grad != false && (:newarray in store.flags) && !(:nograd in store.flags)
        # First see if you can insert hooks for Zygote/Tracker/Yota
        if backward_definitions(create, apply!, store)
            # If so, calculate ∇create() somehow:
            if store.grad == :Dual
                isdefined(store.mod, :ForwardDiff) || error("grad=Dual can only be used when ForwardDiff is visible")
                insert_forward_gradient(create, apply!, store)
            elseif store.grad == :Base
                insert_base_gradient(create, apply!, store)
            end

            # Need to run Zygote.@adjoint etc. at top level, and it must see create() etc.
            # (Maybe only for Zygote? Not sure ??)
            @eval store.mod begin $(store.outex...) end
            append!(store.outeval, store.outex) # keep these for verbose printing
            empty!(store.outex)
        end
    end

    #===== call something =====#
    ST = :($storage_type($(store.leftarray[]), $(store.arrays...)))
    if :newarray in store.flags
        push!(store.outex, quote
            $(store.leftarray[]) = $create($(store.arrays...), $(store.scalars...), )
        end)
    elseif store.threads
        push!(store.outex, quote
            $threader($apply!, $ST, $(store.leftarray[]),
                tuple($(store.arrays...), $(store.scalars...),),
                tuple($(axisleft...),), tuple($(axisred...),),
                $(BLOCK[] ÷ store.cost[]))
            $(store.leftarray[])
        end)
    else
        push!(store.outex, quote
            $apply!($ST, $(store.leftarray[]), $(store.arrays...), $(store.scalars...), $(axislist...),)
            $(store.leftarray[])
        end)
    end

    if isassigned(store.leftscalar)
        push!(store.outex, :($(store.leftscalar[]) = $(store.leftarray[])[]))
    end
end


"""
    make_many_workers(f!, args, ex1, [:i,], ex3, [:k,], ex5, ex6, store)

This makes several functions of this form,
decorated as necessary with `@inbouds` or `@avx` etc,
and with appropriate `storage_type` as the first argument.
```
f!(::Type, args...) where {T}
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
function make_many_workers(apply!, args, ex1, outer::Vector{Symbol}, ex3, inner::Vector{Symbol}, ex5, ex6, store)

    ex4 = recurseloops(ex5, inner)
    ex2 = recurseloops(:($ex3; $ex4; $ex6), outer)

    push!(store.outex, quote

        function $apply!(::Type, $(args...),) where {$TYP}
            @inbounds @fastmath ($ex1; $ex2)
        end

    end)

    expre, exloop, expost = if isempty(outer)
        :($ex1; $ex3), ex4, ex6
    else
        ex1, ex2, nothing
    end

    # if isdefined(store.mod, :LoopVectorization)
    if store.avx != false && !(:noavx in store.flags)
        LoopVecTypes = Union{Float64,Float32,Int64,Int32}
        if store.avx == true
            push!(store.outex, quote

                function $apply!(::Type{<:Array{<:$LoopVecTypes}}, $(args...),) where {$TYP}
                    $expre
                    $LoopVectorization.@avx $exloop
                    $expost
                end

            end)
        else
            push!(store.outex, quote

                function $apply!(::Type{<:Array{<:$LoopVecTypes}}, $(args...),) where {$TYP}
                    $expre
                    $LoopVectorization.@avx unroll=$(store.avx) $exloop
                    $expost
                end

            end)
        end
    end

    axouter = map(i -> Symbol(AXIS, i), outer)

    if store.cuda > 0 &&
        v"1.3" <= VERSION < v"1.4" &&
        isdefined(store.mod, :KernelAbstractions) &&
        isdefined(store.mod, :CuArrays)

        kernel = Symbol(apply!, :🇨🇺)
        asserts = map(ax -> :(@assert first($ax)==1 "KernelAbstractions can't handle OffsetArrays here"), axouter)
        sizes = map(ax -> :(length($ax)), axouter)
        push!(store.outex, quote

            KernelAbstractions.@kernel function $kernel($(args...),) where {$TYP}
                ($(outer...),) = @index(Global, NTuple)
                ($ex1; $ex3; $ex4; $ex6)
            end

            function $apply!(::Type{<:CuArray}, $(args...),) where {$TYP}
                cu_kern! = $kernel(CUDA(), $(store.cuda))
                # types = map(typeof, ($(args...),))
                # @show types
                $(asserts...)
                $ACC = cu_kern!($(args...); ndrange=tuple($(sizes...)))
                KernelAbstractions.wait($ACC)
            end

            # Just for testing really...
            function $apply!(::Type{<:Array}, $(args...),) where {$TYP}
                cpu_kern! = $kernel(CPU(), Threads.nthreads())
                $(asserts...)
                $ACC = cpu_kern!($(args...); ndrange=tuple($(sizes...)))
                KernelAbstractions.wait($ACC)
            end

        end)
        # Also, bypass "threader" functions to come straight here for CuArrays:
        @eval store.mod begin

            Tullio.threader(fun!::Function, T::Type{<:CuArray},
                Z::AbstractArray, As::Tuple, Is::Tuple, Js::Tuple, block::Int) =
                fun!(T, Z, As..., Is..., Js...)

            Tullio.∇threader(fun!::Function, T::Type{<:CuArray},
                As::Tuple, Is::Tuple, Js::Tuple, block::Int) =
                fun!(T, As..., Is..., Js...)
        end
        # Could do this, but seems not to complain:
        # if hasmethod(threader, Tuple{Function, Type{<:Array}, Vararg})
        # if length(methods(threader)) < 2
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

function backward_definitions(create, apply!, store)
    dZ = Symbol(DEL, ZED)
    ∇create = Symbol(:∇, create)
    ∇apply! = Symbol(:∇, apply!)
    needgrad = false

    if isdefined(store.mod, :Zygote)
        push!(store.outex, quote
            Zygote.@adjoint $create(args...) = $create(args...), Δ -> $∇create(Δ, args...)
        end)
        needgrad = true
    end

    if  isdefined(store.mod, :Yota)
        for (n,A) in enumerate(store.arrays)
            push!(store.outex, quote
                Yota.@diffrule  $create($(store.arrays...), $(store.scalars...))  $A  $∇create(dZ, $(store.arrays...), $(store.scalars...))[$n]
            end)
        end
        needgrad = true
    end

    if isdefined(store.mod, :Tracker)
        push!(store.outex, quote
            $create(A::Tracker.TrackedArray, args...) = Tracker.track($create, A, args...)
            $create(A, B::Tracker.TrackedArray, args...) = Tracker.track($create, A, B, args...)
            $create(A::Tracker.TrackedArray, B::Tracker.TrackedArray, args...) = Tracker.track($create, A, B, args...)
            Tracker.@grad $create(args...) =
                $create(Tracker.data.(args)...), Δ -> $∇create(Δ, Tracker.data.(args)...)
        end)
        needgrad = true
    end

    if isdefined(store.mod, :ReverseDiff) # https://github.com/JuliaDiff/ReverseDiff.jl/pull/123
        push!(store.outex, quote
            $create(A::ReverseDiff.TrackedArray, args...) = ReverseDiff.track($create, A, args...)
            $create(A, B::ReverseDiff.TrackedArray, args...) = ReverseDiff.track($create, A, B, args...)
            $create(A::ReverseDiff.TrackedArray, B::ReverseDiff.TrackedArray, args...) = ReverseDiff.track($create, A, B, args...)
            ReverseDiff.@grad $create(args...) =
                $create(ReverseDiff.value.(args)...), Δ -> $∇create(Δ, ReverseDiff.value.(args)...)
        end)
        needgrad = true
    end

    defineempties = map(A -> :(($(Symbol(DEL, A))) = fill!(similar($A), 0)), store.arrays)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    returns = vcat(gradarrays, )

    # loop order may as well be the same as before?
    loopind = vcat(store.leftind, store.redind)
    # "sharedind" go first in argument list, they are safe to thread over
    shared = map(i -> Symbol(AXIS, i), store.sharedind)
    nonshared = map(i -> Symbol(AXIS, i), setdiff(loopind, store.sharedind))

    if needgrad
        ST = :($storage_type($(gradarrays...), $(store.arrays...)))
        if store.threads
            push!(store.outex, quote
                function $∇create($dZ, $(store.arrays...), $(store.scalars...), )
                    $(defineempties...)
                    $(store.axisdefs...)
                    $∇threader($∇apply!, $ST,
                        tuple($(gradarrays...), $dZ, $(store.arrays...), $(store.scalars...),),
                        tuple($(shared...),), tuple($(nonshared...), ),
                        $(BLOCK[] ÷ store.cost[]))
                    return ($(returns...),)
                end
            end)
        else
            push!(store.outex, quote
                function $∇create($dZ, $(store.arrays...), $(store.scalars...), )
                    $(defineempties...)
                    $(store.axisdefs...)
                    $∇apply!($ST, $(gradarrays...), $dZ, $(store.arrays...), $(store.scalars...), $(shared...), $(nonshared...), )
                    return ($(returns...),)
                end
            end)
        end
    end

    return needgrad
end


#========== the end ==========#
