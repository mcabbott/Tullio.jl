
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

By default it uses LoopVectorization.jl when it can, and Threads.@spawn for big enough arrays.
These options disables both. Option `avx=4` will instead use `@avx unroll=4 for i in ...` loops.

    @tullio  grad=false  C[i,k] := ...

If Zygote.jl/Tracker.jl/Yota.jl are loaded, then it will define gradient hooks for these,
unless disabled by `grad=false`. The gradient itself is calculated in one of two ways,
either by symbolic differentiation of the RHS (the default, `grad=Base`)
or by using dual numbers from ForwardDiff.jl (option `grad=Dual`).

    @tullio  verbose=true

This prints out everythinng the macro knows & generates. You can't always use `@macroexpand1`
as the gradients need things `eval`uated at top level.
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
THREADS = Ref(false)
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

    if @capture(left, Z_[leftraw__] ) || @capture(left, [leftraw__] )
    elseif left isa Symbol
        store.leftscalar[] = left
        leftraw = []
    else
        error("can't understand LHS, expected A[i,j,k], got $left")
    end
    append!(store.leftraw, leftraw)

    append!(store.leftind, reverse(filter(i -> i isa Symbol, leftraw))) # outer loop order
    !allunique(store.leftind) && newarray && push!(store.flags, :zero)

    Zed = isnothing(Z) ? ZED : Z
    store.leftarray[] = Zed

    newarray || saveconstraints(Zed, leftraw, store, false)
    unique!(store.leftind)

    right1 = MacroTools.postwalk(rightwalk(store), right)
    store.right[] = MacroTools.postwalk(dollarwalk(store), right1)
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
        ex isa Expr && ex.head == Symbol(".") && push!(store.flags, :noavx)
        ex isa Symbol && startswith(string(ex), ".") && push!(store.flags, :noavx)

        # Second, alter indexing expr. to pull out functions of arrays:
        @capture(ex, A_[inds__]) || return ex

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
    if @capture(A, B_[inds__]) || @capture(A, B_.field_)
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
    elseif @capture(A, B_[inds__])
        return :( first($B) )
    elseif @capture(A, B_.field_)
        return A
    end

primeindices(inds) = map(inds) do ex
    ex isa Expr && ex.head == Symbol("'") &&
        return Symbol(primeindices(ex.args[1]), "′") # normalise i''
    ex
end

dollarwalk(store) = ex -> begin
        ex isa Expr || return ex
        if ex.head == :call # cost model for threading:
            callcost(ex.args[1], store)
        elseif ex.head == :$ # interpolation of $c things:
            ex.args[1] isa Symbol || error("you can only interpolate single symbols, not $ex")
            push!(store.scalars, ex.args[1])
            return ex.args[1]
        end
        ex
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

        # This just evaluates the first entry, but you could try inference first... run if ::Any?
        allfirst = map(i -> :(first($(Symbol(AXIS, i)))), store.rightind)
        push!(store.outex, :( $TYP = typeof($RHS($(store.arrays...), $(allfirst...))) ))

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

        push!(store.outex, :( $(store.leftarray[]) = similar($(store.arrays[1]), $TYP, ($(outaxes...),)) ))
    end

    if :zero in store.flags
        push!(store.outex, :( $(store.leftarray[]) .= zero($TYP) ))
    end

end

#========== action functions ==========#

function action_functions(store)

    rn = abs(rand(Int8))
    apply!, create, kernel = Symbol(:💥, rn), Symbol(:💧, rn), Symbol(:🇨🇺, rn)
    # apply!, create, kernel = gensym(:💥), gensym(:💧), gensym(:🇨🇺)

    axisleft = map(i -> Symbol(AXIS, i), store.leftind)
    axisred = map(i -> Symbol(AXIS, i), store.redind)
    axislist = vcat(axisleft, axisred)

    if :newarray in store.flags
        sofar = Expr(:block, store.outex...)
        empty!(store.outex)
        if store.threads
            push!(store.outex, quote
                function $create($(store.arrays...), $(store.scalars...), )
                    $sofar
                    $divide($apply!,
                        tuple($(store.leftarray[]), $storage_type($(store.leftarray[]), $(store.arrays...)), $(store.arrays...), $(store.scalars...),),
                        tuple($(axisleft...),), tuple($(axisred...),),
                        $(COST[] ÷ store.cost[]))
                    return $(store.leftarray[])
                end
            end)
        else # no threads
            push!(store.outex, quote
                function $create($(store.arrays...), $(store.scalars...), )
                    $sofar
                    $apply!($(store.leftarray[]), $storage_type($(store.leftarray[]), $(store.arrays...)), $(store.arrays...), $(store.scalars...), $(axislist...), )
                    return $(store.leftarray[])
                end
            end)
        end
    end

    init = store.redfun[] == :* ? :(one($TYP)) :
            store.redfun[] == :max ? :(typemin($TYP)) :
            store.redfun[] == :min ? :(typemin($TYP)) :
            :(zero($TYP))

    if isempty(store.redind)
        writeex = :( $ZED[$(store.leftraw...)] = $(store.right[]) )
    else
        ex = :( $ACC = $(store.redfun[])($ACC, $(store.right[]) ) )
        redloopex = recurseloops(ex, (loop = copy(store.redind), store...))
        writeex = :( $ACC = $init; $redloopex; $ZED[$(store.leftraw...)] = $ACC )
    end

    if isempty(store.leftind)
        preex = :( $ACC = $init )
        loopex = redloopex
        postex = :( $ZED[$(store.leftraw...)] = $ACC )
    else
        loopex = recurseloops(writeex, (loop=copy(store.leftind), store...))
        postex, preex = nothing, nothing # these exist to ensure @avx can act directly on a loop
    end

    #===== basic loops =====#
    push!(store.outex, quote
        function $apply!($ZED::AbstractArray{$TYP}, ::Type, $(store.arrays...), $(store.scalars...), $(axislist...), ) where {$TYP}
            @inbounds ($preex; @fastmath $loopex; $postex)
            # @inbounds ($preex; $loopex; $postex)
        end
    end)

    #===== LoopVectorization =====#
    # if isdefined(store.mod, :LoopVectorization)
    if store.avx != false && !(:noavx in store.flags)
        LoopVecTypes = Union{Float64,Float32,Int64,Int32}
        if store.avx == true
            push!(store.outex, quote
                function $apply!($ZED::AbstractArray{$TYP}, ::Type{<:Array{<:$LoopVecTypes}}, $(store.arrays...), $(store.scalars...), $(axislist...), ) where {$TYP}
                    (@inbounds $preex; $LoopVectorization.@avx $loopex; $postex)
                end
            end)
        else
            push!(store.outex, quote
                function $apply!($ZED::AbstractArray{$TYP}, ::Type{<:Array{<:$LoopVecTypes}}, $(store.arrays...), $(store.scalars...), $(axislist...), ) where {$TYP}
                    (@inbounds $preex; $LoopVectorization.@avx unroll=$(store.avx) $loopex; $postex)
                end
            end)
        end
    end

    #===== KernelAbstractions =====#
    if isdefined(store.mod, :KernelAbstractions) && isdefined(store.mod, :CuArrays) &&
        v"1.3" <= VERSION < v"1.4" && store.cuda > 0 && !(:noavx in store.flags) # assume it's about as fussy!

        push!(store.outex, quote

            KernelAbstractions.@kernel function $kernel($ZED::AbstractArray{$TYP}, $(store.arrays...), $(store.scalars...), $(axisred...), ) where {$TYP}
                ($(store.leftind...),) = # KernelAbstractions.@index(Global, NTuple)
                ($(store.leftind...),) = @index(Global, NTuple)
                $writeex
                nothing
            end
#=
@kernel function matmul_kernel!(a, b, c)
    i, j = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(c))
    for k = 1:size(a)[2]
        tmp_sum += a[i,k] * b[k, j]
    end

    c[i,j] = tmp_sum
end
=#
            # Rather than a method of apply!, perhaps this should be a method of the thread launcher function?
            # But CUDA() |> typeof |> parentmodule == KernelAbstractions which I don't want to depend on...
            # A method like launch(::typeof($apply!), ::Type{<:CuArray}) perhaps?
            function $apply!($ZED::AbstractArray{$TYP}, ::Type{<:CuArray}, $(store.arrays...), $(store.scalars...), $(axislist...), ) where {$TYP}
                cu_kern! = $kernel(CUDA(), $(store.cuda))
                sizes = map(length, tuple($(axisleft...)))
                cu_kern!($(store.arrays...), $(store.scalars...), $(axisred...); ndrange=sizes)
                nothing
            end

            # Just for testing today:
            function $apply!($ZED::AbstractArray{$TYP}, ::Type{<:Array}, $(store.arrays...), $(store.scalars...), $(axislist...), ) where {$TYP}
                @info "using KernelAbstractions on CPU"
                cpu_kern! = $kernel(CPU(), 4)
                sizes = map(length, tuple($(axisleft...)))
                cpu_kern!($(store.arrays...), $(store.scalars...), $(axisred...); ndrange=sizes)
                nothing
            end
#=
function matmul!(a, b, c)
    if size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    if isa(a, Array)
        kernel! = matmul_kernel!(CPU(),4)
    else
        kernel! = matmul_kernel!(CUDA(),256)
    end
    kernel!(a, b, c, ndrange=size(c))
end
=#

        end)
    end

    #===== gradient hooks =====#
    if store.grad != false && (:newarray in store.flags) && !(:nograd in store.flags)
        # First see if you can insert hooks for Zygote/Tracker/Yota
        if backward_definitions(create, apply!, store)
            # If so, calculate ∇create() somehow:
            if store.grad == :Dual
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
    if :newarray in store.flags
        push!(store.outex, quote
            $(store.leftarray[]) = $create($(store.arrays...), $(store.scalars...), )
        end)
    elseif store.threads
        push!(store.outex, quote
            $divide($apply!,
                tuple($(store.leftarray[]), $storage_type($(store.leftarray[]), $(store.arrays...)), $(store.arrays...), $(store.scalars...),),
                tuple($(axisleft...),), tuple($(axisred...),),
                $(COST[] ÷ store.cost[]))
            $(store.leftarray[])
        end)
    else
        push!(store.outex, quote
            $apply!($(store.leftarray[]), $storage_type($(store.leftarray[]), $(store.arrays...)), $(store.arrays...), $(store.scalars...), $(axislist...),)
            $(store.leftarray[])
        end)
    end

    if isassigned(store.leftscalar)
        push!(store.outex, :($(store.leftscalar[]) = $(store.leftarray[])[]))
    end
end

#=
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
    ex2 = recurseloops(:(ex3; ex4; ex6), outer)

    push!(store.outex, quote

        function $apply!(::Type, $(args...),) where {$TYP}
            @inbounds $ex1
            @fastmath $ex2
        end

    end)

    expre, exloop, expost = if isempty(outer)
        :(ex1; ex3), ex4, ex6
    else
        ex1, ex2, nothing
    end

    # if isdefined(store.mod, :LoopVectorization)
    if store.avx != false && !(:noavx in store.flags)
        LoopVecTypes = Union{Float64,Float32,Int64,Int32}
        if store.avx == true
            push!(store.outex, quote

                function $apply!(:Type{<:Array{<:$LoopVecTypes}}, $(args...),) where {$TYP}
                    @inbounds $expre
                    $LoopVectorization.@avx $exloop
                    $expost
                end

            end)
        else
            push!(store.outex, quote

                function $apply!(:Type{<:Array{<:$LoopVecTypes}}, $(args...),) where {$TYP}
                    @inbounds $expre
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

        @info "defining KernelAbstractions kernels, but they give wrong answers?" maxlog=3
        push!(store.outex, quote

            KernelAbstractions.@kernel function $kernel($(args...),) where {$TYP}
                ($(outer...),) = KernelAbstractions.@index(Global, NTuple)
                :(ex1; ex3; ex4; ex6)
            end

            function $apply!(::Type{<:CuArray}, $(args...),) where {$TYP}
                cu_kern! = $kernel(CUDA(), $(store.cuda))
                sizes = map(length, tuple($(axouter...)))
                cu_kern!($(args...); ndrange=sizes)
            end

            # Just for testing today:
            function $apply!(::Type{<:Array}, $(args...),) where {$TYP}
                cpu_kern! = $kernel(CPU(), 4)
                sizes = map(length, tuple($(axouter...)))
                cpu_kern!($(args...); ndrange=sizes)
            end

            # Tullio.newdivide(::typeof($apply!), T::Type{<:CuArray}, args...) = $apply!(T, args...)

        end)
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
=#
recurseloops(ex, storeplus) =
    if !isempty(storeplus.loop)
        i = pop!(storeplus.loop)
        r = Symbol(AXIS, i)
        ex = :(for $i in $r; $ex; end)
        return recurseloops(ex, storeplus)
    else
        return ex
    end

#===== define gradient hooks =====#

function backward_definitions(create, apply!, store)
    dZ = Symbol(DEL, ZED)
    ∇create = Symbol(:∇, create)
    worker! = Symbol(:∇, apply!)
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
        if store.threads
            push!(store.outex, quote
                function $∇create($dZ, $(store.arrays...), $(store.scalars...), )
                    $(defineempties...)
                    $(store.axisdefs...)
                    $divide($worker!,
                        tuple($(gradarrays...), $storage_type($(gradarrays...), $(store.arrays...)), $dZ, $(store.arrays...), $(store.scalars...),),
                        tuple($(shared...),), tuple($(nonshared...), ),
                        $(COST[] ÷ store.cost[]))
                    return ($(returns...),)
                end
            end)
        else
            push!(store.outex, quote
                function $∇create($dZ, $(store.arrays...), $(store.scalars...), )
                    $(defineempties...)
                    $(store.axisdefs...)
                    $worker!($(gradarrays...), $storage_type($(gradarrays...), $(store.arrays...)), $dZ, $(store.arrays...), $(store.scalars...), $(shared...), $(nonshared...), )
                    return ($(returns...),)
                end
            end)
        end
    end

    return needgrad
end


#========== the end ==========#
