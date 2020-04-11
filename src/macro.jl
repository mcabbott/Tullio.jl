
VERBOSE = Ref(false)
AVX = Ref(true)
# GRAD = Ref{Union{Symbol,Nothing}}(:ForwardDiff)
GRAD = Ref{Union{Symbol,Nothing}}(:Base)

mutable struct Store store::NamedTuple end
Base.parent(x::Store) = getfield(x, :store)
Base.getproperty(x::Store, y::Symbol) = getproperty(parent(x), y)
Base.merge(x::NamedTuple, y::Store) = Store(merge(x, parent(y)))

#========== the macro! ==========#

const ExprSym = Union{Expr, Symbol}

"""
    @tullio C[i,k] := A[i,j] * B[j,k]
    @tullio C[i,k] := A[i].field[j] * B[row=j, col=k]

This is a replacement for `@einsum` which understands a bit more syntax,
and which uses LoopVectorization.jl when it can (and when `Tullio.AVX[] == true`).

    @tullio Base
    @tullio ForwardDiff

If Zygote.jl/Tracker.jl/Yota.jl are loaded, then it will define gradient hooks for these.
The gradient itself is calculated in one of two ways, either by symbolic differentiation of the RHS,
or by using dual numbers from ForwardDiff.jl, and this is how you choose.
"""
macro tullio(ex)
    _tullio(ex; mod=__module__)
end

function _tullio(ex1, ex2=:+; mod=Main)
    ex1 isa Symbol &&
        if ex1 in [:ForwardDiff, :Base]
            GRAD[] = ex1
            return nothing
        else
            error("don't understand $ex1")
        end

    store = Store((mod = mod,
        flags = Set{Symbol}(),
        upop = Ref{Symbol}(:(=)), # allow *=  for @einsum compat, not yet done
        redfun = Ref{Symbol}(:+),
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
    # Index ranges, first constraints then equal/intersect
        constraints = Dict{Symbol,Set}(),
        ranges = Dict{Symbol,ExprSym}(),
    # Version of right with (A[i,j] + 𝜀A′) etc, with dict[:𝜀A′] = A[i,j]
        epsilonright = Ref{ExprSym}(),
        epsilondict = Dict{Symbol,Expr}(),
    # Expressions: outex is the main one, sometimes wrapped innto functions.
        outpre = ExprSym[], # things never to be inside function
        outeval = ExprSym[], # things already @eval-ed at top level for gradient.
        outex = ExprSym[],
    ))

    parse_input(ex1, ex2, store)

    index_ranges(store)

    output_array(store)

    action_functions(store)

    nt = (parent(store)..., AVX=AVX[], GRAD=GRAD[])
    VERBOSE[] && foreach(keys(nt)) do k
        r = getproperty(nt, k) # startswith(string(k), "out") fails?
        k ∉ [:outpre, :outeval, :outex] && return printstyled("    $k = ", repr(r), "\n", color=:blue)
        printstyled("    $k =\n", color=:blue)
        foreach(ex -> printstyled(MacroTools.prettify(ex) , "\n", color=:green), r)
    end

    Expr(:block, store.outpre..., store.outex...) |> esc
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

    # newarray && all(i -> isa(i,Symbol)||(i==1)
    !allunique(store.leftind) && newarray && push!(store.flags, :zero) # but will that work??

    Zed = isnothing(Z) ? ZED : Z
    store.leftarray[] = Zed

    newarray || saveconstraints(Zed, leftraw, store, false)

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
    is = map(enumerate(inds)) do (d,ex)
        isconst(ex) && return nothing
        ex isa Symbol || push!(store.flags, :intersect) # ?? might not be right
        ri, i = range_expr_walk(:(axes($A1,$d)), ex)
        get!(store.constraints, i, Set{Expr}())
        isnothing(ri) || push!(store.constraints[i], ri)
        i
    end
    if right
        is = filter(!isnothing, is)
        append!(store.rightind, is)
        if isassigned(store.sharedind)
            shared = intersect(is, store.sharedind)
            empty!(store.sharedind)
            append!(store.sharedind, shared)
        else
            append!(store.sharedind, is)
        end
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
        # cost model for threading:
        ex in LOGEXP && push!(store.flags, :logexp)

        # interpolation of $c things:
        ex isa Expr && ex.head == :$ || return ex
        ex.args[1] isa Symbol || error("you can only interpolate single symbols, not $ex")
        push!(store.scalars, ex.args[1])
        return ex.args[1]
    end


#========== index ranges ==========#

function index_ranges(store)

    allinds = vcat(store.leftind, store.redind)
    if :intersect in store.flags
        foreach(resolveintersect(store), allinds)
    else
        foreach(resolvestrict(store), allinds)
    end

    # for j in setdiff(leftind, store.rightind)
    #     haskey(store.ranges, j) || error("unable to infer range of index $j")
    #     push!(store.rightind, j) # I guess right now means correct!
    # end

end

resolvestrict(store) = i ->
    for ax in store.constraints[i]
        if haskey(store.ranges, i)
            str = "range of index $i must agree"
            push!(store.outex, :( @assert $(store.ranges[i]) == $ax $str ))
        else
            r_i = Symbol(AXIS, i)
            push!(store.outex, :( local $r_i = $ax ))
            store.ranges[i] = ax
        end
    end

resolveintersect(store) = i ->
    begin
        res = length(store.constraints[i])==1 ?
            first(store.constraints[i]) : # because intersect(1:3) isa Vector, wtf?
            :( intersect($(store.constraints[i]...)) )
        r_i = Symbol(AXIS, i)
        push!(store.outex, :( local $r_i = $res ))
        store.ranges[i] = r_i
    end

#========== output array + eltype ==========#

function output_array(store)

    if :newarray in store.flags
        funwithargs = :( $RHS($(store.arrays...), $(store.rightind...)) )
        push!(store.outex, :( $funwithargs = $(store.right[]) ))

        # This just evaluates the first entry, but you could try inference first... run if ::Any?
        allfirst = map(i -> :(first($(Symbol(AXIS, i)))), store.rightind)
        push!(store.outex, :( $TYP = typeof($RHS($(store.arrays...), $(allfirst...))) ))

        outranges = map(i -> Symbol(AXIS, i), store.leftraw)
        push!(store.outex, :( $(store.leftarray[]) = similar($(store.arrays[1]), $TYP, ($(outranges...),)) ))
    end

    if :zero in store.flags
        push!(store.outex, :( $(store.leftarray[]) .= zero($TYP) ))
    end

end

#========== action functions ==========#

function action_functions(store)

    rn = abs(rand(Int8))
    apply!, create = Symbol(:💥, rn), Symbol(:💧, rn)

    axislist = map(i -> Symbol(AXIS, i), store.rightind)

    if :newarray in store.flags
        sofar = Expr(:block, store.outex...)
        empty!(store.outex)
        push!(store.outex, quote
            function $create($(store.arrays...), $(store.scalars...), )
                $sofar
                $apply!($(store.leftarray[]), $storage_type($(store.leftarray[]), $(store.arrays...)), $(store.arrays...), $(store.scalars...), $(axislist...), )
                return $(store.leftarray[])
            end
        end)
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
        loopex = recurseloops(writeex, (loop=unique(store.leftind), store...))
        postex, preex = nothing, nothing
    end

    push!(store.outex, quote
        function $apply!($ZED::AbstractArray{$TYP}, ::Type, $(store.arrays...), $(store.scalars...), $(axislist...), ) where {$TYP}
            @inbounds ($preex; @fastmath $loopex; $postex)
            # @inbounds ($preex; $loopex; $postex)
        end
    end)

    # if isdefined(store.mod, :LoopVectorization)
    if AVX[] && !(:noavx in store.flags)
        LoopVecTypes = Union{Float64,Float32,Int64,Int32,Int8}
        push!(store.outex, quote
            function $apply!($ZED::AbstractArray{$TYP}, ::Type{<:Array{<:$LoopVecTypes}}, $(store.arrays...), $(store.scalars...), $(axislist...), ) where {$TYP}
                (@inbounds $preex; $LoopVectorization.@avx $loopex; $postex)
            end
        end)
    end

    if GRAD[] != nothing && (:newarray in store.flags) && !(:nograd in store.flags)
        # First see if you can insert hooks for Zygote/Tracker/Yota
        if backward_definitions(create, apply!, store)
            # If so, calculate ∇create() somehow:
            if GRAD[] == :ForwardDiff
                insert_forward_gradient(create, apply!, store)
            elseif GRAD[] == :Base
                insert_base_gradient(create, apply!, store)
            end

            # Need to run Zygote.@adjoint etc. at top level, and it must see create() etc.
            # (Maybe only for Zygote? Not sure ??)
            @eval store.mod begin $(store.outex...) end
            append!(store.outeval, store.outex) # keep these for verbose printing
            empty!(store.outex)
        end
    end

    if :newarray in store.flags
        push!(store.outex, :( $(store.leftarray[]) = $create($(store.arrays...), $(store.scalars...), ) ))
    else
        push!(store.outex, :( $apply!($(store.leftarray[]), $storage_type($(store.leftarray[]), $(store.arrays...)), $(store.arrays...), $(store.scalars...), $(axislist...),) ))
        push!(store.outex, store.leftarray[])
    end

    if isassigned(store.leftscalar)
        push!(store.outex, :($(store.leftscalar[]) = $(store.leftarray[])[]))
    end
end

recurseloops(ex, storeplus) =
    if !isempty(storeplus.loop)
        i = pop!(storeplus.loop)
        r = Symbol(AXIS, i)
        ex = # (:gpu in storeplus.flags) ?
            # :( Tullio.GPUifyLoops.@loop for $i in $r; $ex; end ) :
            :(for $i in $r; $ex; end)
        return recurseloops(ex, storeplus)
    else
        return ex
    end

# gpuranges(n) = n>3 ? error("only 3 gpu loops for now") :
#     [:( threadIdx().x ), :( threadIdx().y ), :( threadIdx().z )][1:n]

#===== define gradient hooks =====#

function backward_definitions(create, apply!, store)

    dZ = Symbol(DEL, ZED)
    ∇create = Symbol(:∇, create) # takes dZ, then same arguments as create()
    worker! = Symbol(:∇, apply!) # gradarrays, delta, arrays, ranges
    needgrad = false

    if isdefined(store.mod, :Zygote)
        push!(store.outex, quote
            Zygote.@adjoint $create(args...) = $create(args...), Δ -> $∇create(Δ, args...)
        end)
        needgrad = true
    end

    if  isdefined(store.mod, :Yota) # untested!
        for (d,A) in enumerate(store.arrays)
            push!(store.outex, quote
                Yota.@diffrule  $create(args...)  $create(args...)  $A  $∇create(ds, args...)[$d]
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

    defineranges = map(loopind) do i
        :( $(Symbol(AXIS, i)) = $(store.ranges[i]) )
    end

    push!(store.outex, quote
        function $∇create($dZ, $(store.arrays...), $(store.scalars...), )
            $(defineempties...)
            $(defineranges...)
            $worker!($(gradarrays...), $storage_type($(gradarrays...), $(store.arrays...)), $dZ, $(store.arrays...), $(store.scalars...), $(shared...), $(nonshared...), )
            return ($(returns...),)
        end
    end)

    return needgrad
end

#========== the end ==========#
