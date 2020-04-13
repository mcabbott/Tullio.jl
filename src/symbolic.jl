
#========== backward gradient using symbolic derivatives ==========#

using DiffRules

function insert_base_gradient(create, apply!, store)
    store.verbose && @info "using symbolic gradient for: $create ~ $(store.right[])"

    dZ = Symbol(DEL, ZED)
    worker! = Symbol(:∇, apply!)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)

    loopind = vcat(store.leftind, store.redind)
    shared = map(i -> Symbol(AXIS, i), store.sharedind)
    nonshared = map(i -> Symbol(AXIS, i), setdiff(loopind, store.sharedind))

    targets=[]
    MacroTools.postwalk(symbwalk(targets), store.right[])
    unique!(targets)
    inbody = map(targets) do (dt, t)
        drdt = leibnitz(store.right[], t)
        deltar = simplitimes(drdt, :($dZ[$(store.leftraw...)]))
        :($dt = $dt + $deltar)
    end

    ex = commonsubex(quote $(inbody...) end)
    loopex = recurseloops(ex, (loop = loopind, store...))

    push!(store.outex, quote
        function $worker!($(gradarrays...), ::Type, $dZ::AbstractArray{$TYP}, $(store.arrays...), $(store.scalars...), $(shared...), $(nonshared...), ) where {$TYP}
            @fastmath @inbounds $loopex
        end
    end)

    if AVX[] && !(:noavx in store.flags)
        LoopVecTypes = Union{Float64,Float32,Int64,Int32,Int8}
        push!(store.outex, quote
            function $worker!($(gradarrays...), ::Type{<:Array{<:$LoopVecTypes}}, $dZ::AbstractArray{$TYP}, $(store.arrays...), $(store.scalars...), $(shared...), $(nonshared...), ) where {$TYP}
                $LoopVectorization.@avx $loopex
            end
        end)
    end
end

# This could probably use https://github.com/dfdx/XGrad.jl
# or https://github.com/SciML/ModelingToolkit.jl
# but seemed simple enough to just write out, using rules from:
# http://www.juliadiff.org/DiffRules.jl/latest/

symbwalk(targets) = ex -> begin
        @capture(ex, A_[inds__]) && A isa Symbol || return ex
        deltaex = :($(Symbol(DEL, A))[$(inds...)])
        push!(targets, (deltaex, ex))
        return ex
    end

leibnitz(s::Number, target) = 0
leibnitz(s::Symbol, target) = s == target ? 1 : 0
leibnitz(ex::Expr, target) = begin
    ex == target && return 1
    @capture(ex, B_[ijk__]) && return 0
    # if ex.head == Symbol("'")
    #     ex.head = :call
    #     pushfirst!(ex.args, :adjoint)
    # end
    ex.head == :call || error("wtf is $ex")
    fun = ex.args[1]
    if length(ex.args) == 2 # one-arg function
        fx = mydiffrule(fun, ex.args[2])
        dx = leibnitz(ex.args[2], target)
        simplitimes(fx, dx)
    elseif length(ex.args) == 3  # one-arg function
        fx, fy = mydiffrule(fun, ex.args[2:end]...)
        dx = leibnitz(ex.args[2], target)
        dy = leibnitz(ex.args[3], target)
        return simpliplus(simplitimes(fx, dx), simplitimes(fy, dy))
    elseif fun in [:+, :*]
        fun == :* && return leibnitz(:(*($(ex.args[2]), *($(ex.args[3:end]...)))), target)
        dxs = [leibnitz(x, target) for x in ex.args[2:end]]
        fun == :+ && return simpliplus(dxs...)
        error("don't know how to handle $ex")
    end
end

simplitimes(x::Number, y::Number) = x*y
simplitimes(x::Number, y) = x==0 ? 0 : x==1 ? y : x==-1 ? :(-$y) : :($x * $y)
simplitimes(x, y::Number) = y==0 ? 0 : y==1 ? x : y==-1 ? :(-$x) : :($y * $x)
simplitimes(x, y) = :($y * $x)

simpliplus(x::Number, y::Number) = x + y
simpliplus(x::Number, y) = x==0 ? y : :($x + $y)
simpliplus(x, y::Number) = y==0 ? x : :($x + $y)
simpliplus(x, y) = :($x + $y)
simpliplus(x, y, zs...) = simpliplus(simpliplus(x, y), zs...)

mydiffrule(f, xs...) = begin
    f == :+ && return map(_->1, xs)
    f == :- && return length(xs)==1 ? -1 : (1,-1)
    f == :^ && return mypowrule(xs...)
    f == :/ && return mydivrule(xs...)
    f == :log && return simpliinv(xs...)
    f == :trunc && return map(_->0, xs)
    DiffRules.hasdiffrule(:Base, f, length(xs)) &&
        return DiffRules.diffrule(:Base, f, xs...)
    DiffRules.hasdiffrule(:SpecialFunctions, f, length(xs)) &&
        return DiffRules.diffrule(:SpecialFunctions, f, xs...)
    error("don't know about $f()")
end

mydivrule(x, y) = simpliinv(y), :( -$x / ($y * $y) ) # (:(one(x) / y), :(-((x / y) / y)))
mydivrule(x, y::Integer) = (y==1 ? 1 : 1//y), 0
mydivrule(x, y::Number) = (y==1 ? 1 : :(one($TYP)/$y)), 0

simpliinv(x::Expr) = if x.head == :call && x.args[1] == :/
        :($(x.args[3]) / $(x.args[2]))
    else
        :(one($TYP) / $x)
    end

mypowrule(x, p) = simplitimes(p, simplipow(x, simpliplus(p, -1))), simplitimes(simplipow(x,p), :(log($x)))

simplipow(x::Number, p::Number) = x^p
simplipow(x, p::Number) = p==1 ? x : p==2 ? :($x*$x) : :($x^$p)
simplipow(x, p) = :($x^$p)

function commonsubex(expr::Expr)
    seen = Expr[]
    twice = Dict{Expr,Symbol}()
    MacroTools.postwalk(expr) do ex
        if ex in keys(twice)
            return ex
        elseif ex in seen
            twice[ex] = Symbol(string(ex))
            return ex
        elseif ex isa Expr && ex.head != :ref
            push!(seen, ex)
        end
        ex
    end
    rules = Dict{Expr,Symbol}()
    out = commonapply(expr, twice, rules)
    for (ex,sy) in pairs(rules)
        pushfirst!(out.args, :($sy = $ex))
    end
    out
end

commonapply(expr, twice, rules) =
    MacroTools.prewalk(expr) do ex
        ex == expr && return ex
        if ex in keys(twice)
            sy = twice[ex]
            rules[ex] = sy
            return sy
        end
        ex
    end

#========== the end ==========#
