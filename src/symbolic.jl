
#========== backward gradient using symbolic derivatives ==========#

using DiffRules

function insert_symbolic_gradient(act!, store)

    dZ = Symbol(DEL, ZED)
    ∇act! = Symbol(:∇, act!)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    # gradscalars = map(A -> Symbol(DEL, A), store.scalars)

    nonshared = setdiff(vcat(store.leftind, store.redind), store.sharedind)

    axislist = map(i -> Symbol(AXIS, i), vcat(store.sharedind, nonshared))

    targets = []
    MacroTools_postwalk(symbwalk(targets), store.right)
    # append!(targets, scalars)
    unique!(targets)
    inbody = map(targets) do (dt, t)
        drdt = leibnitz(store.right, t)
        deltar = simplitimes(drdt, :(conj($dZ[$(store.leftraw...)])))
        :($dt = $dt + conj($deltar))
    end
    ex_body = commonsubex(quote $(inbody...) end)

    make_many_actors(∇act!,
        vcat(gradarrays, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        # vcat(gradarrays, gradscalars, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        nothing, store.sharedind, nothing, nonshared, ex_body, nothing, store, " (symbolic gradient)")

    if isdefined(store.mod, :Zygote) # special case for FillArrays
        ex_body2 = fillarrayreplace(ex_body, dZ)
        ex_value = :($(Symbol(dZ, :_value)) = $dZ.value) # @avx likes this outside the loop

        make_many_actors(∇act!,
            vcat(gradarrays, :($dZ::Zygote.Fill{$TYP}), store.arrays, store.scalars, axislist),
            ex_value, store.sharedind, nothing, nonshared, ex_body2, nothing, store, " (method for FillArrays)")
    end

end

# This could probably use https://github.com/dfdx/XGrad.jl
# or https://github.com/SciML/ModelingToolkit.jl
# or now I found this: https://github.com/HarrisonGrodin/Simplify.jl
# but seemed simple enough to just write out, using rules from:
# http://www.juliadiff.org/DiffRules.jl/latest/

symbwalk(targets) = ex -> begin
        @capture_(ex, A_[inds__]) && A isa Symbol || return ex
        deltaex = :($(Symbol(DEL, A))[$(inds...)])
        push!(targets, (deltaex, ex))
        return ex
    end

leibnitz(s::Number, target) = 0
leibnitz(s::Symbol, target) = s == target ? 1 : 0
leibnitz(ex::Expr, target) = begin
    ex == target && return 1
    @capture_(ex, B_[ijk__]) && return 0
    if ex.head == Symbol("'")
        ex.head = :call
        pushfirst!(ex.args, :adjoint)
    end
    ex.head == :call || error("expected a functionn call, got $ex.")
    fun = ex.args[1]
    if fun == :log # catch log(a*b) and especially log(a/b)
        arg = ex.args[2]
        if arg isa Expr && arg.args[1] == :* && length(arg.args) == 3
            newex = :(log($(arg.args[2])) + log($(arg.args[3])))
            return leibnitz(newex, target)
        elseif arg isa Expr && arg.args[1] == :/
            newex = :(log($(arg.args[2])) - log($(arg.args[3])))
            return leibnitz(newex, target)
        end
    end
    if length(ex.args) == 2 # one-arg function
        fx = mydiffrule(fun, ex.args[2])
        dx = leibnitz(ex.args[2], target)
        return simplitimes(fx, dx)
    elseif length(ex.args) == 3  # two-arg function
        fx, fy = mydiffrule(fun, ex.args[2:end]...)
        dx = leibnitz(ex.args[2], target)
        dy = leibnitz(ex.args[3], target)
        return simpliplus(simplitimes(fx, dx), simplitimes(fy, dy))
    elseif fun in [:+, :*]
        fun == :* && return leibnitz(:(*($(ex.args[2]), *($(ex.args[3:end]...)))), target)
        dxs = [leibnitz(x, target) for x in ex.args[2:end]]
        fun == :+ && return simpliplus(dxs...)
    end
    error("don't know how to handle $ex.")
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
    f == :// && return mydivrule(xs...)
    f == :inv && return mydivrule(1, xs...)[2]
    f == :log && return simpliinv(xs...)
    f == :sqrt && return mysqrtrule(xs...)
    f == :trunc && return map(_->0, xs)
    f == :round && return map(_->0, xs)
    DiffRules.hasdiffrule(:Base, f, length(xs)) &&
        return DiffRules.diffrule(:Base, f, xs...)
    DiffRules.hasdiffrule(:SpecialFunctions, f, length(xs)) &&
        return DiffRules.diffrule(:SpecialFunctions, f, xs...)
    error("no diffrule found for function $f($(join(map(_->"_",xs),", "))).")
end

# Goals of these rules, besides correctness, are:
# 1. don't cause promotion of Float32, e.g. by factors (1/2)
# 2. make it easy for commonsubex(), e.g. by re-using inv(x)

mydivrule(x, y) = begin # (:(one(x) / y), :(-((x / y) / y)))
    invy = simpliinv(y)
    invy, :( -($x) * $invy * $invy )
end
mydivrule(x, y::Integer) = (y==1 ? 1 : 1//y), 0
mydivrule(x, y::Number) = (y==1 ? 1 : :(one($TYP)/$y)), 0

mydivrule(x::Number, y) = 0, :((-$x)*inv($y)*inv($y))
mydivrule(x::Number, y::Number) = 0, 0
mydivrule(x::Number, y::Integer) = 0, 0

mysqrtrule(x::Number) = sqrt(x)
mysqrtrule(x) = :(inv(sqrt($x))/2)

simpliinv(x) = :(inv($x))
simpliinv(x::Integer) = :(1//$x)
simpliinv(x::Expr) = if x.head == :call && x.args[1] == :/
        :($(x.args[3]) / $(x.args[2]))
    else
        :(inv($x))
    end

mypowrule(x, p) = begin
    dx = simplitimes(p, simplipow(x, simpliplus(p, -1)))
    dp = simplitimes(simplipow(x,p), :(log($x)))
    dx, dp
end

simplipow(x::Number, p::Number) = x^p
simplipow(x, p::Number) = p==1 ? x : p==2 ? :($x*$x) : :($x^$p)
simplipow(x, p) = :($x^$p)

#========== CSE ==========#

# My approach was to look for things occuring twice, biggest first.
# Then I found https://github.com/rdeits/CommonSubexpressions.jl
# which just pulls everything out, but doesn't like indexing expressions.

function commonsubex(expr::Expr)
    dict, defs, nope = Dict(), [], Set()
    if expr.head == :block
        args = [csewalk(ex, dict, defs, nope) for ex in copy(expr).args]
        quote
            $(defs...)
            $(args...)
        end
    else
        res = csewalk(copy(expr), dict, defs, nope)
        quote
            $(defs...)
            $res
        end
    end
end

csewalk(ex, dict, defs, nope) = ex
csewalk(ex::Expr, dict::Dict, defs::Vector, nope::Set) =
    # The goal is to alter RHS of assignments,
    # this mess is the most common case, A = A + stuff
    if ex.head == :(=) && ex.args[2] isa Expr && ex.args[2].head == :call &&
        ex.args[2].args[1] == :+ && ex.args[2].args[2] == ex.args[1]
        for n in 3:length(ex.args[2].args)
            ex.args[2].args[n] = csewalk(ex.args[2].args[n], dict, defs, nope)
        end
        push!(nope, ex.args[1]) # new Ex3 = ... cannot have this on RHS
        ex
    elseif ex.head in (:(=), :(+=)) # easier case of A = stuff
        push!(nope, ex.args[1])
        ex.args[2] = csewalk(ex.args[2], dict, defs, nope)
        ex

    # Then we work on sub-expressions, replace those we're seen immediately,
    # and don't look inside A[i,j] at all:
    elseif haskey(dict, ex)
        dict[ex]
    elseif ex.head == :ref
        ex

    # Simplest case is the last one, replace a whole expression with Ex5 & work inwards.
    # Can't replace "illegal" expressions, but can look for parts which are safe:
    elseif illegal(ex, nope)
        args = Any[x in nope ? x : csewalk(x, dict, defs, nope) for x in ex.args]
        Expr(ex.head, args...)

    elseif ex.head == :call && ex.args[1] in (:*, :+) && length(ex.args) >= 4 # e.g. 1*2*3
        inner = []
        while length(ex.args) >= 3
            pushfirst!(inner, pop!(ex.args))
        end
        binary = Expr(:call, ex.args..., Expr(:call, ex.args[1], inner...))
        csewalk(binary, dict, defs, nope)

    else
        args = Any[csewalk(x, dict, defs, nope) for x in ex.args]
        sy = Symbol(EXPR, length(defs)+1)
        dict[ex] = sy
        # add defn for the outermost operation:
        push!(defs, Expr(:(=), sy, Expr(ex.head, args...)))
        # and return the name for caller:
        sy
    end

illegal(ex, nope) = ex in nope
illegal(ex::Expr, nope) = ex in nope || any(illegal(x, nope) for x in ex.args)

#========== examination ==========#

"""
    Tullio.@printgrad log(x/y) x y

Prints the symbolic gradient, showing `∂f/∂x` and `∂f/∂y` for `f=log(x/y)`.
Useful to check that simplifications, and common subexpression elimination,
are working OK for a given RHS.
"""
macro printgrad(exs...)
    printgrad(exs...)
end

function printgrad(ex::Expr, ts::Symbol...)
    out = quote end
    for t in ts
        df = leibnitz(ex, t)
        dt = Symbol(:δ, t) # Symbol("∂f_∂", t)
        push!(out.args, :($dt = $df))
    end
    print("Initial:\n   ")
    println(join(filter(x -> !(x isa LineNumberNode), out.args), "\n   "))
    print("After CSE:\n   ")
    done = filter(x -> !(x isa LineNumberNode), commonsubex(out).args)
    println(join(done, "\n   "))
    nothing
end

#=

using Tullio: @printgrad

@printgrad  x * y * z   x y z
@printgrad  x * (y * z)   x y z
@printgrad  x + y * z   x y z

@printgrad  1/x   x
@printgrad  x^-1   x   # could make inv(x) for CSE
@printgrad  inv(x)   x
@printgrad  sqrt(x)   x
@printgrad  1/sqrt(x)   x
@printgrad  inv(sqrt(x))   x
@printgrad  x/sqrt(y)   x y

@printgrad  sqrt(x*y)   x y
@printgrad  sqrt(x) * sqrt(y)   x y # worse than line above

@printgrad  1/sqrt(x*y)   x y       # could use repeated CSE

@printgrad  x/sqrt(y*z)   x y z     # could use repeated CSE
@printgrad  x/(sqrt(y)*sqrt(z))   x y z
@printgrad  x*inv(sqrt(y))*inv(sqrt(z))   x y z

@printgrad  x/2   x
@printgrad  x/y   x y

@printgrad  x^2   x
@printgrad  (x*y)^2   x y
@printgrad  (x+y)^3   x y
@printgrad  x^y   x y
@printgrad  log(x)^2   x

@printgrad  log(x)   x
@printgrad  log(x/2)   x
@printgrad  log(2x)   x
@printgrad  log(k*x)   x

@printgrad  x*log(y)   x y

@printgrad  log(x*y)   x y
@printgrad  log(x) + log(y)   x y  # better, now used for log(x*y)

@printgrad  log(x/y)   x y
@printgrad  log(x*inv(y))   x y
@printgrad  log(x)-log(y)   x y    # much better, now used for log(x/y)

@printgrad  log(x/y) * z   x y z
@printgrad  (log(x) - log(y)) * z   x y z
@printgrad  log(x)*z - log(y)* z   x y z

@printgrad  exp(2x)   x)
@printgrad  exp(x/y)   x y
@printgrad  exp((x-y)^2/2)   x y

@printgrad  exp(x) * y   x y
@printgrad  exp(x) / 2y   x y

@printgrad a * b / sqrt(d * e)  a b d e
@printgrad x * z / sqrt(y * z)  x y z

=#


#========== the end ==========#
