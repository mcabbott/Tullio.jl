
#========== backward gradient using symbolic derivatives ==========#

using DiffRules

function insert_symbolic_gradient(act!, store)

    dZ = Symbol(DEL, ZED)
    ∇act! = Symbol(:∇, act!)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    # gradscalars = map(A -> Symbol(DEL, A), store.scalars)

    nonshared = setdiff(vcat(store.leftind, store.redind), store.sharedind)

    axislist = map(i -> Symbol(AXIS, i), vcat(store.sharedind, nonshared))

    targets=[]
    MacroTools_postwalk(symbwalk(targets), store.right[])
    # append!(targets, scalars)
    unique!(targets)
    inbody = map(targets) do (dt, t)
        drdt = leibnitz(store.right[], t)
        deltar = simplitimes(drdt, :($dZ[$(store.leftraw...)]))
        :($dt = $dt + $deltar)
    end
    ex_body = commonsubex(quote $(inbody...) end)

    make_many_actors(∇act!,
        vcat(gradarrays, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        # vcat(gradarrays, gradscalars, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        nothing, store.sharedind, nothing, nonshared, ex_body, nothing, store)

end

# This could probably use https://github.com/dfdx/XGrad.jl
# or https://github.com/SciML/ModelingToolkit.jl
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
    ex.head == :call || error("expected a functionn call, got $ex. Use @tullio grad=false if you do not need the gradient.")
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
    error("don't know how to handle $ex. Use @tullio grad=false if you do not need the gradient.")
end

simplitimes(x::Number, y::Number) = x*y
simplitimes(x::Number, y) = x==0 ? 0 : x==1 ? y : x==-1 ? :(-$y) : :($x * $y)
simplitimes(x, y::Number) = y==0 ? 0 : y==1 ? x : y==-1 ? :(-$x) : :($y * $x)
simplitimes(x, y) = :($y * $x)
# simplitimes(x, y) = begin # not worth the hassle, but .e.g.  @printgrad  1/sqrt(z)  z
#     if x isa Expr && y isa Expr && x.head == y.head == :call
#         x.args[1] == y.args[1] == :* && return Expr(:call, :*, x.args[2:end]..., y.args[2:end]...)
#         x.args[1] == :/ && y.args[1] == :* && return :(*($(x.args[2]), $(y.args[2:end]...))/$(x.args[3]))
#         y.args[1] == :/ && x.args[1] == :* && return :(*($(y.args[2]), $(x.args[2:end]...))/$(y.args[3]))
#     end
#     :($y * $x)
# end

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
    error("no diffrule found for function $f($(join(map(_->"_",xs),", "))). Use @tullio grad=false if you do not need the gradient.")
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

function commonsubex(expr::Expr)
    seen = Expr[]
    twice = Dict{Expr,Symbol}()
    MacroTools_postwalk(expr) do @nospecialize ex
        if ex in keys(twice)
            return ex
        elseif ex in seen
            twice[ex] = Symbol(string(ex))
            return ex
        elseif ex isa Expr && ex.head != :ref # && !(ex.head in [:+, :-, :*])
            push!(seen, ex)
        # elseif ex isa Expr && ex.head == :ref
        #     return nothing
        # trying to prevent pulling out [i+j-1] etc, but needs prewalk, which is worse?
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
    MacroTools_prewalk(expr) do @nospecialize ex
        ex == expr && return ex
        if ex in keys(twice)
            sy = twice[ex]
            rules[ex] = sy
            return sy
        end
        ex
    end

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
    done = filter(x -> !(x isa LineNumberNode), commonsubex(out).args)
    map(println, done)
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

=#


#========== the end ==========#
