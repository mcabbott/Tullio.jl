
#========== backward gradient using symbolic derivatives ==========#

using DiffRules

function insert_symbolic_gradient(axislist, store)

    dZ = Symbol(DEL, ZED)
    ∇act! = Symbol(:∇, ACT!)
    maxflag = Symbol(RHS, :⛰)
    maxdone = Symbol(:🆗, RHS)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    # gradscalars = map(A -> Symbol(DEL, A), store.scalars)

    out_ind, in_ind = if store.redfun == :+
        store.sharedind, setdiff(vcat(store.leftind, store.redind), store.sharedind)
    elseif store.redfun in [:min, :max] # :*,
        store.leftind, store.redind
    else
        throw("can't take gradients with reduction $(store.redfun)")
    end

    targets = []
    MacroTools_postwalk(symbwalk(targets, store), store.right)
    # append!(targets, scalars)

    if isempty(targets) # short-circuit
        push!(store.outpre, :(local @inline $∇act!(::Type, args...) = nothing ))
        store.verbose > 0 && @info "no gradient to calculate"
        return nothing
    end

    inbody, prebody = [], []
    for (dt, t) in unique(targets)
        drdt = leibnitz(store.right, t)
        deltar = if store.finaliser == :identity
            simplitimes(simpliconj(drdt), :($dZ[$(store.leftraw...)]))
        else
            rhs = :($ZED[$(store.leftraw...)])
            dldr = leibfinal(store.finaliser, rhs)
            simplitimes(simpliconj(drdt), simpliconj(dldr), :($dZ[$(store.leftraw...)]))
        end
        if store.redfun == :+
            push!(inbody, :($dt = $dt + $deltar))
        # elseif store.redfun == :*
        #     push!(inbody, :($dt = $deltar * $ZED[$(store.leftraw...)] * inv($(store.right))))
        #     push!(prebody, :($dt = $deltar * $ACC))
        elseif store.redfun in [:min, :max]
            push!(inbody, :($dt += $ifelse($maxflag, $deltar, $zero($TYP))))
        end
    end
    store.verbose>0 && @info "symbolic gradients" inbody
    ex_body = commonsubex(quote $(inbody...) end)

    ex_pre, ex_post = if store.redfun == :* # then nonzero LHS are handled already, but harder cases here:
        product_grad(prebody, store)
    elseif store.redfun in [:min, :max]
        :($maxdone = 0), nothing
    else
        nothing, nothing
    end
    if store.redfun in [:min, :max] # this case really wants sparse 𝛥x!
        ex_body = :(
            $maxflag = Tullio.onlyone($ZED[$(store.leftraw...)] == $(store.right), $maxdone);
            $ex_body;
            $maxdone += Tullio.anyone($maxflag);
            )
    end

    make_many_actors(∇act!,
        vcat(gradarrays, :($dZ::AbstractArray{$TYP}), ZED, store.arrays, store.scalars, axislist),
        # vcat(gradarrays, gradscalars, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        nothing, out_ind, ex_pre, in_ind, ex_body, ex_post, store, "(symbolic gradient)")

    if isdefined(store.mod, :Zygote) && !(store.scalar) # special case for FillArrays
        ex_body2 = fillarrayreplace(ex_body, dZ)
        ex_pre2 = fillarrayreplace(ex_pre, dZ)
        ex_value = :($(Symbol(dZ, :_value)) = $dZ.value) # @avx likes this outside the loop

        make_many_actors(∇act!,
            vcat(gradarrays, :($dZ::Zygote.Fill{$TYP}), ZED, store.arrays, store.scalars, axislist),
            ex_value, out_ind, ex_pre2, in_ind, ex_body2, ex_post, store, "(gradient method for FillArrays)")
    end

end

leibfinal(fun::Symbol, res) =
    if fun == :log
        :(exp(-$res)) # this exp gets done at every element :(
        # :(inv(exp($res)))
    else
        _leibfinal(:($fun($RHS)), res)
    end

_leibfinal(out, res) = begin
    grad1 = leibnitz(out, RHS)
    grad2 = MacroTools_postwalk(grad1) do ex
        # @show ex ex == out
        ex == out ? res : ex
    end
    MacroTools_postwalk(grad2) do ex
        ex == RHS ? throw("couldn't eliminate partial sum") : ex
    end
end

leibfinal(ex::Expr, res) = begin
    if ex.head == :call && ex.args[1] isa Expr &&
        ex.args[1].head == :(->) && ex.args[1].args[1] == RHS # then it came from underscores
        inner = ex.args[1].args[2]
        if inner isa Expr && inner.head == :block
            lines = filter(a -> !(a isa LineNumberNode), inner.args)
            length(lines) == 1 && return _leinfinal(first(lines), res)
        end
    end
    throw("couldn't understand finaliser")
end

#=
Tullio.leibfinal(:exp, :res)   # :res
Tullio.leibfinal(:sqrt, :res)  # :(inv(res) / 2)
Tullio.leibfinal(:tanh, :res)  # :(1 - res ^ 2)
Tullio.leibfinal(:log, :res)   # :(exp(-res))
=#

# This works for simple cases, but the general case is more complicatd.
#=
product_grad(prebody, store) = begin
    cnt = Symbol(DEL,:𝒸ℴ𝓊𝓃𝓉,0)

    inds_orig = :(($(store.redind...),))
    inds_prime = :(($(map(i -> Symbol(i,'′',DEL), store.redind)...),))
    inds_zero = :(($(map(i -> 0, store.redind)...),))

    loop_search = recurseloops(:(
        # find and save the index at which RHS is zero
        if iszero($(store.right))
            $cnt += 1
            $inds_prime = $inds_orig
        end
    ), copy(store.redind))

    loop_accum = recurseloops(:(
        # product of RHS at all redind except the one which gives zero
        $ACC = $ACC * ifelse($inds_orig == $inds_prime, 1, $(store.right))
    ), copy(store.redind))

    store.verbose>0 && @info "symbolic gradients extra..." prebody
    ex_prebody = commonsubex(quote $(prebody...) end)

    ex_pre = quote
        if iszero($ZED[$(store.leftraw...)])
            local $cnt = 0
            local $inds_prime = $inds_zero
            $loop_search
            if $cnt == 1
                local $ACC = one($TYP)
                $loop_accum
                let $inds_orig = $inds_prime
                    $ex_prebody
                end
            end # elseif more than one zero, then leave 𝛥x .== 0
            # continue # i.e. skip the ordinary routine, which divides
            @goto JUMP
        end
    end

    ex_post = quote
        @label JUMP
    end

    push!(store.notfree, cnt) # hack to disable @inbounds, avoids ERROR: syntax: misplaced label

    ex_pre, ex_post
end
=#

#========== symbolic differentiation ==========#

# This could probably use https://github.com/dfdx/XGrad.jl
# or https://github.com/SciML/ModelingToolkit.jl
# or https://github.com/JuliaMath/Calculus.jl/blob/master/src/differentiate.jl
# or now I found this: https://github.com/HarrisonGrodin/Simplify.jl
# but seemed simple enough to just write out, using rules from:
# http://www.juliadiff.org/DiffRules.jl/latest/

symbwalk(targets, store) = ex -> begin
        @capture_(ex, A_[inds__]) && A isa Symbol || return ex
        A in store.nograd && return ex
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
    ex.head == :call || throw("expected a functionn call, got $ex.")
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
    elseif length(ex.args) == 4  # three-arg function such as ifelse
        fx, fy, fz = mydiffrule(fun, ex.args[2:end]...)
        dx = leibnitz(ex.args[2], target)
        dy = leibnitz(ex.args[3], target)
        dz = leibnitz(ex.args[4], target)
        return simpliplus(simplitimes(fx, dx), simplitimes(fy, dy), simplitimes(fz, dz))
    end
    throw("don't know how to handle $ex.")
end

simplitimes(x::Number, y::Number) = x*y
simplitimes(x::Number, y) = x==0 ? 0 : x==1 ? y : x==-1 ? :(-$y) : :($x * $y)
simplitimes(x, y::Number) = y==0 ? 0 : y==1 ? x : y==-1 ? :(-$x) : :($y * $x)
simplitimes(x, y) = :($y * $x)
simplitimes(x, y, zs...) = simplitimes(simplitimes(x, y), zs...)

simpliplus(x::Number, y::Number) = x + y
simpliplus(x::Number, y) = x==0 ? y : :($x + $y)
simpliplus(x, y::Number) = y==0 ? x : :($x + $y)
simpliplus(x, y) = :($x + $y)
simpliplus(x, y, zs...) = simpliplus(simpliplus(x, y), zs...)

simpliconj(x::Number) = conj(x)
simpliconj(x) = :(conj($x))

mydiffrule(f, xs...) = begin
    f == :+ && return map(_->1, xs)
    f == :- && return length(xs)==1 ? -1 : (1,-1)
    f == :^ && return mypowrule(xs...)
    f == :/ && return mydivrule(xs...)
    f == :// && return mydivrule(xs...)
    f == :inv && return mydivrule(1, xs...)[2]
    f == :log && return simpliinv(xs...)
    f == :abs && return myabsrule(xs...)
    f == :sqrt && return mysqrtrule(xs...)
    f == :relu && return myrelurule(xs...)
    f in BASE_NOGRAD && return map(_->0, xs)
    DiffRules.hasdiffrule(:Base, f, length(xs)) &&
        return DiffRules.diffrule(:Base, f, xs...)
    DiffRules.hasdiffrule(:SpecialFunctions, f, length(xs)) &&
        return DiffRules.diffrule(:SpecialFunctions, f, xs...)
    throw("no diffrule found for function $f($(join(map(_->"_",xs),", "))).")
end

BASE_NOGRAD = [:(==), :(!=), :(<), :(<=), :(>), :(>=), :trunc, :round]

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

myrelurule(x::Number) = x>0 ? 1 : 0
myrelurule(x) = :(ifelse($x>0, 1, 0))

myabsrule(x::Number) = x<0 ? -1 : 1
myabsrule(x) = :(ifelse($x<0, -1, 1)) # matches DiffRules._abs_deriv, which uses signbit(x)

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

@printgrad  exp(2x)   x
@printgrad  exp(x/y)   x y
@printgrad  exp((x-y)^2/2)   x y

@printgrad  exp(x) * y   x y
@printgrad  exp(x) / 2y   x y

@printgrad a * b / sqrt(d * e)  a b d e
@printgrad x * z / sqrt(y * z)  x y z

=#


#========== the end ==========#
