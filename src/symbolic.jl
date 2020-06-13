
#========== backward gradient using symbolic derivatives ==========#

using DiffRules

function insert_symbolic_gradient(store)

    dZ = Symbol(DEL, ZED)
    ∇act! = Symbol(:∇, ACT!)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    # gradscalars = map(A -> Symbol(DEL, A), store.scalars)

    out_ind, in_ind = if store.redfun == :+
        nonshared = setdiff(vcat(store.leftind, store.redind), store.sharedind)
        store.sharedind, nonshared
    elseif store.redfun == :*
        store.leftind, store.redind
    else
        error("can't take gradients with reduction $(store.redfun) (but max/min would not be hard to add)")
    end
    axislist = map(i -> Symbol(AXIS, i), vcat(out_ind, in_ind))

    targets = []
    MacroTools_postwalk(symbwalk(targets), store.right)
    # append!(targets, scalars)

    inbody, prebody = [], []
    for (dt, t) in unique(targets)
        drdt = leibnitz(store.right, t)
        deltar = simplitimes(drdt, :(conj($dZ[$(store.leftraw...)])))
        if store.redfun == :+
            push!(inbody, :($dt = $dt + conj($deltar)))
        elseif store.redfun == :*
            push!(inbody, :($dt = conj($deltar) * $ZED[$(store.leftraw...)] * inv($(store.right))))
            push!(prebody, :($dt = conj($deltar) * $ACC))
        end
    end
    store.verbose>0 && @info "symbolic gradients" inbody
    ex_body = commonsubex(quote $(inbody...) end)

    ex_pre, ex_post = if store.redfun == :* # then nonzero LHS are handled already, but harder cases here:
        product_grad(prebody, store)
    else
        nothing, nothing
    end

    make_many_actors(∇act!,
        vcat(gradarrays, :($dZ::AbstractArray{$TYP}), ZED, store.arrays, store.scalars, axislist),
        # vcat(gradarrays, gradscalars, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        nothing, out_ind, ex_pre, in_ind, ex_body, ex_post, store, " (symbolic gradient)")

    if isdefined(store.mod, :Zygote) # special case for FillArrays
        ex_body2 = fillarrayreplace(ex_body, dZ)
        ex_pre2 = fillarrayreplace(ex_pre, dZ)
        ex_value = :($(Symbol(dZ, :_value)) = $dZ.value) # @avx likes this outside the loop

        make_many_actors(∇act!,
            vcat(gradarrays, :($dZ::Zygote.Fill{$TYP}), ZED, store.arrays, store.scalars, axislist),
            ex_value, out_ind, ex_pre2, in_ind, ex_body2, ex_post, store, " (method for FillArrays)")
    end

end


#=
Consider @tullio (*) Z[i] := A[i,j] + B[i]

When Z[i] != 0, then every factor was nonzero, and so we want
    ΔA[i,j] = delta * lhs / rhs * leibnitz(rhs, A[i,j]) = ΔZ[i] * Z[i] / (A[i,j] + B[i]) * 1
    ΔB[j] = delta * lhs / rhs * leibnitz(rhs, B[i,j])   = ΔZ[i] * Z[i] / (A[i,j] + B[i]) * 1
notice the common factor.

When Z[i] == 0, then most stay zero, only when rhs=0 we can't divide so must do this:

    ΔA[i,j] = delta * prod_rest * leibnitz(rhs, A[i,j])
    ΔB[j] = delta * prod_rest * leibnitz(rhs, B[j])

Ideally you could branch before doing the inner loops.
You will need knowledge of the LHS, not kept at the moment.


for i in 1:10 # outer loop(s)

    if Z[i] == 0 # code handled by this function?
        cnt = 0
        j0 = 0 # ...
        for j in 1:10 # loop to look for zeros
            if A[i,j] + B[i] == 0
                cnt += 1
                j0 = j # ...
            end
        end
        if cnt == 1 # if exactly one zero
            rest = 1.0
            for j in 1:10 # ...
                rest = rest * ifelse(j==j0 ? 1 : A[i,j] + B[i])
            end
            let j=j0 # ...
                # run CSE on these -- "prebody"
                ΔA[i,j] = ΔZ[i] * rest * 1
                ΔB[j] = ΔZ[i] * rest * 1
            end
        end
        continue
    end

    # easy case, divide
    for j in 1:10
        # run CSE on these -- "inbody" as before
        ΔA[i,j] = ΔZ[i] * Z[i] * inv(A[i,j] + B[i]) * 1
        ΔB[j] = ΔZ[i] * Z[i] * inv(A[i,j] + B[i]) * 1
    end

end

Wait, shared/nonshared isn't the right division anymore.
It's left/redind that you care about.

=#
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

#========== symbolic differentiation ==========#

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

@printgrad  exp(2x)   x
@printgrad  exp(x/y)   x y
@printgrad  exp((x-y)^2/2)   x y

@printgrad  exp(x) * y   x y
@printgrad  exp(x) / 2y   x y

@printgrad a * b / sqrt(d * e)  a b d e
@printgrad x * z / sqrt(y * z)  x y z

=#


#========== the end ==========#
