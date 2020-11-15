
#========== adjusting index ranges, runtime ==========#

# This is to get the range of j in A[2j], from axes(A,1):

function divrange(r::AbstractUnitRange, f::Integer)
    if f > 0
        # a = div(first(r), f, RoundUp) # onnly 1.4 it seems?
        a = cld(first(r), f)
        # z = div(last(r), f, RoundDown)
        z = fld(last(r), f)
    elseif f < 0
        # a = div(last(r), f, RoundUp)
        a = cld(last(r), f)
        # z = div(first(r), f, RoundDown)
        z = fld(first(r), f)
    else
        throw("can't scale indices by zero")
    end
    a:z
end

#=
divrange(1:10, 2) .* 2
divrange(0:10, 2) .* 2
divrange(1:11, 2) .* 2
divrange(1:10, 3) .* 3

divrange(1:10, -1) .* -1 |> sort
divrange(1:10, -2) .* -2 |> sort
divrange(0:10, -2) .* -2 |> sort
divrange(0:11, -2) .* -2 |> sort
=#

# Special case of A[-i]:

function minusrange(r::AbstractRange)
    -last(r):-first(r)
end

#=
minusrange(1:11) == divrange(1:11, -1)
minusrange(1:10) == divrange(1:10, -1)
=#

# This is to get the range of j in A[j+k], given axes(A,1) and the range of k

function subranges(r::AbstractUnitRange, s::AbstractRange)
    first(r)-minimum(s) : last(r)-maximum(s)
end

function addranges(r::AbstractUnitRange, s::AbstractRange)
    first(r)+maximum(s) : last(r)+minimum(s)
end

#=
issubset(subranges(1:10, 1:3) .+ 1, 1:10)
issubset(subranges(1:10, 1:3) .+ 3, 1:10)

issubset(addranges(1:10, 1:3) .- 1, 1:10)
issubset(addranges(1:10, 1:3) .- 3, 1:10)
=#

# # This is to get range of j in A[j÷2], from axes(A,1):

function mulrange(r::AbstractUnitRange, f::Integer)
    first(r)*f : last(r)*f
end

function dotdiv(r::AbstractUnitRange, f::Integer)
    first(r)÷f : last(r)÷f
end

#=
mulrange(1:10, 2) .÷ 2 |> unique
mulrange(0:10, 2) .÷ 2 |> unique
mulrange(1:11, 2) .÷ 2 |> unique

mulrange(1:10, 3) .÷ 3 |> unique
mulrange(-10:-1, 3) .÷ 3 |> unique
mulrange(-11:-1, 3) .÷ 3 |> unique
mulrange(-11:-2, 3) .÷ 3 |> unique
mulrange(-12:-3, 3) .÷ 3 |> unique
=#

# This is for A[I[j]] (where this range must be a subset of axes(A,1))
# and for A[I[j]+k] (where it enters into the calculation of k's range).

function extremerange(A)
    α, ω = minimum(A), maximum(A)
    α isa Integer && ω isa Integer || throw("expected integers!")
    α:ω
end

# This gives the range of j implied by A[i, pad(j,3)]

function padrange(r::AbstractUnitRange, lo::Integer, hi::Integer)
    first(r)-lo : last(r)+hi
end

#========== functions used by the macro ==========#

@nospecialize

# This is for the bounds check on A[I[j],k]:

function extremeview(ex::Expr)
    @assert ex.head == :ref
    A = ex.args[1]
    if any(is_const, ex.args[2:end])
        ind = map(i -> is_const(i) ? i : (:), ex.args[2:end])
        :(@view $A[$(ind...)])
    else
        A
    end
end

"""
    range_expr_walk(:(axes(A,1)), :(2i+1)) -> range, :i

Given the axis of `A`, and the expression inside `A[2i+1]`,
this returns an expression for the resulting range of index `i`.
Understands operations `+, -, *, ÷`.
(Don't really need `÷`, as this results in a non-`UnitRange`
which can't be a valid index.)

If the expression is from something like `A[2i+j]`, then it returns a tuple of ranges
and a tuple of symbols. The range for `:j` contains `:$(AXIS)i` and v-v.

If the expression is from `A[I[j]]` then it returns `(min:max, nothing)`,
and the caller should check `issubset(min:max, axes(A,1))`.
"""
function range_expr_walk(r, ex::Expr, con=[])
    ex.head == :kw && return range_expr_kw(r, ex)
    if ex.head == :ref # case of M[I[j], k] with r=axes(M,1)
        A = ex.args[1]
        A = extremeview(ex)
        push!(con, :(minimum($A) in $r && maximum($A) in $r || throw("not safe!"))) # not used??
        return (:($extremerange($A)),nothing)
    end
    ex.head == :call || throw("not sure what to do with $ex")
    if ex.args[1] == :pad && length(ex.args) == 3
        _, a, p = ex.args
        return range_expr_walk(:($padrange($r, $p, $p)), a)
    elseif ex.args[1] == :pad && length(ex.args) == 4
        _, a, lo, hi = ex.args
        return range_expr_walk(:($padrange($r, $lo, $hi)), a)
    elseif length(ex.args) == 2
        op, a = ex.args
        if op == :+
            return range_expr_walk(r, a)
        elseif op == :-
            return range_expr_walk(:($minusrange($r)), a)
        end
    elseif length(ex.args) == 3
        op, a, b = ex.args
        if op == :+
            is_const(a) && return range_expr_walk(:($r .- $a), b)
            is_const(b) && return range_expr_walk(:($r .- $b), a)
            # with neither constant, first go outwards from index j to expression b...
            ax_a = range_unwrap(a)
            ax_b = range_unwrap(b)
            #... then use that with given size(A,d) to constrain range of i, and v-v:
            range_a, i_a = range_expr_walk(:($subranges($r, $ax_b)), a)
            range_b, i_b = range_expr_walk(:($subranges($r, $ax_a)), b)
            return (range_a, range_b), (i_a, i_b)

        elseif op == :-
            is_const(a) && return range_expr_walk(:($minusrange($r .- $a)), b)
            is_const(b) && return range_expr_walk(:($r .+ $b), a)
            ax_a = range_unwrap(a)
            ax_b = range_unwrap(b)
            range_a, i_a = range_expr_walk(:($addranges($r, $ax_b)), a)
            range_b, i_b = range_expr_walk(:($minusrange($subranges($r, $ax_a))), b)
            return (range_a, range_b), (i_a, i_b)

        elseif op == :*
            is_const(a) && return range_expr_walk(:($divrange($r, $a)), b)
            is_const(b) && return range_expr_walk(:($divrange($r, $b)), a)
        elseif op == :÷
            is_const(b) && return range_expr_walk(:($mulrange($r, $b)), a)
        elseif op == :/
            throw("not sure what to do with $ex, perhaps you wanted ÷")
        end
    elseif length(ex.args) > 3
        op, a, b, c = ex.args[1:4]
        ds = ex.args[5:end]
        if op == :+
            is_const(a) && return range_expr_walk(:($r .- $a), :(+($b, $c, $(ds...))))
            is_const(b) && return range_expr_walk(:($r .- $b), :(+($a, $c, $(ds...))))
            is_const(c) && return range_expr_walk(:($r .- $c), :(+($a, $b, $(ds...))))
        end
    end
    throw("not sure what to do with $ex, sorry")
end

range_expr_walk(range, s::Symbol) = range, s
range_expr_walk(range, n::Integer) = range, nothing

is_const(::Int) = true
is_const(::Any) = false
is_const(s::Symbol) = s in [:(:), :begin, :end] # : for the purposes of saveconstraints setting :intersect
is_const(ex::Expr) = begin
    ex.head == :$ && return true # what's returned by range_expr_walk will still contain $
    if ex.head == :call && ex.args[1] in (:+, :-, :*, :÷)
        return all(is_const, ex.args[2:end])
    end
    false
end

"""
    range_expr_walk(:(axes(A,1)), :(i=j)) -> :(axes(A, :i)), :j

Special case for keyword indexing, `A[i=j, k=j+2]` comes here.
"""
function range_expr_kw(r::Expr, ex::Expr)
    @assert ex.head == :kw
    # @assert r.head == :call && r.args[1] == :axes
    r.args[3] = QuoteNode(ex.args[1])
    range_expr_walk(r, ex.args[2])
end

"""
    range_unwrap(:(2i+1)) -> :(2 .* AXIS_i .+ 1)

This goes in the opposite direction to `range_expr_walk`, and gives
the range of values taken by the expression, in terms of `Symbol($AXIS, i)`.
"""
range_unwrap(i::Symbol) = Symbol(AXIS, i)
range_unwrap(ex::Expr) = begin
    if ex.head == :ref # case of A[I[j]+k] comes here
        A = ex.args[1]
        return :($extremerange($A))
    end
    ex.head == :call || throw("don't know how to handle $ex")
    # if ex.args[1] == :pad or :clamp or :mod --- find test cases.
    if length(ex.args) == 2
        op, a = ex.args
        if op == :-
            return :($minusrange($(range_unwrap(a))))
        end
    elseif length(ex.args) == 3
        op, a, b = ex.args
        if op == :*
            a == -1 && return :($minusrange($(range_unwrap(b))))
            b == -1 && return :($minusrange($(range_unwrap(a))))
            is_const(a) && return :($a .* $(range_unwrap(b)))
            is_const(b) && return :($b .* $(range_unwrap(a)))
        elseif op == :+
            is_const(a) && return :($a .+ $(range_unwrap(b)))
            is_const(b) && return :($b .+ $(range_unwrap(a)))
        elseif op == :-
            is_const(a) && return :($a .- $(range_unwrap(b)))
            is_const(b) && return :($(range_unwrap(a)) .- $b)
        elseif op == :÷
            is_const(b) && return :($dotdiv($(range_unwrap(a)), $b))
        end
    end
    throw("don't know how to handle $ex, sorry")
end

"""
    range_expr_walk(nothing, :(i+2)) -> nothing, :j

Special case used for `A[mod(i+2)]` etc.
"""
range_expr_walk(r::Nothing, s::Symbol) = r, s
range_expr_walk(r::Nothing, n::Integer) = r, nothing
function range_expr_walk(r::Nothing, ex::Expr)
    is_const(ex) && return r, nothing
    ex.head == :ref && return r, nothing
    ex.head == :call || throw("not sure what to do with $ex")
    # if ex.args[1] in [:+, :-, :*, :÷]
        syms = []
        for x in ex.args[2:end]
            _, i = range_expr_walk(r, x)
            i isa Tuple && push!(syms, i...)
            i isa Symbol && push!(syms, i)
        end
        if isempty(syms)
            return r, nothing
        elseif length(syms) == 1
            return r, syms[1]
        else
            return r, Tuple(syms)
        end
    # end
end

@specialize

#========== the end ==========#
