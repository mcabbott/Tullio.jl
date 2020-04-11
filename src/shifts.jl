
# Given A[i+1], the range of i is axes(A,1) .- 1
#=
:(i + 1) |> dump
:(i - 1) |> dump
:(1 + i) |> dump
:(-1 + i) |> dump

subrange(r::AbstractRange, o::Integer) = r .- o

# Given A[2i], we want axes(A,1)÷2? No not quite.

:(2*i) |> dump
:(-i) |> dump
:(-1 * i) |> dump
=#
function divrange(r::AbstractUnitRange, f::Integer)
    if f > 0
        a = div(first(r), f, RoundUp)
        z = div(last(r), f, RoundDown)
    elseif f < 0
        a = div(last(r), f, RoundUp)
        z = div(first(r), f, RoundDown)
    else
        error("invalid factor $f")
    end
    a:z
end
function divrange_minus(r::AbstractRange)
    -last(r):-first(r)
end
divrange_minus(1:11) == divrange(1:11, -1)
divrange_minus(1:10) == divrange(1:10, -1)
#=
divrange(1:10, 2) .* 2
divrange(0:10, 2) .* 2
divrange(1:11, 2) .* 2
divrange(1:10, 3) .* 3

divrange(1:10, -1) .* -1 |> sort
divrange(1:10, -2) .* -2 |> sort
divrange(0:10, -2) .* -2 |> sort
divrange(0:11, -2) .* -2 |> sort

# Given A[2i+3], the range of i is divrange(subrange(axes(A,1), 3), 2)

:(2i+3) |> dump

:($(Expr(:$, :c))) |> dump
=#
"""
    range_expr_walk(:(axes(A,1)), :(2i+1)) -> range, :i

Given the axis of `A`, and the expression inside `A[2i+1]`,
this returns an expression for the resulting range of index `i`.
Understands operations `+, -, *, ÷`.
(Don't really need `÷`, as this results in a non-`UnitRange`
which can't be a valid index.)
"""
function range_expr_walk(r::Expr, ex::Expr)
    ex.head == :kw && return range_expr_kw(r, ex)
    ex.head == :call || error("not sure what to do with $ex")
    if length(ex.args) == 2
        op, a = ex.args
        if op == :+
            return range_expr_walk(r, a)
        elseif op == :-
            return range_expr_walk(:(divrange_minus($r)), a)
            # return range_expr_walk(:(divrange($r, -1)), a)
        end
    elseif length(ex.args) == 3
        op, a, b = ex.args
        if op == :+
            isconst(a) && return range_expr_walk(:($r .- $a), b)
            isconst(b) && return range_expr_walk(:($r .- $b), a)
        elseif op == :*
            isconst(a) && return range_expr_walk(:($divrange($r, $a)), b)
            isconst(b) && return range_expr_walk(:($divrange($r, $b)), a)
        elseif op == :-
            isconst(a) && return range_expr_walk(:($divrange_minus($r .- $a)), b)
            # isconst(a) && return range_expr_walk(:($divrange($r .- $a, -1)), b)
            isconst(b) && return range_expr_walk(:($r .+ $b), a)
        elseif op == :÷
            isconst(b) && return range_expr_walk(:($r .* $b), a)
        elseif op == :/
            error("not sure what to do with $ex, perhaps you wanted ÷")
        end
    elseif length(ex.args) > 3
        op, a, b = ex.args
        cs = ex.args[4:end]
        if op == :+
            isconst(a) && return range_expr_walk(:($r .- $a), :(+($b, $(cs...))))
            isconst(b) && return range_expr_walk(:($r .- $b), :(+($a, $(cs...))))
        end
    end
    error("not sure what to do with $ex")
end

range_expr_walk(range::Expr, s::Symbol) = range, s

isconst(::Int) = true
isconst(::Any) = false
isconst(ex::Expr) = ex.head == :$
isconst(s::Symbol) = s === :(:) # for the purposes of saveconstraints setting :intersect

"""
    range_expr_walk(:(axes(A,1)), :(i=j)) -> :(axes(A, :i)), :j

Special case for keyword indexing, `A[i=j, k=j+2]` comes here.
"""
function range_expr_kw(r::Expr, ex::Expr)
    @assert ex.head == :kw
    @assert r.head == :call && r.args[1] == :axes
    r.args[3] = QuoteNode(ex.args[1])
    range_expr_walk(r, ex.args[2])
end
