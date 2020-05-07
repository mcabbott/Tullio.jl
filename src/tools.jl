
#========== a mutable, typeless, almost-namedtuple ==========#

struct DotDict
    store::Dict{Symbol,Any}
end
DotDict(;kw...) = DotDict(Dict(pairs(kw)...))

Base.parent(x::DotDict) = getfield(x, :store)

Base.propertynames(x::DotDict) = Tuple(keys(parent(x)))
Base.getproperty(x::DotDict, s::Symbol) = getindex(parent(x), s)
function Base.setproperty!(x::DotDict, s::Symbol, v)
    s in propertynames(x) || error("DotDict has no field $s")
    T = typeof(getproperty(x, s))
    if T == Nothing
        setindex!(parent(x), v, s)
    else
        setindex!(parent(x), convert(T, v), s)
    end
end

function Base.show(io::IO, x::DotDict)
    print(io, "DotDict(")
    strs = map(k -> string(k, " = ", getproperty(x, k)), propertynames(x))
    print(io, join(strs, ", "), ")")
end

#========== capture macro ==========#
# My faster, more limited, version:

"""
    @capture_(ex, A_[ijk__])

Faster drop-in replacement for `MacroTools.@capture`, for a few patterns only:
* `A_[ijk__]` and `A_{ijk__}`
* `[ijk__]`
* `A_.field_`
* `A_ := B_` and  `A_ = B_` and `A_ += B_` etc.
* `f_(x_)`
"""
macro capture_(ex, pat::Expr)

    H = QuoteNode(pat.head)

    A,B = if pat.head in [:ref, :curly] && length(pat.args)==2 &&
        _endswithone(pat.args[1]) && _endswithtwo(pat.args[2]) # :( A_[ijk__] )
        _symbolone(pat.args[1]), _symboltwo(pat.args[2])

    elseif pat.head == :. && pat.args[2] isa QuoteNode &&
        _endswithone(pat.args[1]) && _endswithone(pat.args[2].value) # :( A_.field_ )
        _symbolone(pat.args[1]), _symbolone(pat.args[2].value)

    elseif pat.head == :call  && length(pat.args)==2 &&
        _endswithone(pat.args[1]) && _endswithone(pat.args[2]) # :( f_(x_) )
        _symbolone(pat.args[1]), _symbolone(pat.args[2])

    elseif pat.head in [:call, :(=), :(:=), :+=, :-=, :*=, :/=] &&
        _endswithone(pat.args[1]) && _endswithone(pat.args[2]) # :( A_ += B_ )
        _symbolone(pat.args[1]), _symbolone(pat.args[2])

    elseif pat.head == :vect && _endswithtwo(pat.args[1]) # :( [ijk__] )
        _symboltwo(pat.args[1]), gensym(:ignore)

    else
        error("@capture_ doesn't work on pattern $pat")
    end

    @gensym res
    quote
        $A, $B = nothing, nothing
        $res = $_trymatch($ex, Val($H))
        if $res === nothing
            false
        else
            $A, $B = $res
            true
        end
    end |> esc
end

_endswithone(ex) = endswith(string(ex), '_') && !_endswithtwo(ex)
_endswithtwo(ex) = endswith(string(ex), "__")

_symbolone(ex) = Symbol(string(ex)[1:end-1])
_symboltwo(ex) = Symbol(string(ex)[1:end-2])

_getvalue(::Val{val}) where {val} = val

_trymatch(s, v) = nothing # Symbol, or other Expr
_trymatch(ex::Expr, pat::Union{Val{:ref}, Val{:curly}}) = # A_[ijk__] or A_{ijk__}
    if ex.head === _getvalue(pat)
        ex.args[1], ex.args[2:end]
    else
        nothing
    end
_trymatch(ex::Expr, ::Val{:.}) = # A_.field_
    if ex.head === :. && ex.args[2] isa QuoteNode
        ex.args[1], ex.args[2].value
    else
        nothing
    end
_trymatch(ex::Expr, pat::Val{:call}) =
    if ex.head === _getvalue(pat) && length(ex.args) == 2
        ex.args[1], ex.args[2]
    else
        nothing
    end
_trymatch(ex::Expr, pat::Union{Val{:(=)}, Val{:(:=)}, Val{:(+=)}, Val{:(-=)}, Val{:(*=)}, Val{:(/=)}}) =
    if ex.head === _getvalue(pat)
        ex.args[1], ex.args[2]
    else
        nothing
    end
_trymatch(ex::Expr, ::Val{:vect}) = # [ijk__]
    if ex.head === :vect
        ex.args, nothing
    else
        nothing
    end


# Cases for Tullio:
# @capture(ex, B_[inds__].field_) --> @capture_(ex, Binds_.field_) && @capture_(Binds, B_[inds__])


#========== postwalk ==========#
# Copied verbatim from here:
# https://github.com/MikeInnes/MacroTools.jl/blob/master/src/utils.jl

walk(x, inner, outer) = outer(x)
walk(x::Expr, inner, outer) = outer(Expr(x.head, map(inner, x.args)...))

"""
    postwalk(f, expr)
Applies `f` to each node in the given expression tree, returning the result.
`f` sees expressions *after* they have been transformed by the walk. See also
`prewalk`.
"""
postwalk(f, x) = walk(x, x -> postwalk(f, x), f)

"""
    prewalk(f, expr)
Applies `f` to each node in the given expression tree, returning the result.
`f` sees expressions *before* they have been transformed by the walk, and the
walk will be applied to whatever `f` returns.
This makes `prewalk` somewhat prone to infinite loops; you probably want to try
`postwalk` first.
"""
prewalk(f, x)  = walk(f(x), x -> prewalk(f, x), identity)

replace(ex, s, s′) = prewalk(x -> x == s ? s′ : x, ex)

const MacroTools_prewalk = prewalk
const MacroTools_postwalk = postwalk

#========== prettify ==========#

"""
    isexpr(x, ts...)
Convenient way to test the type of a Julia expression.
Expression heads and types are supported, so for example
you can call
    isexpr(expr, String, :string)
to pick up on all string-like expressions.
"""
isexpr(x::Expr) = true
isexpr(x) = false
isexpr(x::Expr, ts...) = x.head in ts
isexpr(x, ts...) = any(T->isa(T, Type) && isa(x, T), ts)

isline(ex) = isexpr(ex, :line) || isa(ex, LineNumberNode)

iscall(ex, f) = isexpr(ex, :call) && ex.args[1] == f

"""
    rmlines(x)
Remove the line nodes from a block or array of expressions.
Compare `quote end` vs `rmlines(quote end)`
### Examples
To work with nested blocks:
```julia
prewalk(rmlines, ex)
```
"""
rmlines(x) = x
function rmlines(x::Expr)
  # Do not strip the first argument to a macrocall, which is
  # required.
  if x.head == :macrocall && length(x.args) >= 2
    Expr(x.head, x.args[1], nothing, filter(x->!isline(x), x.args[3:end])...)
  else
    Expr(x.head, filter(x->!isline(x), x.args)...)
  end
end

striplines(ex) = prewalk(rmlines, ex)

function flatten1(ex)
  isexpr(ex, :block) || return ex
  #ex′ = :(;)
  ex′ = Expr(:block)
  for x in ex.args
    isexpr(x, :block) ? append!(ex′.args, x.args) : push!(ex′.args, x)
  end
  # Don't use `unblock` to preserve line nos
  return length(ex′.args) == 1 ? ex′.args[1] : ex′
end

"""
    flatten(ex)
Flatten any redundant blocks into a single block, over the whole expression.
"""
flatten(ex) = postwalk(flatten1, ex)

unresolve1(x) = x
unresolve1(f::Function) = methods(f).mt.name

unresolve(ex) = prewalk(unresolve1, ex)

"""
    prettify(ex)
Makes generated code generaly nicer to look at.
"""
prettify(ex; lines = false) =
  ex |> (lines ? identity : striplines) |> flatten |> unresolve # |> resyntax # |> alias_gensyms

const MacroTools_prettify = prettify

#========== the end ==========#
