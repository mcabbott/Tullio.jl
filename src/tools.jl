
#========== a mutable, typeless, almost-namedtuple ==========#

struct DotDict
    store::Dict{Symbol,Any}
end
DotDict(;kw...) = DotDict(Dict(pairs(kw)...))

Base.parent(x::DotDict) = getfield(x, :store)

Base.propertynames(x::DotDict) = Tuple(sort(collect(keys(parent(x)))))
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

    elseif pat.head == :call  && length(pat.args)==3 && pat.args[1] == :!= &&
        _endswithone(pat.args[2]) && _endswithone(pat.args[3]) # :( A_ != B_ )
        H = QuoteNode(pat.args[1])
        _symbolone(pat.args[2]), _symbolone(pat.args[3])

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
_trymatch(ex::Expr, pat::Val{:!=}) =
    if ex.head === :call && length(ex.args) == 3 && ex.args[1] == :!=
        ex.args[2], ex.args[3]
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

verbosetidy(expr) = MacroTools_postwalk(expr) do ex
        if ex isa Expr && ex.head == :block
            args = filter(x -> !(x isa LineNumberNode || x == nothing), ex.args)
            if length(args) == 1 && args[1] isa Expr && args[1].head == :block
                # disallow block(block(stuff))
                args[1]
            else
                Expr(ex.head, args...)
            end
        elseif ex isa Expr && ex.head == :macrocall && length(ex.args) >= 2
            # line number after macro name can't be dropped, but can be nothing:
            Expr(ex.head, ex.args[1], nothing, filter(x -> !(x isa LineNumberNode), ex.args[3:end])...)
        else
            ex
        end
    end

#========== the end ==========#
