const __bodyfunction__ = Dict{Method,Any}()

# Find keyword "body functions" (the function that contains the body
# as written by the developer, called after all missing keyword-arguments
# have been assigned values), in a manner that doesn't depend on
# gensymmed names.
# `mnokw` is the method that gets called when you invoke it without
# supplying any keywords.
function __lookup_kwbody__(mnokw::Method)
    function getsym(arg)
        isa(arg, Symbol) && return arg
        @assert isa(arg, GlobalRef)
        return arg.name
    end

    f = get(__bodyfunction__, mnokw, nothing)
    if f === nothing
        fmod = mnokw.module
        # The lowered code for `mnokw` should look like
        #   %1 = mkw(kwvalues..., #self#, args...)
        #        return %1
        # where `mkw` is the name of the "active" keyword body-function.
        ast = Base.uncompressed_ast(mnokw)
        if isa(ast, Core.CodeInfo) && length(ast.code) >= 2
            callexpr = ast.code[end-1]
            if isa(callexpr, Expr) && callexpr.head == :call
                fsym = callexpr.args[1]
                if isa(fsym, Symbol)
                    f = getfield(fmod, fsym)
                elseif isa(fsym, GlobalRef)
                    if fsym.mod === Core && fsym.name === :_apply
                        f = getfield(mnokw.module, getsym(callexpr.args[2]))
                    elseif fsym.mod === Core && fsym.name === :_apply_iterate
                        f = getfield(mnokw.module, getsym(callexpr.args[3]))
                    else
                        f = getfield(fsym.mod, fsym.name)
                    end
                else
                    f = missing
                end
            else
                f = missing
            end
        else
            f = missing
        end
        __bodyfunction__[mnokw] = f
    end
    return f
end

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(Base.cat)),NamedTuple{(:dims,), Tuple{Val{1}}},typeof(cat),Expr,Vararg{Any, N} where N})
    Base.precompile(Tuple{Core.kwftype(typeof(Base.cat)),NamedTuple{(:dims,), Tuple{Val{1}}},typeof(cat),Vector{Symbol},Vararg{Any, N} where N})
    Base.precompile(Tuple{Core.kwftype(typeof(Base.cat_t)),NamedTuple{(:dims,), Tuple{Val{1}}},typeof(Base.cat_t),Type{Any},Expr,Vararg{Any, N} where N})
    Base.precompile(Tuple{Core.kwftype(typeof(Base.cat_t)),NamedTuple{(:dims,), Tuple{Val{1}}},typeof(Base.cat_t),Type{Any},Vector{Symbol},Vararg{Any, N} where N})
    Base.precompile(Tuple{Type{Dict{Symbol, Any}},NTuple{37, Pair{Symbol, Any}}})
    Base.precompile(Tuple{Type{Dict{Symbol, Any}},Pair{Symbol, Symbol},Vararg{Pair, N} where N})
    Base.precompile(Tuple{Type{Dict},Pair{Symbol, Any},Vararg{Pair{Symbol, Any}, N} where N})
    Base.precompile(Tuple{Type{Pair},Int64,Expr})
    Base.precompile(Tuple{Type{Pair},Int64,Symbol})
    Base.precompile(Tuple{typeof(==),Bool,Symbol})
    Base.precompile(Tuple{typeof(==),Symbol,Bool})
    Base.precompile(Tuple{typeof(>),Bool,Int64})
    Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),Function,Int64,Base.OneTo{Int64}})
    Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),Function,Tuple{Expr, Expr}})
    Base.precompile(Tuple{typeof(Base.__cat),Vector{Any},Tuple{Int64},Tuple{Bool},Expr,Vararg{Any, N} where N})
    Base.precompile(Tuple{typeof(Base.__cat),Vector{Any},Tuple{Int64},Tuple{Bool},Vector{Symbol},Vararg{Any, N} where N})
    Base.precompile(Tuple{typeof(Base._any),Base.Fix2{typeof(==), Symbol},Vector{Symbol},Colon})
    Base.precompile(Tuple{typeof(Base._array_for),Type{Expr},Base.Iterators.Enumerate{Vector{Any}},Base.HasShape{1}})
    Base.precompile(Tuple{typeof(Base._array_for),Type{Expr},Base.Iterators.Enumerate{Vector{Expr}},Base.HasShape{1}})
    Base.precompile(Tuple{typeof(Base._array_for),Type{LineNumberNode},Vector{Any},Base.HasShape{1}})
    Base.precompile(Tuple{typeof(Base._array_for),Type{Symbol},Base.Iterators.Enumerate{Vector{Any}},Base.HasShape{1}})
    Base.precompile(Tuple{typeof(Base._cat),Val{1},Vector{Symbol},Vararg{Any, N} where N})
    Base.precompile(Tuple{typeof(Base._nt_names),Type{NamedTuple{(:redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd), Tuple{Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}}}}})
    Base.precompile(Tuple{typeof(Base._shrink),Function,Vector{Symbol},Tuple{Vector{Symbol}}})
    Base.precompile(Tuple{typeof(Base.cat_indices),Symbol,Int64})
    Base.precompile(Tuple{typeof(Base.cat_similar),Expr,Type,Tuple{Int64}})
    Base.precompile(Tuple{typeof(Base.cat_similar),Vector{Symbol},Type,Tuple{Int64}})
    Base.precompile(Tuple{typeof(Base.cat_size),Symbol,Int64})
    Base.precompile(Tuple{typeof(Base.indexed_iterate),Tuple{Nothing, Nothing},Int64})
    Base.precompile(Tuple{typeof(Base.indexed_iterate),Tuple{Symbol, Symbol, Expr, Expr},Int64})
    Base.precompile(Tuple{typeof(Base.promote_eltypeof),Expr,Vector{Symbol},Vararg{Vector{Symbol}, N} where N})
    Base.precompile(Tuple{typeof(Base.promote_eltypeof),Symbol,Vector{Symbol},Vararg{Vector{Symbol}, N} where N})
    Base.precompile(Tuple{typeof(Base.promote_eltypeof),Vector{Symbol},Expr,Vararg{Any, N} where N})
    Base.precompile(Tuple{typeof(Base.promote_eltypeof),Vector{Symbol}})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Expr},LineNumberNode,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Expr},Symbol,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{LineNumberNode},Expr,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Nothing},LineNumberNode,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Symbol},Expr,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Symbol},Int64,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Symbol},QuoteNode,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Union{Nothing, LineNumberNode}},Expr,Int64})
    Base.precompile(Tuple{typeof(Base.vectorfilter),Function,Vector{Symbol}})
    Base.precompile(Tuple{typeof(Core.Compiler.eltype),Type{Vector{Base.HasShape{1}}}})
    Base.precompile(Tuple{typeof(allunique),Vector{Symbol}})
    Base.precompile(Tuple{typeof(any),Function,Vector{Symbol}})
    Base.precompile(Tuple{typeof(append!),Vector{Expr},Vector{Expr}})
    Base.precompile(Tuple{typeof(append!),Vector{Symbol},Vector{Symbol}})
    Base.precompile(Tuple{typeof(Base.cat_shape),Tuple{Bool},NTuple{4, Tuple{Int64}}})
    Base.precompile(Tuple{typeof(Base.cat_shape),Tuple{Bool},NTuple{6, Tuple{Int64}}})
    Base.precompile(Tuple{typeof(collect),Tuple{Symbol, Symbol}})
    Base.precompile(Tuple{typeof(convert),Type{Vector{Any}},Vector{Expr}})
    Base.precompile(Tuple{typeof(convert),Type{Vector{Any}},Vector{Symbol}})
    Base.precompile(Tuple{typeof(convert),Type{Vector{Symbol}},Vector{Expr}})
    Base.precompile(Tuple{typeof(enumerate),Vector{Expr}})
    Base.precompile(Tuple{typeof(enumerate),Vector{Symbol}})
    Base.precompile(Tuple{typeof(get!),Dict{Symbol, Vector{T} where T},Symbol,Vector{Any}})
    Base.precompile(Tuple{typeof(getindex),Dict{Symbol, Vector{T} where T},Symbol})
    Base.precompile(Tuple{typeof(hash),Pair{Int64, Expr},UInt64})
    Base.precompile(Tuple{typeof(hash),Pair{Int64, Int64},UInt64})
    Base.precompile(Tuple{typeof(hash),Pair{Int64, String},UInt64})
    Base.precompile(Tuple{typeof(hash),Pair{Int64, Symbol},UInt64})
    Base.precompile(Tuple{typeof(haskey),Dict{Symbol, Any},Symbol})
    Base.precompile(Tuple{typeof(haskey),Dict{Symbol, Vector{T} where T},Symbol})
    Base.precompile(Tuple{typeof(in),Symbol,Set{Any}})
    Base.precompile(Tuple{typeof(in),Tuple{Expr, Expr},Set{Any}})
    Base.precompile(Tuple{typeof(intersect),Vector{Symbol},Vector{Symbol}})
    Base.precompile(Tuple{typeof(isassigned),Vector{Symbol}})
    Base.precompile(Tuple{typeof(isequal),Expr})
    Base.precompile(Tuple{typeof(isequal),Int64})
    Base.precompile(Tuple{typeof(isequal),String})
    Base.precompile(Tuple{typeof(isequal),Symbol})
    Base.precompile(Tuple{typeof(iterate),Base.Iterators.Pairs{Symbol, Any, NTuple{37, Symbol}, NamedTuple{(:mod, :redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd, :flags, :redind, :init, :leftraw, :leftind, :leftarray, :leftscalar, :leftnames, :right, :finaliser, :rightind, :sharedind, :unsafeleft, :unsaferight, :arrays, :scalars, :cost, :constraints, :pairconstraints, :notfree, :shiftedind, :axisdefs, :padmodclamp, :outpre, :outex), Tuple{Module, Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}, Set{Symbol}, Vector{Symbol}, Nothing, Vector{Any}, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Int64, Dict{Symbol, Vector{T} where T}, Vector{Tuple}, Vector{Symbol}, Vector{Symbol}, Vector{Expr}, Bool, Vector{Expr}, Vector{Expr}}}},Int64})
    Base.precompile(Tuple{typeof(iterate),Base.Iterators.Pairs{Symbol, Any, NTuple{37, Symbol}, NamedTuple{(:mod, :redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd, :flags, :redind, :init, :leftraw, :leftind, :leftarray, :leftscalar, :leftnames, :right, :finaliser, :rightind, :sharedind, :unsafeleft, :unsaferight, :arrays, :scalars, :cost, :constraints, :pairconstraints, :notfree, :shiftedind, :axisdefs, :padmodclamp, :outpre, :outex), Tuple{Module, Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}, Set{Symbol}, Vector{Symbol}, Nothing, Vector{Any}, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Int64, Dict{Symbol, Vector{T} where T}, Vector{Tuple}, Vector{Symbol}, Vector{Symbol}, Vector{Expr}, Bool, Vector{Expr}, Vector{Expr}}}}})
    Base.precompile(Tuple{typeof(map),Function,Base.Iterators.Enumerate{Vector{Any}}})
    Base.precompile(Tuple{typeof(map),Function,Base.Iterators.Enumerate{Vector{Expr}}})
    Base.precompile(Tuple{typeof(map),Function,Base.Iterators.Enumerate{Vector{Symbol}}})
    Base.precompile(Tuple{typeof(map),Function,Vector{Any},Vector{Symbol}})
    Base.precompile(Tuple{typeof(map),Function,Vector{Expr}})
    Base.precompile(Tuple{typeof(map),Function,Vector{Symbol},Vector{Symbol}})
    Base.precompile(Tuple{typeof(map),Function,Vector{Symbol}})
    Base.precompile(Tuple{typeof(map),typeof(Base.cat_size),Tuple{Expr, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}}})
    Base.precompile(Tuple{typeof(map),typeof(Base.cat_size),Tuple{Vector{Symbol}, Expr, Symbol, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}}})
    Base.precompile(Tuple{typeof(merge),NamedTuple{(:mod, :redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd), Tuple{Module, Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}}},NamedTuple{(:flags, :redind, :init, :leftraw, :leftind, :leftarray, :leftscalar, :leftnames, :right, :finaliser, :rightind, :sharedind, :unsafeleft, :unsaferight, :arrays, :scalars, :cost, :constraints, :pairconstraints, :notfree, :shiftedind, :axisdefs, :padmodclamp, :outpre, :outex), Tuple{Set{Symbol}, Vector{Symbol}, Nothing, Vector{Any}, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Int64, Dict{Symbol, Vector{T} where T}, Vector{Tuple}, Vector{Symbol}, Vector{Symbol}, Vector{Expr}, Bool, Vector{Expr}, Vector{Expr}}}})
    Base.precompile(Tuple{typeof(merge),NamedTuple{(:mod,), Tuple{Module}},NamedTuple{(:redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd), Tuple{Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}}}})
    Base.precompile(Tuple{typeof(pairs),Base.Iterators.Pairs{Symbol, Any, NTuple{37, Symbol}, NamedTuple{(:mod, :redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd, :flags, :redind, :init, :leftraw, :leftind, :leftarray, :leftscalar, :leftnames, :right, :finaliser, :rightind, :sharedind, :unsafeleft, :unsaferight, :arrays, :scalars, :cost, :constraints, :pairconstraints, :notfree, :shiftedind, :axisdefs, :padmodclamp, :outpre, :outex), Tuple{Module, Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}, Set{Symbol}, Vector{Symbol}, Nothing, Vector{Any}, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Int64, Dict{Symbol, Vector{T} where T}, Vector{Tuple}, Vector{Symbol}, Vector{Symbol}, Vector{Expr}, Bool, Vector{Expr}, Vector{Expr}}}}})
    Base.precompile(Tuple{typeof(pairs),NamedTuple{(:mod, :redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd, :flags, :redind, :init, :leftraw, :leftind, :leftarray, :leftscalar, :leftnames, :right, :finaliser, :rightind, :sharedind, :unsafeleft, :unsaferight, :arrays, :scalars, :cost, :constraints, :pairconstraints, :notfree, :shiftedind, :axisdefs, :padmodclamp, :outpre, :outex), Tuple{Module, Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}, Set{Symbol}, Vector{Symbol}, Nothing, Vector{Any}, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Int64, Dict{Symbol, Vector{T} where T}, Vector{Tuple}, Vector{Symbol}, Vector{Symbol}, Vector{Expr}, Bool, Vector{Expr}, Vector{Expr}}}})
    Base.precompile(Tuple{typeof(promote_type),Type{Symbol},Type{Any}})
    Base.precompile(Tuple{typeof(push!),Set{Any},Expr})
    Base.precompile(Tuple{typeof(push!),Set{Any},Tuple{Expr, Expr}})
    Base.precompile(Tuple{typeof(push!),Vector{Symbol},Symbol,Symbol})
    Base.precompile(Tuple{typeof(push!),Vector{Tuple},Tuple{Symbol, Symbol, Expr, Expr}})
    Base.precompile(Tuple{typeof(setdiff),Vector{Symbol},Vector{Symbol}})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Dict{Symbol, Vector{T} where T},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Nothing,Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Set{Symbol},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Vector{Any},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Vector{Expr},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Vector{Symbol},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Vector{Tuple},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Vector{Any},Vector{Symbol},UnitRange{Int64}})
    Base.precompile(Tuple{typeof(union),Vector{Symbol},Vector{Symbol}})
    Base.precompile(Tuple{typeof(unique!),Vector{Expr}})
    Base.precompile(Tuple{typeof(vcat),Expr,Vector{Symbol},Vector{Symbol},Vararg{Vector{Symbol}, N} where N})
    Base.precompile(Tuple{typeof(vcat),Vector{Symbol},Expr,Symbol,Vararg{Any, N} where N})
    Base.precompile(Tuple{typeof(vcat),Vector{Symbol},Vector{Symbol}})
    isdefined(Base, Symbol("#73#74")) && Base.precompile(Tuple{getfield(Base, Symbol("#73#74")),Expr})
    isdefined(Base, Symbol("#73#74")) && Base.precompile(Tuple{getfield(Base, Symbol("#73#74")),Int64})
    isdefined(Base, Symbol("#73#74")) && Base.precompile(Tuple{getfield(Base, Symbol("#73#74")),Symbol})
    let fbody = try __lookup_kwbody__(which(any, (Function,Vector{Symbol},))) catch missing end
        if !ismissing(fbody)
            precompile(fbody, (Function,typeof(any),Function,Vector{Symbol},))
        end
    end
end
