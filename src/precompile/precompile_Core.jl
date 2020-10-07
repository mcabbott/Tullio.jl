function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:mod,), T} where T<:Tuple},Tuple{Module}})
    Base.precompile(Tuple{Type{NamedTuple{(:redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd), T} where T<:Tuple},Core.Tuple{Core.Symbol, Core.Symbol, Core.Symbol, Core.Bool, Core.Bool, Core.Bool, Core.Symbol, Core.Bool, Core.Int64, Core.Bool, Base.Vector{Core.Symbol}}})
    Base.precompile(Tuple{typeof(Core.Compiler.getindex),Base.Vector{Core.Compiler.CallMeta},Int64})
    Base.precompile(Tuple{typeof(Core.Compiler.length),Base.Vector{Core.Compiler.CallMeta}})
end
