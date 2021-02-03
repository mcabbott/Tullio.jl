module Tullio

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)
export @tullio

@nospecialize

include("tools.jl")

include("macro.jl")

include("tensor.jl")

include("symbolic.jl")

include("forward.jl")

include("einsum.jl")

@specialize

include("eval.jl")

include("shifts.jl")

include("threads.jl")

include("precompile.jl")

end # module
