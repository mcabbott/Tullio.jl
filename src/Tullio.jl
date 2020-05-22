module Tullio

export @tullio

@nospecialize

include("tools.jl")

include("macro.jl")

include("symbolic.jl")

include("forward.jl")

include("einsum.jl")

@specialize

include("eval.jl")

include("shifts.jl")

include("threads.jl")

end # module
