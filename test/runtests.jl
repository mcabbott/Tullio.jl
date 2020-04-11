
using Test

t1 = time()
using Tullio
t2 = round(time()-t1, digits=1)
@info "Loading Tullio took $t2 seconds"

@testset "parsing all the things" begin include("parsing.jl") end

@testset "backward gradients" begin include("gradients.jl") end

@testset "internal pieces" begin include("utils.jl") end

@testset "tests from Einsum.jl" begin include("einsum.jl") end
