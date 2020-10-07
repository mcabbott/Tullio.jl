
include("precompile/precompile_Tullio.jl")
_precompile_()

module _Precompile_Core
    include("precompile/precompile_Core.jl")
end

module _Precompile_Base
    include("precompile/precompile_Base.jl")
end

if VERSION >= v"1.6-"
    _Precompile_Base._precompile_()
    _Precompile_Core._precompile_()
end

#=
# To generate these files, following:
# https://timholy.github.io/SnoopCompile.jl/stable/snoopi/
# For the full benefit, it seems you must first disable them here.

VERSION # 1.6

using Pkg
Pkg.activate(mktempdir())
Pkg.add("SnoopCompile")
using SnoopCompile

using Tullio

inf_timing = @snoopi begin
    Tullio._tullio(:( A[i] := (1:10)[i] ))
    Tullio._tullio(:( A[i+_] := (1:10)[i+j] ), :(i in 1:3))
    Tullio._tullio(:( A[i, J[k]] := B[i] * C[j,k] ), :(grad=Dual))
end

pc = SnoopCompile.parcel(inf_timing)

SnoopCompile.write(joinpath(pkgdir(Tullio), "src", "precompile"), pc)

=#

#=
# Simple test:

using Tullio
@time Tullio._tullio(:( A[i] := (1:10)[i+j] + (1:3)[j]) );

# Julia 1.5.2:
# 7.116702 seconds (17.27 M allocations: 873.755 MiB, 4.21% gc time)
# 5.373509 seconds (5.91 M allocations: 307.566 MiB, 1.39% gc time)

# Julia 1.6-
# 7.703672 seconds (19.13 M allocations: 1.073 GiB, 4.19% gc time)
# 5.538456 seconds (6.53 M allocations: 380.879 MiB, 1.88% gc time)

=#
