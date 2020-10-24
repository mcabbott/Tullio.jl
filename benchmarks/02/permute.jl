using Pkg; Pkg.activate("~/.julia/dev/Tullio/benchmarks/02/")
pkg"add Tullio, LoopVectorization, LinearAlgebra, BenchmarkTools, Einsum, TensorOperations, MKL_jll, https://github.com/haampie/FastTranspose.jl, https://github.com/mcabbott/ArrayMeta.jl, StructArrays"
pkg"up"

Threads.nthreads() == 6

using Tullio, LoopVectorization, Einsum, TensorOperations, FastTranspose, ArrayMeta,Libdl, Test, StructArrays, BenchmarkTools, Plots

TVERSION = VersionNumber(Pkg.TOML.parsefile(joinpath(pkgdir(Tullio), "Project.toml"))["version"])

# adapted from https://github.com/haampie/FastTranspose.jl/blob/master/benchmark/mkl.jl
using MKL_jll
const libMKL = Libdl.dlopen(MKL_jll.libmkl_rt)
const MKL_domatcopy = Libdl.dlsym(libMKL, :mkl_domatcopy)
const MKL_somatcopy = Libdl.dlsym(libMKL, :mkl_somatcopy)
function mkl_matcopy!(B::Matrix{Float64}, A::Matrix{Float64}, alpha = 1.0)
    m, n = size(A)
    ordering = 'C'
    trans = 'T'
    ccall(MKL_domatcopy, Cvoid, (Ref{Cchar}, Ref{Cchar}, Ref{Csize_t}, Ref{Csize_t}, Ref{Cdouble}, Ptr{Float64}, Ref{Csize_t}, Ptr{Float64}, Ref{Csize_t}),
           ordering, trans, m, n, alpha, A, m, B, n)
    return nothing
end
function mkl_matcopy!(B::Matrix{Float32}, A::Matrix{Float32}, alpha = 1.0)
    m, n = size(A)
    ordering = 'C'
    trans = 'T'
    ccall(MKL_somatcopy, Cvoid, (Ref{Cchar}, Ref{Cchar}, Ref{Csize_t}, Ref{Csize_t}, Ref{Cdouble}, Ptr{Float32}, Ref{Csize_t}, Ptr{Float32}, Ref{Csize_t}),
           ordering, trans, m, n, alpha, A, m, B, n)
    return nothing
end

for T in [Float64, Float32]
    local A,B = rand(T, 3,3), rand(T, 3,3)
    mkl_matcopy!(B,A)
    # @test B == A' # fails for Float32?
end

# matrix transpose

base_transpose!(y,x) = permutedims!(y, x, (2,1))
base_lazy_transpose!(y,x) = copyto!(y, transpose(x))
einsum_transpose!(y,x) = @einsum y[i,j] = x[j,i]
arraymeta_transpose!(y,x) = @arrayop y[i,j] = x[j,i]
tensor_transpose!(y,x) = @tensor y[i,j] = x[j,i]
avx_transpose!(y,x) = @tullio y[i,j] = x[j,i] threads=false tensor=false
tullio_transpose!(y,x) = @tullio y[i,j] = x[j,i] tensor=false

functions2 = (base_transpose!, base_lazy_transpose!,
    einsum_transpose!, arraymeta_transpose!, tensor_transpose!,
    avx_transpose!, tullio_transpose!,
    recursive_transpose!, mkl_matcopy!)
sizes2 = sort(vcat(vec((2 .^ (4:12))' .+ [-1,0,1]), [25, 50, 100, 150, 200, 1500]))

# 3-array permutedims

base_312!(y, x) = permutedims!(y, x, (3,1,2))
base_lazy_312!(y, x) = copyto!(y, PermutedDimsArray(x, (3,1,2)))
einsum_312!(y, x) = @einsum y[c,a,b] = x[a,b,c]
arraymeta_312!(y, x) = @arrayop  y[c,a,b] = x[a,b,c]
tensor_312!(y, x) = @tensor y[c,a,b] = x[a,b,c]
avx_312!(y, x) = @tullio y[c,a,b] = x[a,b,c] threads=false tensor=false
tullio_312!(y, x) = @tullio y[c,a,b] = x[a,b,c] tensor=false

functions3 = (base_312!, base_lazy_312!,
    einsum_312!, arraymeta_312!, tensor_312!,
    avx_312!, tullio_312!)
sizes3 = sort(vcat(vec((2 .^ (3:9))' .+ [-1,0,1] ), [25, 50, 100, 150, 200]))

# 4-array permutedims

base_4321!(y, x) = permutedims!(y, x, (4,3,2,1))
base_lazy_4321!(y, x) = copyto!(y, PermutedDimsArray(x, (4,3,2,1)))
einsum_4321!(y, x) = @einsum y[d,c,b,a] = x[a,b,c,d]
arraymeta_4321!(y, x) = @arrayop y[d,c,b,a] = x[a,b,c,d]
tensor_4321!(y, x) = @tensor y[d,c,b,a] = x[a,b,c,d]
avx_4321!(y, x) = @tullio y[d,c,b,a] = x[a,b,c,d] threads=false tensor=false
tullio_4321!(y, x) = @tullio y[d,c,b,a] = x[a,b,c,d] tensor=false

functions4 = (base_4321!, base_lazy_4321!,
    einsum_4321!, arraymeta_4321!, tensor_4321!,
    avx_4321!, tullio_4321!)
sizes4 = sort(vcat(vec((2 .^ (2:6))' .+ [-1,0,1]), [25, 50, 99, 100, 101])) # crashes on 129!

# based on https://github.com/chriselrod/PaddedMatrices.jl/blob/master/benchmark/blasbench.jl

function benchmark_fun!(f!, C, A, force_belapsed = false, reference = nothing)
    tmin = @elapsed f!(C, A)
    if force_belapsed || 2tmin < BenchmarkTools.DEFAULT_PARAMETERS.seconds
        tmin = min(tmin, @belapsed $f!($C, $A))
    elseif tmin < BenchmarkTools.DEFAULT_PARAMETERS.seconds
        tmin = min(tmin, @elapsed f!(C, A))
        if tmin < 2BenchmarkTools.DEFAULT_PARAMETERS.seconds
            tmin = min(tmin, @elapsed f!(C, A))
        end
    end
    isnothing(reference) || @assert C ≈ reference
    tmin
end

function runbench(funs::Tuple, dims::Int=2, sizes=[10,30,100])
    (StructVector ∘ map)(sizes) do n
        A = rand(ntuple(_->n, dims)...)
        C0, C = similar(A), similar(A)
        funs[1](C0, A)
        times = map(funs) do f!
            t = benchmark_fun!(f!, C, A)
            C ≈ C0 || @error "disagreement for $f! at size $n"
            t
        end
        nt = NamedTuple{map(Symbol, funs)}(times)
        res = (size=n, length=length(C), nt...)
        @show res
    end
end

r0 = runbench((base_transpose!, mkl_matcopy!, tullio_transpose!), 2, [10,30,100])

function makeplot(res, title="")
    plt = plot()
    for lab in propertynames(res)[3:end]
        times = getproperty(res, lab)
        flops = 1e-9 * res.length ./ times # was wrong in 0.2.0 plots

        if startswith(string(lab), "tullio")
            plot!(plt, res.size, flops, lab=string(lab), m=:circle)
        else
            plot!(plt, res.size, flops, lab=string(lab))
        end
    end
    plot!(plt, yaxis=("numbers / ns", ([0.5,1,2,4,8], ["1/2","1",2,4,8]), :log10), xaxis=("size per dimension", :log10), legend=:bottomleft)
    plot!(plt, title = title * "Julia " * string(VERSION) * ", Intel " * split(Sys.cpu_info()[1].model)[3])
end

makeplot(r0)
# savefig(joinpath("~/.julia/dev/Tullio", "benchmarks/02/trash.png"))

res2 = runbench(functions2, 2, sizes2)
plot2 = makeplot(res2, "Tullio $TVERSION, ")
savefig(plot2, joinpath("~/.julia/dev/Tullio", "benchmarks/02/transpose-$TVERSION-$VERSION.png"))

res3 = runbench(functions3, 3, sizes3)
plot3 = makeplot(res3, "Tullio $TVERSION, ")
savefig(plot3, joinpath("~/.julia/dev/Tullio", "benchmarks/02/permute3-$TVERSION-$VERSION.png"))

res4 = runbench(functions4, 4, sizes4)
plot4 = makeplot(res4, "Tullio $TVERSION, ")
savefig(plot4, joinpath("~/.julia/dev/Tullio", "benchmarks/02/permute4-$TVERSION-$VERSION.png"))
