
using Pkg; Pkg.activate("~/.julia/dev/Tullio/benchmarks/02/")
pkg"add Tullio, LoopVectorization, Compat, PaddedMatrices, StructArrays, LinearAlgebra, BenchmarkTools, Libdl, MKL_jll, OpenBLAS_jll"
pkg"up"

Threads.nthreads() == 6

using Tullio, LoopVectorization, Compat

TVERSION = VersionNumber(Pkg.TOML.parsefile(joinpath(pkgdir(Tullio), "Project.toml"))["version"])

tmul!(C,A,B) = @tullio C[i,j] := A[i,k] * B[k,j]

# Adapted from:
# https://github.com/chriselrod/PaddedMatrices.jl/blob/master/benchmark/blasbench.jl

using PaddedMatrices, StructArrays, LinearAlgebra, BenchmarkTools, Libdl

randa(::Type{T}, dim...) where {T} = rand(T, dim...)
randa(::Type{T}, dim...) where {T <: Signed} = rand(T(-100):T(200), dim...)

using MKL_jll, OpenBLAS_jll

const libMKL = Libdl.dlopen(MKL_jll.libmkl_rt)
const DGEMM_MKL = Libdl.dlsym(libMKL, :dgemm)
const SGEMM_MKL = Libdl.dlsym(libMKL, :sgemm)
const DGEMV_MKL = Libdl.dlsym(libMKL, :dgemv)
const MKL_SET_NUM_THREADS = Libdl.dlsym(libMKL, :MKL_Set_Num_Threads)

const libOpenBLAS = Libdl.dlopen(OpenBLAS_jll.libopenblas)
const DGEMM_OpenBLAS = Libdl.dlsym(libOpenBLAS, :dgemm_64_)
const SGEMM_OpenBLAS = Libdl.dlsym(libOpenBLAS, :sgemm_64_)
const DGEMV_OpenBLAS = Libdl.dlsym(libOpenBLAS, :dgemv_64_)
const OPENBLAS_SET_NUM_THREADS = Libdl.dlsym(libOpenBLAS, :openblas_set_num_threads64_)

istransposed(x) = 'N'
istransposed(x::Adjoint{<:Real}) = 'T'
istransposed(x::Adjoint) = 'C'
istransposed(x::Transpose) = 'T'

for (lib,f) ∈ [(:GEMM_MKL,:gemmmkl!), (:GEMM_OpenBLAS,:gemmopenblas!)]
    for (T,prefix) ∈ [(Float32,:S),(Float64,:D)]
        fm = Symbol(prefix, lib)
        @eval begin
            function $f(C::AbstractMatrix{$T}, A::AbstractMatrix{$T}, B::AbstractMatrix{$T})
                transA = istransposed(A)
                transB = istransposed(B)
                M, N = size(C); K = size(B, 1)
                pA = parent(A); pB = parent(B)
                ldA = stride(pA, 2)
                ldB = stride(pB, 2)
                ldC = stride(C, 2)
                α = one($T)
                β = zero($T)
                ccall(
                    $fm, Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{$T}, Ref{$T},
                     Ref{Int64}, Ref{$T}, Ref{Int64}, Ref{$T}, Ref{$T}, Ref{Int64}),
                    transA, transB, M, N, K, α, pA, ldA, pB, ldB, β, C, ldC
                )
            end
        end
    end
end
mkl_set_num_threads(N::Integer) = ccall(MKL_SET_NUM_THREADS, Cvoid, (Int32,), N % Int32)
# mkl_set_num_threads(1)
openblas_set_num_threads(N::Integer) = ccall(OPENBLAS_SET_NUM_THREADS, Cvoid, (Int64,), N)
# openblas_set_num_threads(1)

function benchmark_fun!(f!, C, A, B, force_belapsed = false, reference = nothing)
    tmin = @elapsed f!(C, A, B)
    if force_belapsed || 2tmin < BenchmarkTools.DEFAULT_PARAMETERS.seconds
        tmin = min(tmin, @belapsed $f!($C, $A, $B))
    elseif tmin < BenchmarkTools.DEFAULT_PARAMETERS.seconds
        tmin = min(tmin, @elapsed f!(C, A, B))
        if tmin < 2BenchmarkTools.DEFAULT_PARAMETERS.seconds
            tmin = min(tmin, @elapsed f!(C, A, B))
        end
    end
    isnothing(reference) || @assert C ≈ reference
    tmin
end

function runbench(::Type{T}, sizes = [2:255..., round.(Int, range(57.16281374121401, length=200) .^ 1.3705658916944428)...]) where {T}
    (StructVector ∘ map)(sizes) do sz
        n, k, m = sz, sz, sz
        C1 = Matrix{T}(undef, n, m)
        C2 = similar(C1);
        C3 = similar(C1);
        C4 = similar(C1);
        # C5 = similar(C1);
        # C6 = similar(C1);
        A  = randa(T, n, k)
        B  = randa(T, k, m)
        # BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.05

        tmlt = benchmark_fun!(tmul!, C1, A, B, sz == first(sizes))

        jmlt = benchmark_fun!(PaddedMatrices.jmul!, C1, A, B, sz == first(sizes))
        # res = if T <: Integer
        #     (matrix_size=sz, MaBLAS_24x9=ma24, MaBLAS_32x6=ma32, MaBLAS_40x5=ma40, PaddedMatrices=jmlt)
        # else
            opbt = benchmark_fun!(gemmopenblas!, C2, A, B, sz == first(sizes), C1)
            mklbt= benchmark_fun!(gemmmkl!, C3, A, B, sz == first(sizes), C1)

            res = (matrix_size=sz, OpenBLAS=opbt, MKL=mklbt, PaddedMatrices=jmlt, Tullio=tmlt)
        # end
        @show res
    end
end

b5 = runbench(Float32, [3,10,30,100,300])

using Plots

function makeplot(res, title="")
    plt = plot()
    for lab in propertynames(res)[2:end]
        times = getproperty(res, lab)
        flops = 2e-9 * res.matrix_size.^3 ./ times
        str = lab==:MKL ? "MKL $(Compat.get_num_threads())" :
            lab==:OpenBLAS ? "OpenBLAS $(Compat.get_num_threads())" :
            lab==:Tullio ? "Tullio $(Threads.nthreads())" :
            string(lab)
        lab==:Tullio ?
            plot!(res.matrix_size, flops, lab=str, m=:circle) :
            plot!(res.matrix_size, flops, lab=str)
    end
    plot!(yaxis=("gigaflops", ([12.5,25,50,100,200,400],["12.5",25,50,100,200,400]), :log10), xaxis=("size", :log10), legend=:bottomright)
    # plot!(1:0, 1:0, c=:white, lab="i7-8700 + $VERSION")
    plot!(1:0, 1:0, c=:white, lab="Intel " * split(Sys.cpu_info()[1].model, " ")[3], title=title * "Julia " * string(VERSION))
end

makeplot(b5, "warmup ")

# Threads.nthreads() <= 6 || error("expected to run with at most 6 threads")
mkl_set_num_threads(Threads.nthreads())
openblas_set_num_threads(Threads.nthreads())

for Ty in [Float64, Float32]
    global b36

    b36 = runbench(Ty, [10,11,12, 20,21, 30,31,32,33, 49,50,51, 63,64,65,66, 77,78, 100,101,102, 127,128,129, 200, 255,256,257, 300, 400, 500, 511,512,513, 600, 700, 800, 999,1000,1024,1025, 1600,1601, 1999,2000])

    p36 = makeplot(b36, "$Ty, Tullio $TVERSION, ")
    savefig(p36, joinpath("~/.julia/dev/Tullio", "benchmarks/02/matmul-$TVERSION-$Ty-$VERSION.png"))

end

# Summary:

# it's fine at small sizes, sometimes beating OpenBLAS, thanks entirely to @avx
# Threading helps at large sizes, but it still ends up about half the speed.
# Which is OK, the goal isn't to replace BLAS, it's to do other things!
# This is just a test to see how much is left on the table.

