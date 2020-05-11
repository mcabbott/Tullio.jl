# Tullio.jl

This is roughly a re-write of the [`Einsum.@einsum`](https://github.com/ahwillia/Einsum.jl) macro, which takes expressions like 
```julia
@tullio C[i,j] := A[i,k] * B[k,j]
```
and writes loops which fill in the matrix `C`, by summing the right hand side at all possible values of free index `k`. The differences are:

1. It understands more syntax, including shifts of indices (by constants or other indices, such as `C[i] := A[i+j-1] * K[j]`), arrays of arrays, fields of their elements, and keyword indexing. 

2. It calculates gradients for reverse-mode auto-differentiation, by making a second pass with either a symbolic derivative of the right hand side, or else using `(A[i,k] + ϵA) * (B[k,j] + ϵB)` with dual numbers `ϵA, ϵB`. 

3. It should be faster, by using [`Threads.@spawn`](https://julialang.org/blog/2019/07/multithreading/) and blocking on large arrays, and using [`LoopVectorization.@avx`](https://github.com/chriselrod/LoopVectorization.jl) when possible. 

4. It uses [`KernelAbstractions.@kernel`](https://github.com/JuliaGPU/KernelAbstractions.jl) to write a GPU version, slightly experimentally.

### Examples

Notation:

```julia
using Pkg; pkg"add https://github.com/mcabbott/Tullio.jl"
using Tullio
A = [abs2(i - 11) for i in 1:21]

# Downsample -- range of i is that allowed by both terms:
@tullio D[i] := (A[2i] + A[2i+1])/2  # 1:10 == intersect(1:10, 0:10)

# Shifts -- range of i calculated in terms of that given for j:
@tullio M[i,j] := A[i+j-1]  (j in 1:15)  # i in 1:7

using OffsetArrays # Convolve a filter:
K = OffsetArray([1,-1,2,-1,1], -2:2)
@tullio C[i] := A[i+j] * K[j]  # j ∈ -2:2 implies i ∈ 3:19

using FFTW # Functions of the indices are OK:
S = [0,1,0,0, 0,0,0,0]
fft(S) ≈ @tullio (k ∈ axes(S,1)) F[k] := S[x] * exp(-im*pi/8 * (k-1) * x)

# Access to fields & arrays -- this uses j ∈ eachindex(first(N).c)
N = [(a=i, b=i^2, c=fill(i^3,3)) for i in 1:10]
@tullio T[i,j] := (N[i].a // 1, N[i].c[j])

# Functions which create arrays are evaluated once:
@tullio R[i,j] := abs.((rand(Int8, 5)[i], rand(Int8, 5)[j]))
R == reverse.(permutedims(T))
```

Threads & SIMD:

```julia
using Tullio, LoopVectorization, NNlib, BenchmarkTools

# Batched matmul with batch index first in B, defined with @avx loops:
bmm_rev(A, B) = @tullio C[i,k,b] := A[i,j,b] * B[b,k,j]  # (sum over j)

A = randn(20,30,500); B = randn(500,40,30);
bmm_rev(A, B) ≈ NNlib.batched_mul(A, permutedims(B, (3,2,1))) # true

@btime bmm_rev($A, $B); # 317.526 μs μs, same speed as un-permuted bmm
@btime NNlib.batched_mul($A, permutedims($B, (3,2,1))); # 1.478 ms, with MKL

# Complete reduction, without first materialising X .* log.(Y')
sum_opp(X, Y=X) = @tullio s := X[i,j] * log(Y[j,i])

X = rand(1000,1000);
@btime sum_opp($X)                    #   499.814 μs (173 allocations: 14.20 KiB)
@btime sum($X .* log.(transpose($X))) # 8.759 ms (2 allocations: 7.63 MiB)
```

Derivatives & GPU:

```julia
using Tullio, Tracker # This is defined with a gradient:
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k] 

A = rand(3,40); B = rand(40,500);
A * B ≈ mul(A, B) # true

ΔA = Tracker.gradient((A,B) -> sum(mul(A, B)), A, B)[1]
ΔA ≈ ones(3,500) * B' # true

using CuArrays, KernelAbstractions # Now defined with a GPU version:
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

cu(A * B) ≈ mul(cu(A), cu(B)) # true

cu(ΔA) ≈ Tracker.gradient((A,B) -> sum(mul(A, B)), cu(A), cu(B))[1] # true
```

Larger expressions:

```julia
mat = zeros(10,10,1); mat[1,1] = 101;
@tullio kern[i,j] := 1/(1+i^2+j^2)  (i in -2:2, j in -2:2)

@tullio out[x,y,c] := begin
    xi = mod(x+i, axes(mat,1)) # xi = ... means that it won't be summed,
    yj = mod(y+j, axes(mat,2))
    @inbounds trunc(Int, mat[xi, yj, c] * kern[i,j]) # and disables automatic @inbounds,
end (x in 1:10, y in 1:10) # and prevents range of x from being inferred.
```

### Options

The default setting is:
```@tullio threads=true avx=true grad=Base verbose=false A[i,j] := ...``` 
* `threads=false` turns off threading, while `threads=64^3` sets a threshold size at which to divide the work (replacing the macro's best guess).
* `avx=false` turns off the use of `LoopVectorization`, while `avx=4` inserts `@avx unroll=4 for i in ...`.
* `grad=false` turns off gradient calculation, and `grad=Dual` switches it to use `ForwardDiff` (which must be loaded).
* Assignment `xi = ...` removes `xi` from the list of indices: its range is note calculated, and it will not be summed over. It also disables `@inbounds` since this is now up to you.
* `verbose=true` prints everything; you can't use `@macroexpand1` as it needs to `eval` rather than return gradient definitions.
* `A[i,j] := ...` makes a new array, while `A[i,j] = ...` and `A[i,j] += ...` write into an existing one. `A[row=i, col=j] := ...` makes a new `NamedDimsArray`.

Implicit:
* Indices without shifts must have the same range everywhere they appear, but those with shifts (even `A[i+0]`) run over the inersection of possible ranges.
* Shifted output indices must start at 1, unless `OffsetArrays` is visible in the calling module.
* The use of `@avx`, and the calculation of gradients, are switched off by sufficiently complex syntax (such as arrays of arrays). 
* Gradient hooks are attached for any or all of `ReverseDiff`, `Tracker`, `Zygote` & `Yota`, according to which of these packages are visible. 
* GPU kernels are only constructed when both `KernelAbstractions` and `CuArrays` are visible.

Extras:
* `A[i] := i^2  (i in 1:10)` is how you specify a range for indices when this can't be inferred. 
* `Tullio.@printgrad (x+y)*log(x/z)   x y z` prints out how symbolic derivatives will be done. 

### Elsewhere

Back-end friends & relatives:

* [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) is used here, if available. 

* [Gaius.jl](https://github.com/MasonProtter/Gaius.jl) and [PaddedMatrices.jl](https://github.com/chriselrod/PaddedMatrices.jl) build on that.

* [GPUifyLoops.jl](https://github.com/vchuravy/GPUifyLoops.jl) and [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) generate GPU-compatable kernels.

* [ThreadsX.jl](https://github.com/tkf/ThreadsX.jl) does threaded reductions, and much else.

* [Strided.jl](https://github.com/Jutho/Strided.jl) does multi-threaded broadcasting.

Front-end near-lookalikes:

* [Einsum.jl](https://github.com/ahwillia/Einsum.jl) makes simple loops. See [tests/einsum.jl](https://github.com/mcabbott/Tullio.jl/blob/master/test/einsum.jl) where `using Tullio: @einsum` is an almost-seamless replaceement.

* [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) and [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) identify patterns on which they can call various basic operations.

* [TensorCast.jl](https://github.com/mcabbott/TensorCast.jl) expresses everything as Julia array operations, broadcasting and reduction. (OMEinsum.jl also treats some cases as a special lazy broadcast-reduction.)

Things you can't run:

* [Tortilla.jl](https://www.youtube.com/watch?v=Rp7sTl9oPNI) seems to exist, publicly, only in this very nice talk. 

* [ArrayMeta.jl](https://github.com/shashi/ArrayMeta.jl) was a Julia 0.5 take on some of this.

* [Tokamak.jl](https://github.com/tkelman/Tokamak.jl) was another.
