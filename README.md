# Tullio.jl

This is roughly a re-write of the [`Einsum.@einsum`](https://github.com/ahwillia/Einsum.jl) macro, which takes expressions like 
```julia
@tullio C[i,j] := A[i,k] * B[k,j]
```
and writes loops which fill in the matrix `C`, by summing the right hand side at all possible values of free index `k`. The differences are:

1. It understands more syntax, including shifts of indices (by constants or other indices, such as `C[i] := A[i+j-1] * K[j]`), arrays of arrays, fields of their elements, and keyword indexing. Shifts result in indices running over the intersection of ranges inferred (rather than demanding agreement).

2. It calculates gradients for reverse-mode auto-differentiation, by making a second pass with either a symbolic derivative of the right hand side, or else using `(A[i,k] + ϵA) * (B[k,j] + ϵB)` with dual numbers `ϵA, ϵB`. 

3. It should be faster, by using blocking and [`Threads.@spawn`](https://julialang.org/blog/2019/07/multithreading/) on large arrays, and by using [`LoopVectorization.@avx`](https://github.com/chriselrod/LoopVectorization.jl) when possible. 

4. It uses [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) to write a GPU version, slightly experimentally.

### Options

The default setting is:
```@tullio threads=true grad=Base avx=true verbose=false A[i,j] := ...``` 
* `grad=false` turns off gradient calculation, and `grad=Dual` switches it to use `ForwardDiff` (which must be loaded).
* `avx=false` turns off `@avx`, while `avx=4` inserts `@avx unroll=4 for i in ...`.
* `threads=false` turns off threading, while `threads=64^3` sets a threshold size at which to divide the work.
* `verbose=true` prints everything; you can't use `@macroexpand1` as it needs to `eval` rather than return gradient definitions.
* `A[i,j] := ...` makes a new array, while `A[i,j] = ...` and `A[i,j] += ...` write into an existing one.
* `A[row=i, col=j] := ...` makes a new NamedDimsArray.

Implicit:
* Output indices must start at 1, unless `OffsetArrays` is visible in the calling module.
* The use of `@avx`, and the calculation of gradients, are switched off by sufficiently complex syntax (such as arrays of arrays). 
* Gradient hooks are attached for any or all of `ReverseDiff`, `Tracker`, `Zygote` & `Yota`, according to which of these packages are visible. 
* GPU kernels are only constructed when both `KernelAbstractions` and `CuArrays` are visible, and `VERSION` is 1.3.

Extras:
* `A[i] := i^2  (i in 1:10)` is how you specify a range for indices when this can't be inferred. 
* `Tullio.@printgrad (x+y)*log(x/z)   x y z` prints out how symbolic derivatives will be done. 

### Examples

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

# Access to fields & arrays -- this uses `axes(first(N).c, 1)`
N = [(a=i, b=i^2, c=fill(i^3,3)) for i in 1:10]
@tullio T[i,j] := (N[i].a // 1, N[i].c[j])

# Functions which create arrays are evaluated once:
@tullio T[i,j] := abs.((rand(Int8, 5)[i], rand(Int8, 5)[j]))
T == reverse.(permutedims(T))
```

Derivatives & GPU:

```julia
using Tullio, Tracker, CuArrays, KernelAbstractions
A = rand(3,40); B = rand(40,500);
cA = cu(A); cB = cu(B);

mul(A,B) = @tullio C[i,k] := A[i,j] * B[j,k]
cC = mul(cA,cB) 

ΔA = Tracker.gradient((A,B) -> sum(identity, mul(A,B)), cA, cB)[1]
collect(ΔA) ≈ ones(size(A*B)) * B'

A0 = rand(300,400); B0 = rand(400,500);
@btime mul($A0, $B0);
@btime ($A0 * $B0); # twice as quick, thanks to MKL

cA0 = cu(A0); cB0 = cu(B0);
@btime CuArrays.@sync mul($cA0, $cB0);
@btime CuArrays.@sync ($cA0 * $cB0); # 400 times as quick!
```

### Elsewhere

Back-end friends & relatives:

* [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) is used here, if available. 

* [Gaius.jl](https://github.com/MasonProtter/Gaius.jl) is a pure-Julia BLAS, using that.

* [GPUifyLoops.jl](https://github.com/vchuravy/GPUifyLoops.jl) and [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) generate GPU-compatable kernels.

* [ThreadsX.jl](https://github.com/tkf/ThreadsX.jl) does threaded reductions, and much else.

* [Strided.jl](https://github.com/Jutho/Strided.jl) does multi-threaded broadcasting.

Front-end near-lookalikes:

* [Einsum.jl](https://github.com/ahwillia/Einsum.jl) makes simple loops. This package should (eventually) be a strict upgrade: `using Tullio: @einsum` even gives a macro with the same name.

* [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) and [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) identify patterns on which they can call various basic operations.

* [TensorCast.jl](https://github.com/mcabbott/TensorCast.jl) expresses everything as Julia array operations, broadcasting and reduction.

