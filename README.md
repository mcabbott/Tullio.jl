# Tullio.jl

[![Build Status](https://travis-ci.org/mcabbott/Tullio.jl.svg?branch=master)](https://travis-ci.org/mcabbott/Tullio.jl)

This is a package is for writing array operations in index notation, such as:

```julia
@tullio M[x,y,c] := N[x+i, y+j,c] * K[i,j]     # sum over i,j, and create M

@tullio S[x] = P[x,y] * log(Q[x,y] / R[y])     # sum over y, and write into S

@tullio A[i,j] += B[i,k,l] * C[l,j] * D[k,j]   # sum over k,l, and add to values in A

@tullio (*) Z[j] := X[ind[k],j] * exp(-Y[k])   # product over k
```

Used by itself the macro writes ordinary nested loops much like [`Einsum.@einsum`](https://github.com/ahwillia/Einsum.jl).
One difference is that it can parse more expressions (such as the convolution `M`, and worse).
Another is that it will use multi-threading (via [`Threads.@spawn`](https://julialang.org/blog/2019/07/multithreading/)), dividing large enough arrays into blocks. 
But it also co-operates with various other packages, provided they are loaded before the macro is called:

* It can use [`LoopVectorization.@avx`](https://github.com/chriselrod/LoopVectorization.jl) to speed many things up. (Disable with `avx=false`.)

* It can use [`KernelAbstractions.@kernel`](https://github.com/JuliaGPU/KernelAbstractions.jl) to make a GPU version. (Disable with `cuda=false`.)

* It can use [`TensorOperations.@tensor`](https://github.com/Jutho/TensorOperations.jl) on expressions which this understands. (Disable with `tensor=false`.) These must be Einstein-convention contractions of one term; none of the examples above qualify.

Gradients are handled as follows:

* It will try to take a symbolic derivative of the right hand side expression, for use with any of [Tracker](https://github.com/FluxML/Tracker.jl), [Zygote](https://github.com/FluxML/Zygote.jl) or [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl). (Disable with `grad=false`.) When using `@tensor`, this writes another `@tensor` expression for each input array. Otherwise, it generates one set of loops to fill in all the gradient arrays at once.  

* The option `grad=Dual` uses instead [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) to differentiate the right hand side. This allows for more complicated expressions.

The expression need not be just one line, for example:

```julia
@tullio out[x,y,n] := begin              # sum over a,b
        i = mod(x+a, axes(mat,1))
        j = mod(y+b, axes(mat,2))
        @inbounds mat[i,j,n] * abs(kern[a,b])
    end (x in axes(mat,1), y in axes(mat,2)) grad=Dual
```

Here the macro cannot infer the range of the output's indices `x,y`, 
so they must be provided explicitly. (If writing into an existing array, 
with `out[x,y,n] = begin ...` or `+=`, then ranges would be taken from there.) 
It knows that it should not sum over indices `i,j`, but since it can't be sure 
of their ranges, it will not add `@inbounds` in such cases. 
It will also not be able to take a symbolic derivative here, but dual numbers will work fine.

The option `@tullio verbose=true` will cause it to print index ranges, symbolic derivatives,
and notices when it is unable to use the packages mentioned above.

<details><summary><b>Notation</b></summary>

```julia
using Pkg; pkg"add Tullio" # now registered
using Tullio
A = [abs2(i - 11) for i in 1:21]

# Downsample -- range of i is that allowed by both terms:
@tullio D[i] := (A[2i] + A[2i+1])/2  # 1:10 == intersect(1:10, 0:10)

# Shifts -- range of i calculated in terms of that given for j:
@tullio M[i,j] := A[i+j-1]  (j in 1:15)  # i in 1:7

using OffsetArrays # Convolve a filter:
K = OffsetArray([1,-1,2,-1,1], -2:2)
@tullio C[i] := A[i+j] * K[j]  # j ∈ -2:2 implies i ∈ 3:19

# Index by the values in K
@tullio D[i,j] := A[2K[j]+i] ÷ K[j] # extrema(K)==(-1,2) implies i ∈ 3:17

using FFTW # Functions of the indices are OK:
S = [0,1,0,0, 0,0,0,0]
fft(S) ≈ @tullio (k ∈ axes(S,1)) F[k] := S[x] * exp(-im*pi/8 * (k-1) * x)

# Access to fields & arrays -- this uses j ∈ eachindex(first(N).c)
N = [(a=i, b=i^2, c=fill(i^3,3)) for i in 1:10]
@tullio T[i,j] := (N[i].a // 1, N[i].c[j])

# Functions which create arrays are evaluated once:
@tullio R[i,j] := abs.((rand(Int8, 5)[i], rand(Int8, 5)[j]))

using NamedDims, AxisKeys # Dimension names, plus pretty printing:
@tullio M[row=i, col=j, z=k] := A[i+j-1]  (j in 1:15, k in 1:2)
@tullio S[i] := M[col=j-i, z=k, row=i+1] # sum over j,k
```

</details>
<details><summary><b>Threads & SIMD</b></summary>

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

</details>
<details><summary><b>Derivatives & GPU</b></summary>

```julia
using Tullio
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k] 

A = rand(3,40); B = rand(40,500);
A * B ≈ mul(A, B) # true

using Tracker
ΔA = Tracker.gradient((A,B) -> sum(mul(A, B)), A, B)[1]
ΔA ≈ ones(3,500) * B' # true

using CuArrays, KernelAbstractions # Now defined with a GPU version:
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

cu(A * B) ≈ mul(cu(A), cu(B)) # true

cu(ΔA) ≈ Tracker.gradient((A,B) -> sum(mul(A, B)), cu(A), cu(B))[1] # true
```

</details>
<details><summary><b>Larger expressions</b></summary>

```julia
mat = zeros(10,10,1); mat[1,1] = 101;
@tullio kern[i,j] := 1/(1+i^2+j^2)  (i in -2:2, j in -2:2)

@tullio out[x,y,c] := begin
    xi = mod(x+i, axes(mat,1)) # xi = ... means that it won't be summed,
    yj = mod(y+j, axes(mat,2))
    @inbounds trunc(Int, mat[xi, yj, c] * kern[i,j]) # and disables automatic @inbounds,
end (x in 1:10, y in 1:10) # and prevents range of x from being inferred.
```

</details>
<details><summary><b>Options</b></summary>

The default setting is:
```@tullio threads=true fastmath=true avx=true tensor=true cuda=256 grad=Base verbose=false A[i,j] := ...``` 
* `threads=false` turns off threading, while `threads=64^3` sets a threshold size at which to divide the work (replacing the macro's best guess).
* `avx=false` turns off the use of `LoopVectorization`, while `avx=4` inserts `@avx unroll=4 for i in ...`.
* `grad=false` turns off gradient calculation, and `grad=Dual` switches it to use `ForwardDiff` (which must be loaded).
* `tensor=false` turns off the use of `TensorOperations`.
* Assignment `xi = ...` removes `xi` from the list of indices: its range is note calculated, and it will not be summed over. It also disables `@inbounds` since this is now up to you.
* `verbose=true` prints things like the index ranges inferred, and gradient calculations. `verbose=2` prints absolutely everything.
* `A[i,j] := ...` makes a new array, while `A[i,j] = ...` and `A[i,j] += ...` write into an existing one. `A[row=i, col=j] := ...` makes a new `NamedDimsArray`.
* `@tullio (*) A[i,j] := ...` is a product, as is `@tullio A[i,j] *= ...`. 

Implicit:
* Indices without shifts must have the same range everywhere they appear, but those with shifts (even `A[i+0]`) run over the inersection of possible ranges.
* Shifted output indices must start at 1, unless `OffsetArrays` is visible in the calling module.
* The use of `@avx`, and the calculation of gradients, are switched off by sufficiently complex syntax (such as arrays of arrays). 
* Gradient hooks are attached for any or all of `ReverseDiff`, `Tracker` & `Zygote`. These packages need not be loaded when the macro is run.
* Gradients are only defined for reductions over `(+)` (default) and `min`, `max`.
* GPU kernels are only constructed when both `KernelAbstractions` and `CuArray` are visible. The default `cuda=256` is passed to `kernel(CUDA(), 256)`.
* The CPU kernels from `KernelAbstractions` are called only when `threads=false`; they are not at present very fast, but perhaps useful for testing.

Extras:
* `A[i] := i^2  (i in 1:10)` is how you specify a range for indices when this can't be inferred. 
* `A[i] := B[i, $col] - C[i, 2]` is how you fix one index to a constant (to prevent `col` being summed over).
* `A[i] := $d * B[i]` is the preferred way to include other constants. Note that no gradient is calculated for `d`. 
* `Tullio.@printgrad (x+y)*log(x/z)   x y z` prints out how symbolic derivatives will be done. 

</details>
<details><summary><b>Elsewhere</b></summary>

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

* [Tokamak.jl](https://github.com/MikeInnes/Tokamak) was another, see [readme here](https://github.com/tkelman/Tokamak.jl).
