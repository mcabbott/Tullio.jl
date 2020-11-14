<div align="center">
<h1>Tullio.jl</h1>

<!--[![Travis CI](https://img.shields.io/travis/mcabbott/Tullio.jl/master?logo=travis)](https://travis-ci.org/mcabbott/Tullio.jl)-->
[![Github CI](https://github.com/mcabbott/TransmuteDims.jl/workflows/CI/badge.svg)](https://github.com/mcabbott/TransmuteDims.jl/actions)
[![Gitlab GPU](https://img.shields.io/gitlab/pipeline/JuliaGPU/Tullio.jl/master?logo=nvidia&color=ddd)](https://gitlab.com/JuliaGPU/Tullio.jl/-/pipelines)
[![Tag Version](https://img.shields.io/github/v/tag/mcabbott/Tullio.jl?color=red&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB3aWR0aD0iMzI1cHQiIGhlaWdodD0iMzAwcHQiIHZpZXdCb3g9IjAgMCAzMjUgMzAwIiB2ZXJzaW9uPSIxLjEiPgo8ZyBpZD0ic3VyZmFjZTkxIj4KPHBhdGggc3R5bGU9IiBzdHJva2U6bm9uZTtmaWxsLXJ1bGU6bm9uemVybztmaWxsOnJnYig3OS42JSwyMy41JSwyMCUpO2ZpbGwtb3BhY2l0eToxOyIgZD0iTSAxNTAuODk4NDM4IDIyNSBDIDE1MC44OTg0MzggMjY2LjQyMTg3NSAxMTcuMzIwMzEyIDMwMCA3NS44OTg0MzggMzAwIEMgMzQuNDc2NTYyIDMwMCAwLjg5ODQzOCAyNjYuNDIxODc1IDAuODk4NDM4IDIyNSBDIDAuODk4NDM4IDE4My41NzgxMjUgMzQuNDc2NTYyIDE1MCA3NS44OTg0MzggMTUwIEMgMTE3LjMyMDMxMiAxNTAgMTUwLjg5ODQzOCAxODMuNTc4MTI1IDE1MC44OTg0MzggMjI1ICIvPgo8cGF0aCBzdHlsZT0iIHN0cm9rZTpub25lO2ZpbGwtcnVsZTpub256ZXJvO2ZpbGw6cmdiKDIyJSw1OS42JSwxNC45JSk7ZmlsbC1vcGFjaXR5OjE7IiBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSAiLz4KPHBhdGggc3R5bGU9IiBzdHJva2U6bm9uZTtmaWxsLXJ1bGU6bm9uemVybztmaWxsOnJnYig1OC40JSwzNC41JSw2OS44JSk7ZmlsbC1vcGFjaXR5OjE7IiBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1ICIvPgo8L2c+Cjwvc3ZnPgo=)](https://github.com/mcabbott/Tullio.jl/releases)
</div>

Tullio is a very flexible einsum macro. It understands many array operations written in index notation, for example:

```julia
@tullio M[x,y,c] := N[x+i, y+j,c] * K[i,j]     # sum over i,j, and create M

@tullio S[x] = P[x,y] * log(Q[x,y] / R[y])     # sum over y, and write into S

@tullio A[i,j] += B[i,k,l] * C[l,j] * D[k,j]   # sum over k,l, and add to values in A

@tullio (*) Z[j] := X[ind[k],j] * exp(-Y[k])   # product over k
```

Used by itself the macro writes ordinary nested loops much like [`Einsum.@einsum`](https://github.com/ahwillia/Einsum.jl).
One difference is that it can parse more expressions (such as the convolution `M`, and worse).
Another is that it will use multi-threading (via [`Threads.@spawn`](https://julialang.org/blog/2019/07/multithreading/)) and recursive tiling, on large enough arrays. 
But it also co-operates with various other packages, provided they are loaded before the macro is called:

* It uses [`LoopVectorization.@avx`](https://github.com/chriselrod/LoopVectorization.jl) to speed many things up. (Disable with `avx=false`.) On a good day this will match the speed of OpenBLAS for matrix multiplication.

* It uses [`TensorOperations.@tensor`](https://github.com/Jutho/TensorOperations.jl) on expressions which this understands. (Disable with `tensor=false`.) These must be Einstein-convention contractions of one term; none of the examples above qualify.

* It uses [`KernelAbstractions.@kernel`](https://github.com/JuliaGPU/KernelAbstractions.jl) to make a GPU version. (Disable with `cuda=false`.) This is somewhat experimental, and may not be fast.

The macro also tries to provide a gradient for use with [Tracker](https://github.com/FluxML/Tracker.jl) or [Zygote](https://github.com/FluxML/Zygote.jl). <!-- or [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl). -->
(Disable with `grad=false`, or `nograd=A`.) This is done in one of two ways:

* By default it takes a symbolic derivative of the right hand side expression. When using `@tensor`, this writes another `@tensor` expression for each input array, otherwise it simply fills in all the gradient arrays at once. (Only for reductions over `+` or `min`/`max`.)

* The option `grad=Dual` uses instead [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) to differentiate the right hand side (only for reductions over `+`). This allows for more complicated expressions.

The expression need not be just one line, for example:

```julia
@tullio out[x, y] := @inbounds(begin  # sum over k
        a,b = off[k]
        mat[mod(x+a), mod(y+b)]
    end) (x in axes(mat,1), y in axes(mat,2)) grad=Dual nograd=off
```

Here the macro cannot infer the range of the output's indices `x,y`, so they must be provided explicitly.
(If writing into an existing array, with `out[x,y] = begin ...` or `+=`, then ranges would be taken from there.)
Because it sees assignment being made, it does not attempt to sum over `a,b`, and it assumes that indices could go out of bounds so does not add `@inbounds` for you. 
(Although in fact `mod(x+a) == mod(x+a, axes(mat,1))` is safe.)
It will also not be able to take a symbolic derivative, but dual numbers will work fine.

Pipe operators `|>` or `<|` indicate functions to be performed *outside* the sum, for example:

```julia
@tullio lse[j] := log <| exp(mat[i,j])   # vec(log.(sum(exp.(mat), dims=1))) 
```

The option `@tullio verbose=true` will cause it to print index ranges, symbolic derivatives,
and notices when it is unable to use the packages mentioned above. 
And `verbose=2` will print everything.

<details><summary><b>Notation</b></summary>

```julia
using Pkg; Pkg.add("Tullio")
using Tullio
A = [abs2(i - 11) for i in 1:21]

# Downsample -- range of i is that allowed by both terms:
@tullio D[i] := (A[2i] + A[2i+1])/2  # 1:10 == intersect(1:10, 0:10)

# Shifts -- range of i calculated in terms of that given for j:
@tullio M[i,j] := A[i+j-1]  (j in 1:15)  # i in 1:7
@tullio M[i+_,j] := A[i+j]  (j in 1:15)  # i in 0:6, automatic shift "i+_"

using OffsetArrays # Convolve a filter:
K = OffsetArray([1,-1,2,-1,1], -2:2)
@tullio C[i] := A[i+j] * K[j]    # j ∈ -2:2 implies i ∈ 3:19

# Index by the values in K
@tullio D[i,j] := A[2K[j]+i] ÷ K[j] # extrema(K)==(-1,2) implies i ∈ 3:17

# Wrapped & padded:
@tullio M[i,j] := A[mod(i+j)]  (j in 1:15, i in 1:15)   # wraps around, mod(i+j, axes(A,1))
@tullio M[i,j] := A[clamp(i+j)]  (j in 1:15, i in 1:15) # instead repeats "100"
@tullio M[i+_,j] := A[pad(i+j, 3)]  (j in 1:15)         # fills with zeros

using FFTW # Functions of the indices are OK:
S = [0,1,0,0, 0,0,0,0]
fft(S) ≈ @tullio F[k] := S[x] * exp(-im*pi/8 * (k-1) * x)  (k ∈ axes(S,1))

# Finalisers <| or |> are applied after sum (the two are equivalent):
@tullio N2[j] := sqrt <| M[i,j]^2     # N2 ≈ map(norm, eachcol(M)) 
@tullio n3[_] := A[i]^3  |> (_)^(1/3) # n3[1] ≈ norm(A,3), with _ anon. func.

# Reduction over any function:
@tullio (*) P[i] := A[i+k]  (k in 0:2) # product
@tullio (max) X[i,_] := D[i,j]         # maximum(D, dims=2), almost

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
<details><summary><b>Fast & slow</b></summary>

When used with LoopVectorization, on straightforward matrix multiplication of real numbers, 
`@tullio` tends to be about as fast as OpenBLAS. Depending on the size, and on your computer. 
Here's a speed comparison on mine: [v2.5](https://github.com/mcabbott/Tullio.jl/blob/master/benchmarks/02/matmul-0.2.5-Float64-1.5.0.png).

This is a useful diagnostic, but isn't really the goal. Two things `@tullio` is often
very fast at are weird tensor contractions (for which you'd need `permutedims`),
and broadcast-reductions (where it can avoid large allocations). For example:

```julia
using Tullio, LoopVectorization, NNlib, BenchmarkTools

# Batched matmul with batch index first in B, defined with @avx loops:
bmm_rev(A, B) = @tullio C[i,k,b] := A[i,j,b] * B[b,k,j]  # (sum over j)

A = randn(20,30,500); B = randn(500,40,30);
bmm_rev(A, B) ≈ NNlib.batched_mul(A, permutedims(B, (3,2,1))) # true

@btime bmm_rev($A, $B); # 317.526 μs, same speed as un-permuted bmm
@btime NNlib.batched_mul($A, permutedims($B, (3,2,1))); # 1.478 ms, with MKL

# Complete reduction, without first materialising X .* log.(Y')
sum_opp(X, Y=X) = @tullio s := X[i,j] * log(Y[j,i])

X = rand(1000,1000);
@btime sum_opp($X)                    #   499.814 μs (173 allocations: 14.20 KiB)
@btime sum($X .* log.(transpose($X))) # 8.759 ms (2 allocations: 7.63 MiB)
```

Complex numbers aren't handled by LoopVectorization, so will be much slower.
Repeated multiplication is also very slow, because it doesn't know there's a better
algorithm. It just makes 4 loops here instead of multiplying sequentially, 
`30^4` instead of `2 * 30^3` operations:

```julia
M1, M2, M3 = randn(30,30), randn(30,30), randn(30,30);
@btime $M1 * $M2 * $M3;                                   #  3.525 μs
@btime @tullio M4[i,l] := $M1[i,j] * $M2[j,k] * $M3[k,l]; # 30.401 μs
```

At present indices using `pad`, `clamp` or `mod` are also slow. These result in extra 
checks or operations at every iteration, not just around the edges:

```julia
conv1(x,k) = @tullio y[i+_, j+_] := x[i+a, j+b] + k[a,b]
conv2(x,k) = @tullio y[i+_, j+_] := x[2i+a, 2j+b] + k[a,b] avx=false
conv3(x,k) = @tullio y[i+_, j+_] := x[pad(i+a,3), pad(j+b,3)] + k[a,b] avx=false

x100 = rand(100,100); k7 = randn(7,7);
@btime conv1($x100, $k7); #  20.968 μs
@btime conv2($x100, $k7); #  156.768 μs
@btime conv3($x100, $k7); #  301.124 μs
```

</details>
<details><summary><b>Derivatives & GPU</b></summary>

```julia
using Tullio
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k] 

A = rand(3,40); B = rand(40,500);
A * B ≈ mul(A, B) # true

using Tracker # or Zygote
ΔA = Tracker.gradient((A,B) -> sum(mul(A, B)), A, B)[1]
ΔA ≈ ones(3,500) * B' # true

using CUDA, KernelAbstractions # Now defined with a GPU version:
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

cu(A * B) ≈ mul(cu(A), cu(B)) # true

cu(ΔA) ≈ Tracker.gradient((A,B) -> sum(mul(A, B)), cu(A), cu(B))[1] # true

# Reduction over min/max:
Tracker.gradient(x -> (@tullio (max) res := x[i]^3), [1,2,3,-2,-1,3])[1]
```

</details>
<details><summary><b>Larger expressions</b></summary>

```julia
using Tullio, OffsetArrays

# A convolution with cyclic indices
mat = zeros(10,10,1); mat[2,2] = 101; mat[10,10] = 1;
@tullio kern[i,j] := 1/(1+i^2+j^2)  (i in -3:3, j in -3:3)

@tullio out[x,y,c] := begin
    xi = mod(x+i, axes(mat,1)) # xi = ... means that it won't be summed,
    # yj = mod(y+j, axes(mat,2))
    @inbounds trunc(Int, mat[xi, mod(y+j), c] * kern[i,j]) # and disables automatic @inbounds,
end (x in 1:10, y in 1:10) # and prevents range of x from being inferred.

# A stencil?
offsets = [(a,b) for a in -2:2 for b in -2:2 if a>=b] # vector of tuples

@tullio out[x,y,1] = begin 
        a,b = offsets[k]
        i = clamp(x+a, extrema(axes(mat,1))...)
        # j = clamp(y+b, extrema(axes(mat,2))...) # can be written clamp(y+b)
        @inbounds mat[i, clamp(y+b), 1] * 10
    end # ranges of x,y read from out[x,y,1]

# Applying a vector of functions
fs = [sin, cos, tan]
xs = randn(3,100)
@tullio ys[r,c] := (fs[r])(xs[r,c])

using Zygote, ForwardDiff
rowmap(fs, xs) = @tullio ys[r,c] := (fs[r])(xs[r,c]) grad=Dual nograd=fs
Zygote.gradient(sum∘rowmap, fs, ones(3,2))
[f'(1) for f in fs] # agrees
```

</details>
<details><summary><b>Options</b></summary>

The default setting is:
```@tullio threads=true fastmath=true avx=true tensor=true cuda=256 grad=Base verbose=false A[i,j] := ...``` 
* `threads=false` turns off threading, while `threads=64^3` sets a threshold size at which to divide the work (replacing the macro's best guess).
* `avx=false` turns off the use of `LoopVectorization`, while `avx=4` inserts `@avx unroll=4 for i in ...`.
* `grad=false` turns off gradient calculation, and `grad=Dual` switches it to use `ForwardDiff` (which must be loaded).
* `nograd=A` turns of the gradient calculation just for `A`, and `nograd=(A,B,C)` does this for several arrays. 
* `tensor=false` turns off the use of `TensorOperations`.
* Assignment `xi = ...` removes `xi` from the list of indices: its range is note calculated, and it will not be summed over. It also disables `@inbounds` since this is now up to you.
* `verbose=true` prints things like the index ranges inferred, and gradient calculations. `verbose=2` prints absolutely everything.
* `A[i,j] := ...` makes a new array, while `A[i,j] = ...` and `A[i,j] += ...` write into an existing one. `A[row=i, col=j] := ...` makes a new `NamedDimsArray`.
* `@tullio (*) A[i,j] := ...` is a product, as is `@tullio A[i,j] *= ...`. For other reductions, `@tullio (f) A[i,j] ^= ...` is an in-place update.
* `init=0.0` gives the initial value for reductions. For `+`, `*`, `min`, `min`, `&`, `|` it has sensible defaults, for other reductions uses zero.

Implicit:
* Indices without shifts must have the same range everywhere they appear, but those with shifts (even `A[i+0]`) run over the intersection of possible ranges.
* Shifted output indices must start at 1, unless `OffsetArrays` is visible in the calling module.
* The use of `@avx`, and the calculation of gradients, are switched off by sufficiently complex syntax (such as arrays of arrays). 
* Gradient hooks are attached for any or all of `ReverseDiff`, `Tracker` & `Zygote`. These packages need not be loaded when the macro is run.
* Gradients are only defined for reductions over `(+)` (default) and `min`, `max`.
* GPU kernels are only constructed when both `KernelAbstractions` and `CUDA` are visible. The default `cuda=256` is passed to `kernel(CUDA(), 256)`.
* The CPU kernels from `KernelAbstractions` are called only when `threads=false`; they are not at present very fast, but perhaps useful for testing.

Extras:
* `A[i] := i^2  (i in 1:10)` is how you specify a range for indices when this can't be inferred. 
* `A[i] := B[i, $col] - C[i, 2]` is how you fix one index to a constant (to prevent `col` being summed over).
* `A[i] := $d * B[i]` is the preferred way to include other constants. Note that no gradient is calculated for `d`. 
* Within indexing, `A[mod(i), clamp(j)]` both maps `i` & `j` to lie within `axes(A)`, and disables inference of their ranges from `A`.
* Similarly, `A[pad(i,3)]` extends the range of `i`, inserting zeros outside of `A`. Instead of zero, `pad=NaN` uses this value as padding. The implementation of this (and `mod`, `clamp`) is not very fast at present.
* On the left, when making a new array, an underscore like `A[i+_] :=` inserts whatever shift is needed to make `A` one-based.
* `Tullio.@printgrad (x+y)*log(x/z)   x y z` prints out how symbolic derivatives will be done. 

</details>
<details><summary><b>Interals</b></summary>

The following three macros all end up calling the same functions as does `C = A * B`:

```julia
@tensor C[i,j] := A[i,k] * B[k,j]         # TensorOperations.jl
@ein C[i,j] := A[i,k] * B[k,j]            # OMEinsum.jl
@matmul C[i,j] := sum(k) A[i,k] * B[k,j]  # TensorCast.jl
```

But this one writes its own for-loops:

```julia
@einsum C[i,j] := A[i,k] * B[k,j]         # Einsum.jl
```

expanding out to roughly this:

```julia
T = promote_type(eltype(A), eltype(B))
C = Array{T}(undef, size(A,1), size(B,2))
@inbounds for j in 1:size(B,2)
    for i in 1:size(A,1)
        acc = zero(T)
        for k in 1:size(A,2)
            acc += A[i,k] * B[k,j]
        end
        C[i,j] = acc
    end
end
```

Tullio does something similar, but working through a few functions. Taking a slightly more complicated example, this:

```julia
@tullio C[i,j] := tanh <| A[i,k] * B[k,j]
```

expands to roughly this:

```julia
function act!(::Type, C::AbstractArray{T}, A, B, ax_i, ax_j, ax_k, keep=nothing, final=true) where T
    @inbounds @fastmath for i in ax_i
        for j in ax_j
            acc = isnothing(keep) ? zero(T) : C[i,j]
            for k in ax_k
                acc += A[i,k] * B[k,j]
            end
            C[i,j] = isnothing(final) ? acc : tanh(acc)
        end
    end
end

function make(A, B)
    ax_i = axes(A,1)
    ax_j = axes(B,2)
    ax_k = axes(A,2) # and check this is == axes(B,1)
    rhs(A,B,i,j,k) = tanh(A[i,k] * B[k,j])
    T = Core.Compiler.return_type(rhs, eltype.((A,B,1,1,1))) # plus a fallback
    C = similar(A, T, (ax_i, ax_j))
    Tullio.threader(act!, Array{T}, C, (A,B), (ax_i,ax_j), (ax_k,), +, 64^3)
    return C
end

C = Tullio.Eval(make, ∇make)(A, B)
```

This division allows it to dispatch to other methods of `act!`: one generated with `@avx` if LoopVectorization is loaded, and one for `::CuArray` if KernelAbstractions is loaded.

It also allows `threader` to divide the work, calling `act!` many times, from different threads, on small tiles made by dividing the longest axis (say `ax_i`) in half, repeatedly. If it divides up `ax_k`, these are done sequentially, with `keep=true` on all ranges except the first, and `final=nothing` on all except the last. But `ax_i` and `ax_j` are safe to do in parallel.

Finally, `Eval` exists to give Zygote and friends somewhere to attach themselves. The gradient calculation looks roughly like this:

```julia
@adjoint function (e::Eval)(AB...)
    C = e.fwd(AB...)
    C, ΔC -> e.rev(ΔC, C, AB...)
end

function ∇act!(::Type, ΔC, ΔA, ΔB, C, A, B, ax_i, ax_j, ax_k, keep)
    for k in ax_k, i in ax_i, j in ax_j
        ex = ΔC[i,j] * (1-C[i,j])^2
        ΔA[i,k] += ex * B[k,j]
        ΔB[k,j] += A[i,k] * ex
    end
end

function ∇make(ΔC, C, A, B)
    ΔA = similar(A) .= 0
    ΔB = similar(B) .= 0
    ax_i, ax_k = axes(A); ax_j = axes(B,2)
    Tullio.∇threader(∇act!, Array{T}, (ax_k,), (ax_i, ax_j), nothing)
    return (ΔA, ΔB)
end
```

In this case, it is the loop over `k` which can be safely broken among different threads, since both `ΔA` and `ΔB` have this index. Both `ΔA` and `ΔB` are filled in at once.

Notice that the derivative of `y = tanh(z)` has been written in terms of `y` (the final result of the forward pass) but free of `z` (the result of the sum, which was not saved). If this is not possible, it will fail.

If using `grad=Dual`, the gradient kernel looks different. This method cannot handle finalisers like `tanh` above, but for plain `@tullio C[i,j] := A[i,k] * B[k,j]` it would read:

```julia
function ∇act!(::Type, ΔC, ΔA, ΔB, C, A, B, ax_i, ax_j, ax_k, keep)
    eps1 = ForwardDiff.Dual(0, (1,0))
    eps2 = ForwardDiff.Dual(0, (0,1))
    for k in ax_k, i in ax_i, j in ax_j
        res = (A[i,k] + eps1) * (B[k,j] + eps2)
        ΔA[i,k] += ForwardDiff.partials(res, 1) * ΔC[i,j]
        ΔB[k,j] += ForwardDiff.partials(res, 2) * ΔC[i,j]
    end
end
```

Writing `@tullio verbose=2` will print all of these functions out. 

Scalar reductions, such as `@tullio s := A[i,j] * log(B[j,i])`, are slightly different in that the `act!` function simply returns the sum, i.e. the variable `acc` above.

</details>
<details><summary><b>Elsewhere</b></summary>

Back-end friends & relatives:

* [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) is used here, if available. 

* [Gaius.jl](https://github.com/MasonProtter/Gaius.jl) and [PaddedMatrices.jl](https://github.com/chriselrod/PaddedMatrices.jl) build on that.

* [GPUifyLoops.jl](https://github.com/vchuravy/GPUifyLoops.jl) and [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) generate GPU-compatible kernels.

* [ThreadsX.jl](https://github.com/tkf/ThreadsX.jl) does threaded reductions, and much else.

* [Strided.jl](https://github.com/Jutho/Strided.jl) does multi-threaded broadcasting.

Front-end near-lookalikes:

* [Einsum.jl](https://github.com/ahwillia/Einsum.jl) makes simple loops. See [tests/einsum.jl](https://github.com/mcabbott/Tullio.jl/blob/master/test/einsum.jl) where `using Tullio: @einsum` is an almost-seamless replacement.

* [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) and [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) identify patterns on which they can call various basic operations.

* [TensorCast.jl](https://github.com/mcabbott/TensorCast.jl) expresses everything as Julia array operations, broadcasting and reduction. (OMEinsum.jl also treats some cases as a special lazy broadcast-reduction.)

Things you can't run:

* [Tortilla.jl](https://www.youtube.com/watch?v=Rp7sTl9oPNI) seems to exist, publicly, only in this very nice talk. 

* [ArrayMeta.jl](https://github.com/shashi/ArrayMeta.jl) was a Julia 0.5 take on some of this.

* [Tokamak.jl](https://github.com/MikeInnes/Tokamak) was another, see [readme here](https://github.com/tkelman/Tokamak.jl).
