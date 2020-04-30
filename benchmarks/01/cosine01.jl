
using Tullio
cd(joinpath(dirname(pathof(Tullio)), "..", "benchmarks", "01"))
using Pkg; pkg"activate ."

# or

using Pkg; pkg"add LoopVectorization"
using Pkg; pkg"add TensorCast NNlib ForwardDiff Zygote"


# This example is a nice usage:
# https://discourse.julialang.org/t/help-to-improve-performance-of-gradient-calculation-on-tensor-operations/37773/3

using TensorCast

function cosinesim(a, b)
    @reduce similarity[i, j, k] := sum(s) a[s, j, k] * b[s, i, k] /
        sqrt( @reduce [_, j, k] := sum(s') a[s', j, k]^2) /
        sqrt( @reduce [_, i, k] := sum(s'') b[s'', i, k]^2)
end

function mysoftmax(a)
    @cast submax[i, j, k] := a[i, j, k] - @reduce [_, j, k] := maximum(i) a[i, j, k]
    @cast r[i, j, k] := exp(submax[i, j, k]) / @reduce [_, j, k] := sum(i) exp(submax[i, j, k])
end

pairwise2(a) = mysoftmax(cosinesim(a))

N, W, R, B = 16, 64, 4, 16;
a = rand(Float32, W, R, B);
b = rand(Float32, W, N, B);

#=
First look just at the numerator. This should be easy with `NNlib.batched_mul`,
but there is [this issue](https://github.com/FluxML/Zygote.jl/issues/552)
so I'm using [my PR](https://github.com/FluxML/NNlib.jl/pull/191)...
but the end result is a gradient 4x faster.

Almost as fast as Tullio! (which is faster without multi-threading today, something is broken)
=#

using Zygote, NNlib

reducemul(a, b) = @reduce z[i, j, k] := sum(s) a[s, j, k] * b[s, i, k]
reducemul(a, b) ≈ batched_mul(batched_transpose(b), a) # true

@btime reducemul($a, $b); # 98.262 μs (52 allocations: 342.41 KiB)
@btime batched_mul(batched_transpose($b), $a); # 13.605 μs (5 allocations: 4.20 KiB) -> 6.447 μs (2 allocations: 4.14 KiB) with MKL

using Tullio, LoopVectorization, ForwardDiff

tullio_mul(a, b) = @tullio z[i, j, k] := a[s, j, k] * b[s, i, k]
tullio_mul2(a, b) = @tullio z[i, j, k] := a[s, j, k] * b[s, i, k]  grad=Dual
reducemul(a, b) ≈ tullio_mul(a, b) # true

@btime tullio_mul($a, $b); # 2.696 μs (1 allocation: 4.13 KiB) -- improved!

#=
For gradients, I've inserted `(x->x).()` into these because sum creates a `FillArray`
which some methods don't like, but any useful calculation would have something else after this.

Now not needed for @tullio!
=#

grad_a = gradient((a,b) -> sum(reducemul(a, b)), a, b)[1];
grad_a ≈ gradient((a,b) -> sum((x->x), batched_mul(PermutedDimsArray(b, (2,1,3)), a)), a, b)[1] # true
grad_a ≈ gradient((a,b) -> sum(tullio_mul(a, b)), a, b)[1] # true
grad_a ≈ gradient((a,b) -> sum(tullio_mul2(a, b)), a, b)[1] # true

@btime gradient((a,b) -> sum(reducemul(a, b)), $a, $b);       # 307.836 μs (193 allocations: 1.17 MiB)
@btime gradient((a,b) -> sum(x->x, reducemul(a, b)), $a, $b); # 378.264 μs (3316 allocations: 1.23 MiB)

@btime gradient((a,b) -> sum(x->x, batched_mul(PermutedDimsArray(b, (2,1,3)), a)), $a, $b); # 101.782 μs (3159 allocations: 151.52 KiB) -> 218.180 μs (3334 allocations: 161.70 KiB) MKL?

@btime gradient((a,b) -> sum(tullio_mul(a, b)), $a, $b);       # 21.190 μs (41 allocations: 85.89 KiB)
@btime gradient((a,b) -> sum(x->x, tullio_mul(a, b)), $a, $b); # 79.291 μs (3156 allocations: 151.19 KiB)

# version with ForwardDiff:
@btime gradient((a,b) -> sum(tullio_mul2(a, b)), $a, $b);      # 25.168 μs (37 allocations: 85.70 KiB)

#=
Tullio ought to make it easy to fuse that multiplication with the division, perhaps?
But this is slow!
=#

function cosine_nnlib(a, b)
    @reduce den1[j, k] := sum(s) a[s, j, k]^2
    @reduce den2[i, k] := sum(s) b[s, i, k]^2
    bmm = batched_mul(PermutedDimsArray(b, (2,1,3)), a)
    @cast similarity[i, j, k] := bmm[i, j, k] / sqrt(den1[j, k] * den2[i, k])
end

function cosine_fused(a, b)
    @tullio den1[j, k] := a[s, j, k]^2
    @tullio den2[i, k] := b[s, i, k]^2
    @tullio similarity[i, j, k] := a[s, j, k] * b[s, i, k] / sqrt(den1[j, k] * den2[i, k])
end

function cosine_separated(a, b)
    @tullio bmm[i, j, k] := a[s, j, k] * b[s, i, k]

    @tullio den1[j, k] := a[s, j, k]^2
    @tullio den2[i, k] := b[s, i, k]^2

    @tullio f1[j, k] := 1/sqrt(den1[j, k])
    @tullio f2[i, k] := 1/sqrt(den2[i, k])
    @tullio similarity[i, j, k] := bmm[i, j, k] * f1[j, k] * f2[i, k]
end

cosinesim(a, b) ≈ cosine_nnlib(a, b) # true
cosinesim(a, b) ≈ cosine_fused(a, b) # true
cosinesim(a, b) ≈ cosine_separated(a, b) # true

@btime cosinesim($a, $b);    # 318.341 μs (87 allocations: 425.14 KiB)
@btime cosine_nnlib($a, $b); #  34.407 μs (42 allocations: 91.20 KiB) -> 112.308 μs (179 allocations: 99.05 KiB) MKL
@btime cosine_fused($a, $b);   #  7.678 μs (3 allocations: 5.59 KiB)
@btime cosine_separated($a, $b);   #  4.672 μs (6 allocations: 11.19 KiB)

grad_a2 = gradient((a,b) -> sum(cosinesim(a, b)), a, b)[1];
grad_a2 ≈ gradient((a,b) -> sum(cosine_nnlib(a, b)), a, b)[1] # true
grad_a2 ≈ gradient((a,b) -> sum(cosine_fused(a, b)), a, b)[1] # true
grad_a2 ≈ gradient((a,b) -> sum(cosine_separated(a, b)), a, b)[1] # true

@btime gradient((a,b) -> sum(cosinesim(a, b)), $a, $b);    #  984.351 μs (1414 allocations: 3.51 MiB)
@btime gradient((a,b) -> sum(cosine_nnlib(a, b)), $a, $b); #  232.991 μs (6375 allocations: 579.78 KiB) -> 280.876 μs (3442 allocations: 520.61 KiB) MKL
@btime gradient((a,b) -> sum(cosine_fused(a, b)), $a, $b);     # 80.207 μs (129 allocations: 252.86 KiB)
@btime gradient((a,b) -> sum(cosine_separated(a, b)), $a, $b); # 80.367 μs (207 allocations: 267.08 KiB)

# cosine_fused might end up computing / sqrt N^3 times, instead of 2N^2 times,
# so it's not obviously a great idea.
# But I like LoopVectorization is smart enough to avoid that? Gradient is as fast now.

# Its gradient now looks like this:

Tullio.@printgrad x/sqrt(y*z)   x y z


#=
Aside, things about softmax:
=#

mysoftmax(a) ≈ softmax(a, dims=1)

@btime mysoftmax($a);       # 77.517 μs (37 allocations: 50.17 KiB)
@btime softmax($a, dims=1); # 45.405 μs (8 allocations: 33.03 KiB)

@btime gradient(a -> sum(mysoftmax(a)), $a);      # 640.619 μs
@btime gradient(a -> sum(softmax(a, dims=1)), $a); # 95.622 μs
@btime gradient(a -> sum(x->x, mysoftmax(a)), $a);      #
@btime gradient(a -> sum(x->x, softmax(a, dims=1)), $a); # 322.103 μs (12358 allocations: 356.17 KiB)

# add avx to versions from https://github.com/FluxML/NNlib.jl/pull/135/
function NNlib.softmax(xs::Array; dims=1)
    max_ = maximum(xs, dims=dims)
    out = @avx exp.(xs .- max_)
    @avx out .= out ./ sum!(max_, out)
end

@btime softmax($a, dims=1); # 16.084 μs -- 3x quicker

function NNlib.∇softmax(Δ::Array, xs::Array; dims=1)
    sf = softmax(xs, dims=dims)
    @avx sf .* (Δ .- sum(Δ .* sf; dims=dims))
end

@btime gradient(a -> sum(x->x, softmax(a, dims=1)), $a); # 260.441 μs (12358 allocations: 323.36 KiB)
# also 3x quicker, once you subtract the big cost of broadcasting (x->x)!

using Tracker # quicker at broadcasting, not sure it has NNlib gradient at all

@btime Tracker.gradient(a -> sum(mysoftmax(a)), $a);       # 324.806 μs
@btime Tracker.gradient(a -> sum(softmax(a, dims=1)), $a); # 252.686 μs


using ReverseDiff

@btime ReverseDiff.gradient(a -> sum(mysoftmax(a)), $a); # 6.359 ms

