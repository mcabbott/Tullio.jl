# Tullio.jl

This is a sketch of a replacement for [Einsum.jl](https://github.com/ahwillia/Einsum.jl), 
which similarly generates loops over indices:

```
julia> @pretty  @tullio A[i,j] := B[i] * log(C[j]) 
begin
    local @inline rhs(i, j, B, C) = @inbounds B[i] * log(C[j])
    local T = typeof(rhs(1, 1, B, C))
    A = Array{T, 2}(undef, length(axes(B, 1)), length(axes(C, 1)))
    for j in axes(C, 1)
        for i in axes(B, 1)
            @inbounds A[i, j] = rhs(i, j, B, C)
        end
    end
    A
end
```

It exists to experiment with various things. 
First, you can explicitly unroll loops (using [GPUifyLoops.jl](https://github.com/vchuravy/GPUifyLoops.jl))

```
julia> @pretty  @tullio A[i] := B[i,j]  (+, unroll(10), j)
┌ Warning: can't unroll loops on Julia 1.1.0
└ @ Tullio ~/.julia/dev/Tullio/src/Tullio.jl:86
...
            σ = zero(T)
            @unroll 10 for j in axes(B, 2)
                σ = σ + rhs(i, j, B)
            end
            A[i] = σ
...
```

The tuple `(+,i,j)` also lets you specify the reduction function,
and the order of loops. 

Second, you can access things in tiled order (using [TiledIteration.jl](https://github.com/JuliaArrays/TiledIteration.jl))

```
julia> @pretty  @tullio A[i,j] := B[j,i]  {tile(1024),i,j} 
...
    local tiles = collect(TileIterator((axes(B, 2), axes(B, 1)), (32, 32)))
    for tile in tiles
        for j in tile[2]
            for i in tile[1]
                @inbounds A[i, j] = rhs(j, i, B)
            end
        end
    end
...
```

Third, there is multi-threading, which is a little smarter about accumulation space 
than `@vielsum`:

```
julia> @pretty  @tullio A[i] := B[i,j]  (+,j)  {threads}
...
    local cache = Vector{T}(undef, nthreads())
    @threads for i in axes(B, 1)
...
            cache[threadid()] = zero(T)
            for j in axes(B, 2)
                cache[threadid()] = cache[threadid()] + rhs(i, j, B)
            end
            A[i] = cache[threadid()]
...
```

Fourth, the expression on the right need not be just simple arrays, 
and the output element type `T` will be calculated from this expression:

```
julia> @pretty @tullio A[i,_,j] := B.field[C[i]] + exp(D[i].field[j])
...
    T = typeof(rhs(1, 1, C, B, D))
...
    @assert axes(C, 1) == axes(D, 1) "range of index i must agree"
    for j = axes((first(D)).field, 1)
        for i = axes(C, 1)
            @inbounds A[i, 1, j] = rhs(i, j, C, B, D)
        end
    end
...
```

And finally, I'm trying to learn how to make it write loops for `CuArrays`
(also using [GPUifyLoops.jl](https://github.com/vchuravy/GPUifyLoops.jl)):

```
julia> @pretty @tullio A[i,j] = B[i,j,k] {gpu}
...
    function kernel!(A, B)
        GPUifyLoops.@loop for j = (axes(B, 2); threadIdx().z)
...
        @synchronize
        nothing
    end
    function kernel!(A::CuArray, B)
        GPUifyLoops.launch(CUDA(), kernel!, A, B; threads=(axes(B, 3), axes(B, 1), axes(B, 2)))
    end
    kernel!(A, B)
...
```

This isn't registered, so install like so:

```
] add https://github.com/mcabbott/Tullio.jl
```

## Elsewhere 

Used:
* https://github.com/vchuravy/GPUifyLoops.jl
* https://github.com/JuliaArrays/TiledIteration.jl
* https://github.com/MikeInnes/MacroTools.jl

Related:
* https://github.com/ahwillia/Einsum.jl (writes simple loops)
* https://github.com/Jutho/TensorOperations.jl (reduces to BLAS etc)
* https://github.com/mcabbott/TensorCast.jl (re-writes to broadcasting)

Being born:
* https://github.com/under-Peter/OMEinsum.jl (aims to be differentiable)

Dead:
* https://github.com/shashi/ArrayMeta.jl ()
* https://github.com/tkelman/Tokamak.jl (this fork has the readme!)
