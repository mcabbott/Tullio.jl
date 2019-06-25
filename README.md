# Tullio.jl

This is a sketch of a replacement for `Einsum.jl`, which similarly generates 
loops over indices:

```julia
julia> @pretty  @tullio A[i,j] := B[i] * log(C[j]) 
begin
    local @inline rhs(i, j, B, C) = @inbounds B[i] * log(C[j])
    local T = typeof(rhs(first(axes(B, 1)), first(axes(C, 1)), B, C))
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
First, you can explicitly unroll loops (using GPUifyLoops.jl)

```julia
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

Second, you can access things in tiled order (using TiledIteration.jl):

```julia
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

```julia
julia> @pretty  @moltullio A[i] := B[i,j]  (+,j)
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



Alive:
* https://github.com/cstjean/Unrolled.jl
* https://github.com/vchuravy/GPUifyLoops.jl
* https://github.com/JuliaArrays/TiledIteration.jl

Dead:
* https://github.com/shashi/ArrayMeta.jl
* https://github.com/tkelman/Tokamak.jl (this fork has the readme!)

At work:
* https://github.com/ahwillia/Einsum.jl (writes simple loops)
* https://github.com/Jutho/TensorOperations.jl (reduces to BLAS etc)
* https://github.com/mcabbott/TensorCast.jl (re-writes to broadcasting)

Being born:
* https://github.com/under-Peter/OMEinsum.jl
