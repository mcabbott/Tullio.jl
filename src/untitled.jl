
# Trying out OMEinsum's benchmark... 

using Tullio, GPUifyLoops, TiledIteration, TensorOperations, TensorCast

d = 5; χ = 10; 
a = randn(χ,χ); b= randn(χ,d,χ); c = randn(d,d,d,d);

## Loops
tt1(a,b,c) = @tullio z[l,n,o,p] := a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p]  (+,x,y,k,m)
tt2(a,b,c) = @tullio z[l,n,o,p] := a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p]  (+,x,y,unroll,k,m)
tt3(a,b,c) = @tullio z[l,n,o,p] := a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p]  (+,x,y,k,m) {tile(2000),l,n}
tt4(a,b,c) = @moltullio z[l,n,o,p] := a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p]

ee(a,b,c) = @einsum z[l,n,o,p] := a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p]
ee2(a,b,c) = @vielsum z[l,n,o,p] := a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p]

ee(a,b,c) ≈ tt1(a,b,c) ≈ tt2(a,b,c) ≈ tt3(a,b,c)

@time ee(a,b,c);
@time tt1(a,b,c);
@time tt2(a,b,c);
@time tt3(a,b,c);
@time tt4(a,b,c);

## confusion
# julia> @time @einsum z[l,n,o,p] := a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p];
# julia> @time @tullio z[l,n,o,p] := a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p];

## TC

cc1(a,b,c) = @reduce z[l,n,o,p] := sum(x,y,k,m) a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p] 
cc2(a,b,c) = @reduce z[l,n,o,p] := sum(x,y,k,m) a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p] lazy

@time cc1(a,b,c);
@time cc2(a,b,c);

## TO
ff1(a,b,c) = @tensor z[l,n,o,p] := a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p]
ff2(a,b,c) = @tensoropt z[l,n,o,p] := a[x,y] * b[x,k,l] * b[y,m,n] * c[k,m,o,p]

ee(a,b,c) ≈ ff1(a,b,c) ≈ ff2(a,b,c)

@time ff1(a,b,c);
@time ff2(a,b,c);

## OME
mm(a,b,c) = ein"xy,xkl,ymn,kmop->lnop"(a,b,b,c)

ee(a,b,c) ≈ mm(a,b,c)
@time mm(a,b,c);

# oo(a,b,c) = einsumopt((('x','y'), ('x','k','l'), ('y','m','n'), ('k','m','o','p')), (a, b, b, c), ('l','n','o','p'))

## Bigger?

d = 5; χ = 30; 
a = randn(χ,χ); b= randn(χ,d,χ); c = randn(d,d,d,d);

@time ee(a,b,c);
@time ee2(a,b,c);

@time tt1(a,b,c);
@time tt2(a,b,c);
@time tt3(a,b,c);
@time tt4(a,b,c);

@time mm(a,b,c);

@time ff1(a,b,c);
@time ff2(a,b,c);

@time cc2(a,b,c);

julia> d = 5; χ = 30;

julia> a = randn(χ,χ); b= randn(χ,d,χ); c = randn(d,d,d,d);

julia> @time ee(a,b,c);
  0.570150 seconds (7 allocations: 176.094 KiB)

julia> @time ee2(a,b,c);
  2.565833 seconds (22.53 k allocations: 530.328 KiB)

julia> @time tt1(a,b,c);
  0.582443 seconds (7 allocations: 176.094 KiB, 1.57% gc time)

julia> @time tt2(a,b,c);
  0.573431 seconds (7 allocations: 176.094 KiB)

julia> @time tt3(a,b,c);
  0.576748 seconds (8 allocations: 176.203 KiB)

julia> @time tt4(a,b,c);
  0.794736 seconds (40 allocations: 179.094 KiB)

julia> @time mm(a,b,c);
  4.638845 seconds (75 allocations: 179.313 KiB)

julia> @time ff1(a,b,c);
  0.001405 seconds (234 allocations: 193.297 KiB)

julia> @time ff2(a,b,c);
  0.000747 seconds (231 allocations: 193.234 KiB)

julia> @time cc2(a,b,c);
  1.760996 seconds (80 allocations: 255.125 KiB)



# So the lesson here is that fancy stuff with unrolling, tiles & threads doesn't matter at all here. I'm a bit surprised!
