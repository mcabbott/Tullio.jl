
# @pretty @tullio Z[i,j] = (A[i,j] * A[j,i] + B[i] * B[j])
tul1(Z,A,B) = begin
    local @inline(g(i, j, A, B) = begin
                    @inbounds A[i, j] * A[j, i] + B[i] * B[j]
                end)
    local T = eltype(Z)
    for j = axes(A, 2)
        for i = axes(A, 1)
            @inbounds Z[i, j] = g(i, j, A, B)
        end
    end
    Z
end

tul2(Z,A,B) = begin
    local @inline(g(i, j, A, B) = begin
                    @inbounds A[i, j] * A[j, i] + B[i] * B[j]
                end)
    local T = eltype(Z)

        for i = axes(A, 1) # reversed
    for j = axes(A, 2)
            @inbounds Z[i, j] = g(i, j, A, B)
        end
    end
    Z
end

A = rand(500,500); B = rand(500); Z = similar(A);

@btime tul1($Z, $A, $B);
@btime tul2($Z, $A, $B);

using TiledIteration

tul3(Z,A,B) = begin
    local @inline(g(i, j, A, B) = begin
                    @inbounds A[i, j] * A[j, i] + B[i] * B[j]
                end)
    local T = eltype(Z)
    # tiles = collect(TileIterator((axes(Z)), (250,250)))
    tiles = TileIterator((axes(Z)), (250,250))
    for t in tiles

            for i in t[1]
        for j in t[2]
                @inbounds Z[i, j] = g(i, j, A, B)
            end
        end
    end
    Z
end

@btime tul3($Z, $A, $B);










A = rand(200,200); B = rand(200); Z = zeros(200,200); W = zeros(200);
A = rand(2000,2000); B = rand(2000); Z = zeros(2000,2000); W = zeros(2000);


ein(Z,A,B) = @einsum Z[i,j] = (A[j,i] * B[i] / B[j])
cast(Z,A,B) = @cast Z[i,j] = (A[j,i] * B[i] / B[j])
tul(Z,A,B) = @tullio Z[i,j] = (A[j,i] * B[i] / B[j])

ein2(Z,A,B) = @vielsum Z[i,j] = (A[j,i] * B[i] / B[j])
cast2(Z,A,B) = @cast Z[i,j] = (A[j,i] * B[i] / B[j]) strided
tul2(Z,A,B) = @moltullio Z[i,j] = (A[j,i] * B[i] / B[j])


@btime ein($Z,$A,$B);
@btime cast($Z,$A,$B);
@btime tul($Z,$A,$B); # 18.017 ms

@btime ein2($Z,$A,$B);
@btime cast2($Z,$A,$B);
@btime tul2($Z,$A,$B);

tul3(Z,A,B) = @tullio Z[i,j] = (A[j,i] * B[i] / B[j]) {tile(1000),i,j}
tul4(Z,A,B) = @moltullio Z[i,j] = (A[j,i] * B[i] / B[j]) {tile(1000),i,j}

@btime tul3($Z,$A,$B); # 6.704 ms
@btime tul4($Z,$A,$B);

A = rand(3,3); B = 1:3; Z = zeros(3,3);

ein(Z,A,B)
tul(Z,A,B)
cast(Z,A,B)
tul2(Z,A,B)


using Tullio, Einsum, LinearAlgebra, TensorCast, GPUifyLoops, TiledIteration, OMEinsum

A = rand(1000,1000); C = rand(1000,1000); D = rand(1000,1000); Z = zeros(1000,1000);


einM(Z,A,C) = @einsum Z[i,j] = A[i,k] * C[k,j]
tulM(Z,A,C) = @tullio Z[i,j] = A[i,k] * C[k,j]  (+,k) {tile(1000),i,j,k}

@btime mul!($Z,$A,$C); # 21.581 ms
@btime einM($Z,$A,$C); # 1.744 s
@btime tulM($Z,$A,$C); # 582.245 ms

tulM2(Z,A,C) = @tullio Z[i,j] = A[i,k] * C[k,j]  (+,unroll, k) {tile(1000),i,j,k}
@btime tulM2($Z,$A,$C); # 572.245 ms

tulM3(Z,A,C) = @tullio Z[i,j] = A[i,k] * C[k,j]  (+,unroll, k) {tile(8000),i,j,k}
@btime tulM3($Z,$A,$C); # 582.245 ms


A = rand(100,100); B = rand(100,100); C = rand(100,100); W = rand(100,100,100);


einS(W,A,B,C) = @einsum W[i,j,l] = A[i,k] * B[k,j] * C[l,k]
tilS(W,A,B,C) = @tullio W[i,j,l] = A[i,k] * B[k,j] * C[l,k] {tile(10000),i,j,k,l}
# omeS(W,A,B,C) = @ein W[i,j,l] = A[i,k] * B[k,j] * C[l,k]
omeS(W,A,B,C) = OMEinsum.einsumexp!(((1,3),(2,3),(3,4)),(A,B,C),(1,2,4),W)
omeS(W,A,B,C) = W .= OMEinsum.einsum(((1,3),(2,3),(3,4)),(A,B,C),(1,2,4))


@time einS(W,A,B,C); # 0.11
@time tilS(W,A,B,C); # 0.08  # best case 30% better
@time omeS(W,A,B,C); # 0.41

# @btime einS($W, $A, $B, $C); #
# @btime tilS($Z, $A, $B, $C); #
