
using Tullio
cd(joinpath(dirname(pathof(Tullio)), "..", "benchmarks"))
using Pkg; pkg"activate ."

# or

using Pkg; pkg"add LoopVectorization#master"
using Pkg; pkg"add Strided Einsum IntelVectorMath ForwardDiff Tracker"


########## an example

f_base(A,B,C) = sum(A .* log.(B ./ C'), dims=2)

# using IntelVectorMath
# f_intel(A,B,C) = (z = IVM.log!(B ./ C'); sum(z .= A .* z, dims=2))

using Einsum
f_einsum(A,B,C) = @einsum s[i] := A[i] * log(B[i,j] / C[j])
f_vielsum(A,B,C) = @vielsum s[i] := A[i] * log(B[i,j] / C[j])

using Strided
f_strided(A,B,C) = @strided sum(A .* log.(B ./ C'), dims=2)

using Tullio, LoopVectorization
f_tullio(A,B,C) = @tullio s[i] := A[i] * log(B[i,j] / C[j])
f_avx(A,B,C) = sum(@avx A .* log.(B ./ C'); dims=2)


n = 200; A = rand(n); B = rand(n, 2n); C = rand(2n);

# f_base(A,B,C) ≈ f_intel(A,B,C)
f_base(A,B,C) ≈ f_einsum(A,B,C) ≈ f_vielsum(A,B,C)
f_base(A,B,C) ≈ f_strided(A,B,C)
f_base(A,B,C) ≈ f_tullio(A,B,C) ≈ f_avx(A,B,C)


########## times, forwards

julia> @btime f_base($A,$B,$C);
  972.948 μs (9 allocations: 627.03 KiB)

# julia> @btime f_intel($A,$B,$C);
#   184.631 μs (9 allocations: 627.03 KiB)

julia> @btime f_einsum($A,$B,$C);
  1.167 ms (1 allocation: 1.77 KiB)

julia> @btime f_vielsum($A,$B,$C); # 4 threads, laptop
  543.292 μs (24 allocations: 4.52 KiB)

julia> @btime f_strided($A,$B,$C);
  1.170 ms (14 allocations: 627.27 KiB)

julia> @btime f_tullio($A,$B,$C); # with avx, no threads yet
  241.869 μs (6 allocations: 1.86 KiB)

julia> @btime f_avx($A,$B,$C);
  179.732 μs (16 allocations: 627.17 KiB)

julia> f_tullio(A,B,C) = @tullio avx=false s[i] := A[i] * log(B[i,j] / C[j]);

julia> @btime f_tullio($A,$B,$C); # without avx
  1.215 ms (3 allocations: 1.80 KiB)

# julia> @btime create125($A, $B, $C); # avx + Threads.@spawn by hand
#   125.700 μs (308 allocations: 38.86 KiB)

julia> @btime create109($A,$B,$C); # version below, with unroll=4 (unroll=1 was same as above)
  180.128 μs (3 allocations: 1.80 KiB)

########## gradients

using Tracker, ForwardDiff
unfill(x) = x
unfill(x::TrackedArray) = Tracker.track(unfill, x)
Tracker.@grad unfill(x) = unfill(Tracker.data(x)), dx -> (collect(dx),) # deal with FillArrays

f_sym(A,B,C) = @tullio grad=Base avx=false s[i] := A[i] * log(B[i,j] / C[j]);
f_sym_avx(A,B,C) = @tullio grad=Base avx=true s[i] := A[i] * log(B[i,j] / C[j]);

Tracker.gradient(sum∘f_base, A, B, C)[1] ≈ Tracker.gradient(sum∘unfill∘f_sym, A, B, C)[1]
Tracker.gradient(sum∘f_base, A, B, C)[1] ≈ Tracker.gradient(sum∘unfill∘f_sym_avx, A, B, C)[1]

f_fwd(A,B,C) = @tullio grad=Dual avx=false s[i] := A[i] * log(B[i,j] / C[j]);
f_fwd_avx(A,B,C) = @tullio grad=Dual avx=false s[i] := A[i] * log(B[i,j] / C[j]);

using ForwardDiff: partials # some weird scope issue? only with avx

Tracker.gradient(sum∘f_base, A, B, C)[1] ≈ Tracker.gradient(sum∘unfill∘f_fwd, A, B, C)[1]
Tracker.gradient(sum∘f_base, A, B, C)[1] ≈ Tracker.gradient(sum∘unfill∘f_fwd_avx, A, B, C)[1]


########## gradient times

julia> @btime Tracker.gradient(sum∘f_base, $A, $B, $C);
  5.895 ms (240093 allocations: 12.22 MiB)

julia> @btime Tracker.gradient(sum∘f_sym, $A, $B, $C);
  2.874 ms (51 allocations: 633.92 KiB)

julia> @btime Tracker.gradient(sum∘f_fwd, $A, $B, $C);
  2.918 ms (51 allocations: 633.92 KiB)

julia> @btime Tracker.gradient(sum∘unfill∘f_sym_avx, $A, $B, $C);
  597.748 μs (46 allocations: 635.56 KiB)

julia> @btime Tracker.gradient(sum∘unfill∘f_fwd_avx, $A, $B, $C); # using "take I" definitions
  3.382 ms (180046 allocations: 16.18 MiB)

julia> @btime Tracker.gradient(sum∘unfill∘f_fwd_avx, $A, $B, $C); # using "take II" definitions
  27.043 ms (720346 allocations: 65.96 MiB)

julia> @btime Tracker.gradient(sum∘unfill∘f_fwd_avx, $A, $B, $C); # using "take III" definitions
  4.197 ms (180057 allocations: 16.18 MiB)

julia> @btime Tracker.gradient(sum∘unfill∘f_fwd_avx, $A, $B, $C); # using "take IV" definitions
  3.307 ms (180046 allocations: 16.18 MiB)

julia> @btime Tracker.gradient(sum∘unfill∘f_fwd_avx, $A, $B, $C); # after removing where Z, https://github.com/chriselrod/LoopVectorization.jl/issues/93
  1.830 ms (299 allocations: 2.49 MiB)
# much better, but still far from symbolic 597.748 μs

julia> @btime Tracker.gradient(sum∘unfill∘create109, $A, $B, $C); # below, currently identical?
  3.237 ms (180009 allocations: 16.18 MiB)

  1.029 ms (60046 allocations: 5.81 MiB) # if I comment out partials(res,d) lines
  246.807 μs (9 allocations: 633.69 KiB) # if I put C[j] = 𝛥C[j] + one(𝒯) instead
  5.672 ms (300009 allocations: 27.17 MiB) # using mypartials(res, Val(d))

########## code!

@tullio verbose=true grad=Dual s[i] := A[i] * log(B[i,j] / C[j]);
# using Tullio: storage_type
storage_type(As...) = Array{Float64}


function create109(A, B, C)
    local 📏i = axes(A, 1)
    @assert axes(A, 1) == axes(B, 1) "range of index i must agree"
    local 📏j = axes(C, 1)
    @assert axes(C, 1) == axes(B, 2) "range of index j must agree"
    🖐(A, B, C, i, j) = A[i] * log(B[i, j] / C[j])
    𝒯 = typeof(🖐(A, B, C, first(📏i), first(📏j)))
    s = similar(A, 𝒯, (📏i,))
    apply!109(s, storage_type(s, A, B, C), A, B, C, 📏i, 📏j)
    return s
end

function apply!109(ℛℰ𝒮::AbstractArray{𝒯}, ::Type, A, B, C, 📏i, 📏j) where 𝒯
    @inbounds begin
            nothing
            @fastmath for i = 📏i
                    𝒜 = zero(𝒯)
                    for j = 📏j
                        𝒜 = 𝒜 + A[i] * log(B[i, j] / C[j])
                    end
                    ℛℰ𝒮[i] = 𝒜
                end
            nothing
        end
end

function apply!109(ℛℰ𝒮::AbstractArray{𝒯}, ::Type{<:Array{<:Union{Float32, Float64, Int32, Int64, Int8}}}, A, B, C, 📏i, 📏j) where 𝒯
    @inbounds nothing
    # (LoopVectorization).@avx for i = 📏i
    (LoopVectorization).@avx unroll=4 for i = 📏i   # unroll=1 ok here, 4 is faster
            𝒜 = zero(𝒯)
            for j = 📏j
                𝒜 = 𝒜 + A[i] * log(B[i, j] / C[j])
            end
            ℛℰ𝒮[i] = 𝒜
        end
    nothing
end

create109(A::Tracker.TrackedArray, args...) = Tracker.track(create109, A, args...)
create109(A, B::Tracker.TrackedArray, args...) = Tracker.track(create109, A, B, args...)
create109(A::Tracker.TrackedArray, B::Tracker.TrackedArray, args...) = Tracker.track(create109, A, B, args...)

Tracker.@grad create109(args...) = (create109(Tracker.data.(args)...), (Δ->∇create109(Δ, Tracker.data.(args)...)))

function ∇create109(𝛥ℛℰ𝒮, A, B, C)
    𝛥A = fill!(similar(A), 0)
    𝛥B = fill!(similar(B), 0)
    𝛥C = fill!(similar(C), 0)
    📏i = axes(A, 1)
    📏j = axes(C, 1)
    ∇apply!109(𝛥A, 𝛥B, 𝛥C, storage_type(𝛥A, 𝛥B, 𝛥C, A, B, C), 𝛥ℛℰ𝒮, A, B, C, 📏i, 📏j)
    return (𝛥A, 𝛥B, 𝛥C)
end

function ∇apply!109(𝛥A, 𝛥B, 𝛥C, ::Type, 𝛥ℛℰ𝒮::AbstractArray{𝒯}, A, B, C, 📏i, 📏j) where 𝒯
    𝜀B = (ForwardDiff).Dual(zero(𝒯), (one(𝒯), zero(𝒯), zero(𝒯)))
    𝜀C = (ForwardDiff).Dual(zero(𝒯), (zero(𝒯), one(𝒯), zero(𝒯)))
    𝜀A = (ForwardDiff).Dual(zero(𝒯), (zero(𝒯), zero(𝒯), one(𝒯)))
    @fastmath @inbounds(for i = 📏i
                for j = 📏j
                    ℛℰ𝒮 = (A[i] + 𝜀A) * log((B[i, j] + 𝜀B) / (C[j] + 𝜀C))
                    𝛥B[i, j] = 𝛥B[i, j] + (ForwardDiff).partials(ℛℰ𝒮, 1) * 𝛥ℛℰ𝒮[i]
                    𝛥C[j] = 𝛥C[j] + (ForwardDiff).partials(ℛℰ𝒮, 2) * 𝛥ℛℰ𝒮[i]
                    𝛥A[i] = 𝛥A[i] + (ForwardDiff).partials(ℛℰ𝒮, 3) * 𝛥ℛℰ𝒮[i]
                end
            end)
end

function ∇apply!109(𝛥A, 𝛥B, 𝛥C, ::Type{<:Array{<:Union{Float32, Float64, Int32, Int64, Int8}}}, 𝛥ℛℰ𝒮::AbstractArray{𝒯}, A, B, C, 📏i, 📏j) where 𝒯
    𝜀B = (ForwardDiff).Dual(zero(𝒯), (one(𝒯), zero(𝒯), zero(𝒯)))
    𝜀C = (ForwardDiff).Dual(zero(𝒯), (zero(𝒯), one(𝒯), zero(𝒯)))
    𝜀A = (ForwardDiff).Dual(zero(𝒯), (zero(𝒯), zero(𝒯), one(𝒯)))
    (LoopVectorization).@avx for i = 📏i

    # (LoopVectorization).@avx unroll=1 for i = 📏i # UndefVarError: ####op#1311_ not defined

            for j = 📏j
                ℛℰ𝒮 = (A[i] + 𝜀A) * log((B[i, j] + 𝜀B) / (C[j] + 𝜀C))
                𝛥B[i, j] = 𝛥B[i, j] + (ForwardDiff).partials(ℛℰ𝒮, 1) * 𝛥ℛℰ𝒮[i]
                𝛥C[j] = 𝛥C[j] + (ForwardDiff).partials(ℛℰ𝒮, 2) * 𝛥ℛℰ𝒮[i]
                𝛥A[i] = 𝛥A[i] + (ForwardDiff).partials(ℛℰ𝒮, 3) * 𝛥ℛℰ𝒮[i]

                # 𝛥B[i, j] = 𝛥B[i, j] + ℛℰ𝒮.partials[1] * 𝛥ℛℰ𝒮[i]
                # 𝛥C[j] = 𝛥C[j] + ℛℰ𝒮.partials[2] * 𝛥ℛℰ𝒮[i]
                # 𝛥A[i] = 𝛥A[i] + ℛℰ𝒮.partials[3] * 𝛥ℛℰ𝒮[i] # LoadError: TypeError: in typeassert, expected Symbol, got Expr

                # part = ℛℰ𝒮.partials.values # LoadError: "Expression not recognized:\nℛℰ𝒮.partials.values"

                # part = getfield(ℛℰ𝒮, :partials) # MethodError: no method matching add_constant!(::LoopVectorization.LoopSet, ::QuoteNode, ::Int64)

                # 𝛥B[i, j] = 𝛥B[i, j] + part[1] * 𝛥ℛℰ𝒮[i]
                # 𝛥C[j] = 𝛥C[j] + part[2] * 𝛥ℛℰ𝒮[i]
                # 𝛥A[i] = 𝛥A[i] + part[3] * 𝛥ℛℰ𝒮[i]

                # 𝛥B[i, j] = 𝛥B[i, j] + mypartials(ℛℰ𝒮, Val(1)) * 𝛥ℛℰ𝒮[i]
                # 𝛥C[j] = 𝛥C[j] + mypartials(ℛℰ𝒮, Val(2)) * 𝛥ℛℰ𝒮[i]
                # 𝛥A[i] = 𝛥A[i] + mypartials(ℛℰ𝒮, Val(3)) * 𝛥ℛℰ𝒮[i]
            end
        end
end

s = create109(A, B, C)

Tracker.gradient(sum∘f_base, A, B, C)[1] ≈ Tracker.gradient(sum∘unfill∘create109, A, B, C)[1]


mypartials(x::Dual{Z,SVec{N,T},D}, m::Int) where {Z,N,T,D} = getfield(getfield(getfield(x, :partials), :values), m)
mypartials(x::Dual{Z,SVec{N,T},D}, ::Val{m}) where {Z,N,T,D,m} = getfield(getfield(getfield(x, :partials), :values), m)::SVec{N,T}
