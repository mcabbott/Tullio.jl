
# This is a batched matrix multiplication test, with the batch index coming first,
# from this thread:
# https://discourse.julialang.org/t/non-matching-indices-error-using-tensoroperations/35136
# Some improvements to OMEinsum since then make it much quicker.

julia> using Einsum, OMEinsum, TensorCast, Tullio, LoopVectorization

julia> f_ome(a,b) = @ein c[k, n] := a[k, n, c] * conj(b)[c,k]; # permutedims + batched_mul

julia> f_cast(a,b) = @reduce c[k, n] := sum(l) a[k, n, l] * conj(b[l, k]); # broadcasting

julia> f_ein(a,b) = @einsum c_[k, n] := a[k, n, c] * conj(b[c,k]); # naiive loops

julia> f_viel(a,b) = @vielsum c_[k, n] := a[k, n, c] * conj(b[c,k]); # plus threads

julia> f_tul(a,b) = @tullio c[k, n] := a[k, n, c] * conj(b[c,k]); # less naiive loops?

julia> a = randn(ComplexF64, 300, 400, 500); b = randn(ComplexF64, 500, 300);

julia> f_ome(a,b) ≈ f_cast(a,b) ≈ f_ein(a,b) ≈ f_tul(a,b)
true

julia> @btime f_ome($a, $b);
  342.619 ms (104 allocations: 921.49 MiB)

julia> @btime f_cast($a, $b);
  557.428 ms (26 allocations: 919.65 MiB)

julia> @btime f_ein($a, $b);
  264.429 ms (2 allocations: 1.83 MiB)

julia> @btime f_viel($a, $b);
  147.739 ms (25 allocations: 1.83 MiB)

julia> @btime f_tul($a, $b);
  191.859 ms (836 allocations: 1.86 MiB)

# But LoopVectorization isn't yet in play, as the arrays have complex elements.
# On real numbers, it makes a big difference!

julia> ar = real(a); br = real(b);

julia> @btime f_ome($ar, $br);
  190.672 ms (102 allocations: 459.60 MiB)

julia> @btime f_ein($ar, $br);
  122.616 ms (2 allocations: 937.58 KiB)

julia> @btime f_viel($ar, $br);
  82.922 ms (25 allocations: 940.33 KiB)

julia> @btime f_tul($ar, $br);
  39.105 ms (836 allocations: 967.17 KiB)

# Can we get there with StructArrays?
# (Perhaps this could be made automatic, with yet more macrology...)

julia> using StructArrays, LoopVectorization

julia> @time sa = StructArray(a); sb = StructArray(b);
  4.160518 seconds (120.00 M allocations: 3.576 GiB, 29.61% gc time)

julia> function f_tul(a::StructArray, b::StructArray)
           a_re, a_im = a.re, a.im
           b_re, b_im = b.re, b.im
           @tullio c_re[k, n] := a_re[k, n, c] * b_re[c,k] + a_im[k, n, c] * b_im[c,k]
           @tullio c_im[k, n] := a_im[k, n, c] * b_re[c,k] - a_re[k, n, c] * b_im[c,k]
           StructArray{eltype(a)}((c_re, c_im))
       end
f_tul (generic function with 2 methods)

julia> f_ein(sa, sb) ≈ f_tul(sa, sb)
true

julia> @btime f_tul($sa, $sb);
  136.801 ms (1670 allocations: 1.89 MiB)

# That's worse than twice the real calculation, but I guess each line is harder.
# Converting to StructArrays is really slow though!

julia> @btime StructArray($a);
  3.068 s (119999498 allocations: 3.58 GiB)

julia> @btime real($a);
  248.114 ms (2 allocations: 457.76 MiB)

# Compare Einsum on StructArrays: not great, however you do it.

julia> @btime f_viel($sa, $sb);
  479.932 ms (25 allocations: 1.83 MiB)

julia> typeof(f_viel(sa, sb))
Array{Complex{Float64},2}

julia> similar(sa, 1,3)
1×3 StructArray(::Array{Float64,2}, ::Array{Float64,2}) with eltype Complex{Float64}:
 2.29092e-314+2.27751e-314im  2.29092e-314+3.7518e-314im  3.7518e-314+3.75184e-314im

julia> similar(sa, ComplexF64, 1,3)
1×3 Array{Complex{Float64},2}:
 2.66668e-314+2.66668e-314im  2.66668e-314+2.29246e-314im  2.29246e-314+2.27751e-314im

julia> f_viel!(c_,a,b) = @vielsum c_[k, n] = a[k, n, c] * conj(b[c,k]);

julia> sc = similar(f_tul(sa, sb)); typeof(sc)
StructArray{Complex{Float64},2,NamedTuple{(:re, :im),Tuple{Array{Float64,2},Array{Float64,2}}},Int64}

julia> @btime f_viel!($sc, $sa, $sb);
  478.993 ms (240023 allocations: 5.50 MiB)

# Try KernelAbstractions? This should handle threading instead of my code.

julia> using KernelAbstractions, CuArrays # CuArrays just to trigger it!

julia> ENV["JULIA_DEBUG"] = Main;

julia> f_tul2(a,b) = @tullio c[k, n] := a[k, n, c] * conj(b[c,k])  threads=false;

julia> f_tul2(a, b);
┌ Debug: KernelAbstractions CPU actor:
│   typeof.(tuple(ℛ::AbstractArray{𝒯}, a, b, 𝒶_n, 𝒶_k, 𝒶_c)) = (Array{Complex{Float64},2}, Array{Complex{Float64},3}, Array{Complex{Float64},2}, UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64})
└ @ Main ~/.julia/dev/Tullio/src/macro.jl:724

julia> ENV["JULIA_DEBUG"] = "none";

julia> @btime f_tul2($a, $b);
  599.003 ms (64 allocations: 1.84 MiB)

# Note that if you just run f_tul threads=false without KernelAbstractions, it takes 1.3sec,
# which is much much slower than @einsum, so perhaps that's another bug.


#########################

julia> using StructArrays

julia> a = randn(ComplexF64, 300, 400, 500);

julia> @time StructArray(a);
  4.967282 seconds (120.00 M allocations: 3.576 GiB, 21.80% gc time)

julia> @time StructArray{ComplexF64}((real(a), imag(a)));
  0.630163 seconds (7 allocations: 915.528 MiB, 10.91% gc time)

julia> @code_warntype StructArray(a)
Variables
  #self#::Type{StructArray}
  v::Array{Complex{Float64},3}
  #46::StructArrays.var"#46#48"

Body::StructArray{Complex{Float64},3,NamedTuple{(:re, :im),Tuple{Array{Float64,3},Array{Float64,3}}},Int64}
1 ─      (#46 = %new(StructArrays.:(var"#46#48")))
│   %2 = #46::Core.Compiler.Const(StructArrays.var"#46#48"(), false)
│   %3 = StructArrays.:(var"#StructArray#45")(%2, #self#, v)::StructArray{Complex{Float64},3,NamedTuple{(:re, :im),Tuple{Array{Float64,3},Array{Float64,3}}},Int64}
└──      return %3
