# Some quick and dirty broadcasting AD benchmarks, 25 April 2020

julia> using Zygote, Tracker, ReverseDiff, Tullio

julia> x1 = randn(10); f1(x) = 1+tanh(x); # A simple function which isn't hard-coded

julia> Tracker.gradient(x -> sum(f1.(x)), x1)[1]'
Tracked 1×10 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.333674  0.174402  0.786548  0.222385  …  0.143003  0.361015  0.910887  0.940078

julia> Zygote.gradient(x -> sum(f1.(x)), x1)[1]'
1×10 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.333674  0.174402  0.786548  0.222385  …  0.143003  0.361015  0.910887  0.940078

julia> ReverseDiff.gradient(x -> sum(f1.(x)), (x1,))[1]'
1×10 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.333674  0.174402  0.786548  0.222385  …  0.143003  0.361015  0.910887  0.940078

julia> Tracker.gradient(x -> (@tullio s = 1+tanh(x[i])), x1)[1]'
Tracked 1×10 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.333674  0.174402  0.786548  0.222385  …  0.143003  0.361015  0.910887  0.940078

julia> Zygote.gradient(x -> sum(@tullio s[i] := 1+tanh(x[i])), x1)[1]'
1×10 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.333674  0.174402  0.786548  0.222385  …  0.143003  0.361015  0.910887  0.940078

# Time that on a bigger array:
# right now Zygote is very slow, and ReverseDiff worse.

julia> x2 = randn(1000,1000);

julia> @btime sum(f1.($x2));
  25.181 ms (2 allocations: 7.63 MiB)

julia> @btime sum(f1, $x2);
  24.414 ms (0 allocations: 0 bytes)

julia> @btime sum(Broadcast.broadcasted(f1, $x2)) # on Julia 1.5
  29.678 ms (0 allocations: 0 bytes)

julia> @btime @tullio s = f1($x2[i,j])
  13.195 ms (41 allocations: 2.66 KiB)

julia> @btime Tracker.gradient(x -> sum(f1.(x)), $x2);
  58.780 ms (44 allocations: 53.41 MiB)

julia> @btime Zygote.gradient(x -> sum(f1.(x)), $x2);
  119.821 ms (3000046 allocations: 122.07 MiB)

julia> @btime ReverseDiff.gradient(x -> sum(f1.(x)), ($x2,));
  941.047 ms (15000027 allocations: 567.06 MiB)

julia> @btime Tracker.gradient(x -> (@tullio s = 1+tanh(x[i,j])), $x2);
  33.391 ms (88 allocations: 30.52 MiB)

julia> @btime Zygote.gradient(x -> sum(@tullio s[i] := 1+tanh(x[i,j])), $x2);
  29.928 ms (84 allocations: 7.64 MiB)

julia> Tullio.@printgrad 1+tanh(x) x
δx = 1 - tanh(x) ^ 2

# Compare to tanh, which is a special case of Zygote's:

julia> @btime Zygote.gradient(x -> sum(tanh.(x)), $x2);
  25.633 ms (20 allocations: 15.26 MiB)

# Fancier ways to use @tullio:

julia> using LoopVectorization

julia> @btime @tullio s = f1($x2[i,j])
  3.194 ms (39 allocations: 2.59 KiB)

julia> @btime Zygote.gradient(x -> sum(@tullio s[i] := 1+tanh(x[i,j])), $x2);
  10.663 ms (81 allocations: 7.64 MiB)

julia> @btime Zygote.gradient(x -> (@tullio s := 1+tanh(x[i,j])), $x2); # skip making s[i]
  9.406 ms (79 allocations: 7.63 MiB)

julia> using ForwardDiff

julia> @btime Zygote.gradient(x -> sum(@tullio s[i] := 1+tanh(x[i,j]) grad=Dual), $x2);
  10.512 ms (81 allocations: 7.64 MiB)

# Another problem:

julia> @btime sum($x2 .+ $x2' ./ 2);
  2.968 ms (2 allocations: 7.63 MiB)

julia> @btime Tracker.gradient(x -> sum(x .+ x' ./ 2), $x2);
  25.699 ms (210 allocations: 68.67 MiB)

julia> @btime Zygote.gradient(x -> sum(x .+ x' ./ 2), $x2);
  9.006 ms (15 allocations: 38.15 MiB)

julia> @btime ReverseDiff.gradient(x -> sum(x .+ x' ./ 2), $x2);
  1.219 s (19000027 allocations: 719.65 MiB)

julia> @btime Zygote.gradient(x -> (@tullio s := x[i,j] + x[j,i]/2), $x2);
  1.756 ms (164 allocations: 7.64 MiB)

# And without @avx magic...

julia> @btime Zygote.gradient(x -> (@tullio s := x[i,j] + x[j,i]/2  avx=false), $x2);
  2.368 ms (165 allocations: 7.64 MiB)

julia> @btime Zygote.gradient(x -> (@tullio s := x[i,j] + x[j,i]/2  avx=false threads=false), $x2);
  5.146 ms (25 allocations: 7.63 MiB)

julia> @btime Tracker.gradient(x -> (@tullio s := x[i,j] + x[j,i]/2  avx=false), $x2);
  7.169 ms (172 allocations: 30.53 MiB)

julia> @btime ReverseDiff.gradient(x -> (@tullio s := x[i,j] + x[j,i]/2  avx=false), $x2);
  7.955 ms (164 allocations: 30.53 MiB)

# ReverseDiff should soon have forward-mode broadcasting from this package:

julia> using DistributionsAD

julia> @btime ReverseDiff.gradient(x -> sum(f1.(x)), ($x2,));
  35.920 ms (34 allocations: 53.41 MiB)

julia> @btime ReverseDiff.gradient(x -> sum(x .+ x' ./ 2), $x2);
  1.365 s (19000087 allocations: 727.28 MiB)

julia> @btime ReverseDiff.gradient(x -> sum(x .+ x'), $x2); # without the ./2 bit!
  48.920 ms (1000030 allocations: 99.18 MiB)

# Yota only supports broadcasting of primitives, i.e. things with an explicit scalar rule

julia> using Yota

julia> @diffrule f1(x)  x  dy*(1-tanh(x)^2)

julia> @btime Yota.grad(x -> sum(f1.(x)), $x2);
  77.449 ms (86 allocations: 45.78 MiB)

julia> @btime Yota.grad(x -> sum(x .+ x' ./ 2), $x2);
  8.423 ms (89 allocations: 38.15 MiB)
