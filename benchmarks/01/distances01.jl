
# A Julia-vs-Jax game, from here:
# https://twitter.com/cgarciae88/status/1254269041784561665
# https://discourse.julialang.org/t/improving-an-algorithm-that-compute-gps-distances/38213/19
# https://gist.github.com/cgarciae/a69fa609f8fcd0aacece92660b5c2315

# These versions are updated to use Float32, which is what Jax is using (by default).
# It helped a little to avoid Float64 constants, and to add @inbounds in a few places.

using Pkg; pkg"add LoopVectorization Einsum TensorCast https://github.com/mcabbott/Tullio.jl"
using LoopVectorization, Tullio, Einsum, TensorCast, Test, BenchmarkTools

a = -100 .+ 200 .* rand(Float32, 5000, 2);
b = -100 .+ 200 .* rand(Float32, 5000, 2);

const None = [CartesianIndex()]

function distances(data1, data2)
    data1 = deg2rad.(data1)
    data2 = deg2rad.(data2)
    lat1 = @view data1[:, 1]
    lng1 = @view data1[:, 2]
    lat2 = @view data2[:, 1]
    lng2 = @view data2[:, 2]
    diff_lat = @view(lat1[:, None]) .- @view(lat2[None, :])
    diff_lng = @view(lng1[:, None]) .- @view(lng2[None, :])
    data = (
        @. sin(diff_lat / 2)^2 +
        cos(@view(lat1[:, None])) * cos(@view(lat2[None,:])) * sin(diff_lng / 2)^2
    )
    data .= @. 2.0 * 6373.0 * atan(sqrt(abs(data)), sqrt(abs(1.0 - data)))
    return reshape(data, (size(data1, 1), size(data2, 1)))
end

res = distances(a, b);
@test eltype(res) == Float32

function distances_threaded(data1, data2)
    lat1 = [deg2rad(data1[i,1]) for i in 1:size(data1, 1)]
    lng1 = [deg2rad(data1[i,2]) for i in 1:size(data1, 1)]
    lat2 = [deg2rad(data2[i,1]) for i in 1:size(data2, 1)]
    lng2 = [deg2rad(data2[i,2]) for i in 1:size(data2, 1)]
    # data = Matrix{Float64}(undef, length(lat1), length(lat2))
    data = Matrix{eltype(data1)}(undef, length(lat1), length(lat2))
    @inbounds Threads.@threads for i in eachindex(lat2)
        lat, lng = lat2[i], lng2[i]
        data[:, i] .= @. sin((lat1 - lat) / 2)^2 + cos(lat1) * cos(lat) * sin((lng1 - lng) / 2)^2
    end
    Threads.@threads for i in eachindex(data)
        # data[i] = 2.0 * 6373.0 * atan(sqrt(abs(data[i])), sqrt(abs(1.0 - data[i])))
        @inbounds data[i] = 2 * 6373 * atan(sqrt(abs(data[i])), sqrt(abs(1 - data[i])))
    end
    return data
end

function distances_threaded_simd(data1, data2) # @baggepinnen
    lat1 = [deg2rad(data1[i,1]) for i in 1:size(data1, 1)]
    lng1 = [deg2rad(data1[i,2]) for i in 1:size(data1, 1)]
    lat2 = [deg2rad(data2[i,1]) for i in 1:size(data2, 1)]
    lng2 = [deg2rad(data2[i,2]) for i in 1:size(data2, 1)]
    # data = Matrix{Float64}(undef, length(lat1), length(lat2))
    data = Matrix{eltype(data1)}(undef, length(lat1), length(lat2))
    Threads.@threads for i in eachindex(lat2)
        @inbounds lat, lng = lat2[i], lng2[i]
        @avx data[:, i] .= @. sin((lat1 - lat) / 2)^2 + cos(lat1) * cos(lat) * sin((lng1 - lng) / 2)^2
    end
    Threads.@threads for i in eachindex(data)
        # @avx data[i] = 2.0 * 6373.0 * atan(sqrt(abs(data[i])), sqrt(abs(1.0 - data[i])))
        @avx data[i] = 2 * 6373 * atan(sqrt(abs(data[i])), sqrt(abs(1 - data[i])))
    end
    return data
end

@test res ≈ distances_threaded(a, b)
@test eltype(distances_threaded(a, b)) == Float32
@test res ≈ distances_threaded_simd(a, b)
@test eltype(distances_threaded_simd(a, b)) == Float32

function distances_bcast(data1, data2) # @DNF
    data1 = deg2rad.(data1)
    data2 = deg2rad.(data2)
    lat1 = @view data1[:, 1]
    lng1 = @view data1[:, 2]
    lat2 = @view data2[:, 1]
    lng2 = @view data2[:, 2]
    data = sin.((lat1 .- lat2') ./ 2).^2 .+ cos.(lat1) .* cos.(lat2') .* sin.((lng1 .- lng2') ./ 2).^2
    @. data = 2 * 6373 * atan(sqrt(abs(data)), sqrt(abs(1 - data)))
    return data
end

function distances_bcast_simd(data1, data2)
    data1 = deg2rad.(data1)
    data2 = deg2rad.(data2)
    lat1 = @view data1[:, 1]
    lng1 = @view data1[:, 2]
    lat2 = @view data2[:, 1]
    lng2 = @view data2[:, 2]
    @avx data = sin.((lat1 .- lat2') ./ 2).^2 .+ cos.(lat1) .* cos.(lat2') .* sin.((lng1 .- lng2') ./ 2).^2
    @. data = 2 * 6373 * atan(sqrt(abs(data)), sqrt(abs(1 - data)))
    return data
end

@test res ≈ distances_bcast(a, b)
@test eltype(distances_bcast(a, b)) == Float32
@test res ≈ distances_bcast_simd(a, b)
@test eltype(distances_bcast_simd(a, b)) == Float32

function distances_einsum(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @einsum data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cos(data1[n,1]) * cos(data2[m,1]) * sin((data1[n,2] - data2[m,2])/2)^2

    @einsum data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
end

function distances_vielsum(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @vielsum data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cos(data1[n,1]) * cos(data2[m,1]) * sin((data1[n,2] - data2[m,2])/2)^2

    @vielsum data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
end

@test res ≈ distances_einsum(a, b)
@test eltype(distances_einsum(a, b)) == Float32
@test res ≈ distances_vielsum(a, b)
@test eltype(distances_vielsum(a, b)) == Float32

function distances_cast(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @cast data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cos(data1[n,1]) * cos(data2[m,1]) * sin((data1[n,2] - data2[m,2])/2)^2

    @cast data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
end

function distances_cast_avx(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @cast data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cos(data1[n,1]) * cos(data2[m,1]) * sin((data1[n,2] - data2[m,2])/2)^2  avx

    @cast data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))  avx
end

@test res ≈ distances_cast(a, b)
@test eltype(distances_cast(a, b)) == Float32
@test res ≈ distances_cast_avx(a, b)
@test eltype(distances_cast_avx(a, b)) == Float32

function distances_tullio(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @tullio data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cos(data1[n,1]) * cos(data2[m,1]) * sin((data1[n,2] - data2[m,2])/2)^2

    @tullio data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
end

@test res ≈ distances_tullio(a, b)
@test eltype(distances_tullio(a, b)) == Float32



##### laptop (2 cores, 4 threads)

julia> a = -100 .+ 200 .* rand(Float32, 5000, 2);
julia> b = -100 .+ 200 .* rand(Float32, 5000, 2);

julia> @btime distances($a, $b);
  1.522 s (26 allocations: 286.18 MiB)

julia> @btime distances_threaded($a, $b);
  516.937 ms (64 allocations: 95.45 MiB)

julia> @btime distances_threaded_simd($a, $b);
  235.098 ms (64 allocations: 95.45 MiB)

julia> @btime distances_bcast($a, $b);
  1.352 s (10 allocations: 95.44 MiB)

julia> @btime distances_bcast_simd($a, $b);
  641.506 ms (43 allocations: 95.44 MiB)

julia> @btime distances_einsum($a, $b);
  1.435 s (6 allocations: 95.44 MiB)

julia> @btime distances_vielsum($a, $b);
  572.038 ms (53 allocations: 95.45 MiB)

julia> @btime distances_cast($a, $b); # should be much like distances_bcast
  1.339 s (12 allocations: 95.44 MiB)

julia> @btime distances_cast_avx($a, $b);
  170.653 ms (45 allocations: 190.81 MiB)

julia> @btime distances_tullio($a, $b);
  51.442 ms (636 allocations: 95.47 MiB)

julia> a = -100 .+ 200 .* rand(Float64, 5000, 2); ##### repeat everythinng in Float64
julia> b = -100 .+ 200 .* rand(Float64, 5000, 2);

julia> @btime distances($a, $b);
  1.716 s (26 allocations: 572.36 MiB)

julia> @btime distances_threaded($a, $b);
  550.213 ms (64 allocations: 190.89 MiB)

julia> @btime distances_threaded_simd($a, $b);
  348.886 ms (64 allocations: 190.89 MiB)

julia> @btime distances_bcast($a, $b);
  1.533 s (10 allocations: 190.89 MiB)

julia> @btime distances_bcast_simd($a, $b);
  981.204 ms (43 allocations: 190.89 MiB)

julia> @btime distances_einsum($a, $b);
  1.454 s (6 allocations: 190.89 MiB)

julia> @btime distances_vielsum($a, $b);
  549.669 ms (52 allocations: 190.89 MiB)

julia> @btime distances_cast($a, $b);
  1.546 s (12 allocations: 190.89 MiB)

julia> @btime distances_cast_avx($a, $b);
  607.745 ms (45 allocations: 381.62 MiB)

julia> @btime distances_tullio($a, $b);
  188.599 ms (636 allocations: 190.91 MiB)



##### desktop (6 cores, 12 threads)

julia> a = -100 .+ 200 .* rand(Float32, 5000, 2);
julia> b = -100 .+ 200 .* rand(Float32, 5000, 2);

julia> @btime distances($a, $b);
  1.167 s (26 allocations: 286.18 MiB)

julia> @btime distances_threaded($a, $b);
  140.228 ms (144 allocations: 95.46 MiB)

julia> @btime distances_threaded_simd($a, $b);
  65.031 ms (144 allocations: 95.46 MiB)

julia> @btime distances_bcast($a, $b);
  1.033 s (10 allocations: 95.44 MiB)

julia> @btime distances_bcast_simd($a, $b);
  501.345 ms (43 allocations: 95.44 MiB)

julia> @btime distances_einsum($a, $b);
  993.101 ms (6 allocations: 95.44 MiB)

julia> @btime distances_vielsum($a, $b);
  145.149 ms (133 allocations: 95.46 MiB)

julia> @btime distances_tullio($a, $b);
  27.329 ms (789 allocations: 95.48 MiB)

julia> a = -100 .+ 200 .* rand(Float64, 5000, 2); ##### repeat everythinng in Float64
julia> b = -100 .+ 200 .* rand(Float64, 5000, 2);

julia> @btime distances($a, $b);
  1.308 s (26 allocations: 572.36 MiB)

julia> @btime distances_threaded($a, $b);
  146.515 ms (145 allocations: 190.90 MiB)

julia> @btime distances_threaded_simd($a, $b);
  98.534 ms (144 allocations: 190.90 MiB)

julia> @btime distances_bcast($a, $b);
  1.136 s (10 allocations: 190.89 MiB)

julia> @btime distances_bcast_simd($a, $b);
  729.348 ms (43 allocations: 190.89 MiB)

julia> @btime distances_einsum($a, $b);
  1.087 s (6 allocations: 190.89 MiB)

julia> @btime distances_vielsum($a, $b);
  149.393 ms (133 allocations: 190.90 MiB)

julia> @btime distances_tullio($a, $b);
  75.990 ms (790 allocations: 190.93 MiB)



##### GPU (an ancient one!)

julia> using CuArrays, KernelAbstractions # and then re-run defn. of distances_tullio

julia> CuArrays.allowscalar(false)

julia> ca = cu(a); cb = cu(b); # Float32

julia> cres = distances_bcast(ca, cb);

julia> @test cres ≈ distances_tullio(ca, cb)
Test Passed

julia> @btime CuArrays.@sync distances_bcast($ca, $cb);
  31.550 ms (420 allocations: 18.42 KiB)

julia> @btime CuArrays.@sync distances_tullio($ca, $cb);
  188.699 ms (330445 allocations: 5.05 MiB)


