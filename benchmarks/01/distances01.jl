
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
    data = similar(data1, length(lat1), length(lat2))
    cos_lat1 = @avx cos.(lat1)
    Threads.@threads for i in eachindex(lat2)
        # lat, lng = lat2[i], lng2[i]
        @inbounds lat, cos_lat, lng = lat2[i], cos(lat2[i]), lng2[i]
        # @avx data[:, i] .= @. sin((lat1 - lat) / 2)^2 + cos(lat1) * cos(lat) * sin((lng1 - lng) / 2)^2
        @avx data[:, i] .= @. sin((lat1 - lat) / 2)^2 + cos_lat1 * cos_lat * sin((lng1 - lng) / 2)^2
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

    @einsum cd1[n] := cos(data1[n,1])
    @einsum cd2[m] := cos(data2[m,1])

    @einsum data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cd1[n] * cd2[m] * sin((data1[n,2] - data2[m,2])/2)^2

    @einsum data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
end

function distances_vielsum(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @vielsum cd1[n] := cos(data1[n,1])
    @vielsum cd2[m] := cos(data2[m,1])

    @vielsum data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cd1[n] * cd2[m] * sin((data1[n,2] - data2[m,2])/2)^2

    @vielsum data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
end

@test res ≈ distances_einsum(a, b)
@test eltype(distances_einsum(a, b)) == Float32
@test res ≈ distances_vielsum(a, b)
@test eltype(distances_vielsum(a, b)) == Float32

function distances_cast(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @cast cd1[n] := cos(data1[n,1]) # pulling these out is worth 25%
    @cast cd2[m] := cos(data2[m,1])

    @cast data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cd1[n] * cd2[m] * sin((data1[n,2] - data2[m,2])/2)^2

    @cast data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
end

function distances_cast_avx(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @cast cd1[n] := cos(data1[n,1])  avx
    @cast cd2[m] := cos(data2[m,1])  avx

    @cast data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cd1[n] * cd2[m] * sin((data1[n,2] - data2[m,2])/2)^2  avx

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

# function distances_tullio2(data1deg, data2deg)
#     data1 = deg2rad.(data1deg)
#     data2 = deg2rad.(data2deg)

#     @tullio cd1[n] := cos(data1[n,1]) # has no effect
#     @tullio cd2[m] := cos(data2[m,1])

#     @tullio data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
#         cd1[n] * cd2[m] * sin((data1[n,2] - data2[m,2])/2)^2

#     @tullio data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
# end

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
  215.938 ms (66 allocations: 95.47 MiB)

julia> @btime distances_bcast($a, $b);
  1.352 s (10 allocations: 95.44 MiB)

julia> @btime distances_bcast_simd($a, $b);
  641.506 ms (43 allocations: 95.44 MiB)

julia> @btime distances_einsum($a, $b);
  983.168 ms (10 allocations: 95.48 MiB)

julia> @btime distances_vielsum($a, $b);
  389.831 ms (103 allocations: 95.49 MiB)

julia> @btime distances_cast($a, $b); # unlike distances_bcast, this pulls out cos(...)
  1.034 s (16 allocations: 95.48 MiB)

julia> @btime distances_cast_avx($a, $b); # and this applies more @avx than bcast_simd
  137.557 ms (43 allocations: 190.85 MiB)

julia> @btime distances_tullio($a, $b);
  51.442 ms (636 allocations: 95.47 MiB)



##### desktop (6 cores, 12 threads)

julia> a = -100 .+ 200 .* rand(Float32, 5000, 2);
julia> b = -100 .+ 200 .* rand(Float32, 5000, 2);

julia> @btime distances($a, $b);
  1.166 s (26 allocations: 286.18 MiB)

julia> @btime distances_threaded($a, $b);
  140.062 ms (144 allocations: 95.46 MiB)

julia> @btime distances_threaded_simd($a, $b);
  64.382 ms (147 allocations: 95.48 MiB)

julia> @btime distances_bcast($a, $b);
  1.033 s (10 allocations: 95.44 MiB)

julia> @btime distances_bcast_simd($a, $b);
  501.002 ms (43 allocations: 95.44 MiB)

julia> @btime distances_einsum($a, $b);
  756.749 ms (10 allocations: 95.48 MiB)

julia> @btime distances_vielsum($a, $b);
  108.200 ms (262 allocations: 95.51 MiB)

julia> @btime distances_cast($a, $b); # unlike distances_bcast, this pulls out cos(...)
  795.199 ms (16 allocations: 95.48 MiB)

julia> @btime distances_cast_avx($a, $b); # and this applies more @avx than bcast_simd
  112.824 ms (43 allocations: 190.85 MiB)

julia> @btime distances_tullio($a, $b);
  28.151 ms (788 allocations: 95.48 MiB)




julia> a = -100 .+ 200 .* rand(Float64, 5000, 2); ##### repeat everythinng in Float64
julia> b = -100 .+ 200 .* rand(Float64, 5000, 2);

julia> @btime distances($a, $b);
  1.308 s (26 allocations: 572.36 MiB)

julia> @btime distances_threaded($a, $b);
  146.374 ms (144 allocations: 190.90 MiB)

julia> @btime distances_threaded_simd($a, $b);
  92.097 ms (146 allocations: 190.94 MiB)

julia> @btime distances_bcast($a, $b);
  1.134 s (10 allocations: 190.89 MiB)

julia> @btime distances_bcast_simd($a, $b);
  728.564 ms (43 allocations: 190.89 MiB)

julia> @btime distances_einsum($a, $b);
  874.312 ms (10 allocations: 190.96 MiB)

julia> @btime distances_vielsum($a, $b);
  123.725 ms (262 allocations: 191.00 MiB)

julia> @btime distances_cast($a, $b);
  902.447 ms (16 allocations: 190.96 MiB)

julia> @btime distances_cast_avx($a, $b);
  431.136 ms (43 allocations: 381.70 MiB)

julia> @btime distances_tullio($a, $b);
  75.608 ms (786 allocations: 190.93 MiB)



##### GPU (an ancient one!)

julia> using CuArrays, KernelAbstractions # and then re-run defn. of distances_tullio

julia> CuArrays.allowscalar(false)

julia> ca = cu(a); cb = cu(b); # Float32

julia> cres = distances_bcast(ca, cb);

julia> @test cres ≈ distances_tullio(ca, cb)
Test Passed

julia> @test cres ≈ distances_cast(ca, cb)
Test Passed

julia> @btime CuArrays.@sync distances_bcast($ca, $cb);
  31.558 ms (420 allocations: 18.42 KiB)

julia> @btime CuArrays.@sync distances_cast($ca, $cb);
  29.728 ms (546 allocations: 22.48 KiB)

julia> @btime CuArrays.@sync distances_tullio($ca, $cb);
  187.258 ms (173551 allocations: 2.66 MiB)


