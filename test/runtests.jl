using Test, Printf

t1 = @elapsed using Tullio
@info @sprintf("Loading Tullio took %.1f seconds", t1)

@info "Testing with $(Threads.nthreads()) threads"
if Threads.nthreads() > 1 # use threading even on small arrays
    Tullio.BLOCK[] = 32
    Tullio.TILE[] = 32
end

is_buildkite = parse(Bool, get(ENV, "BUILDKITE", "false"))
if is_buildkite
    test_group = "2" # if this is Buildkite, we only run group 2
else
    test_group = string(get(ENV, "TULLIO_TEST_GROUP", "all"))
end
@info "" test_group is_buildkite

num_test_groups = 5
if !(test_group in vcat(["all"], string.(1:num_test_groups)))
    throw(ArgumentError("\"$(test_group)\" is not a valid test group"))
end
# for i in 1:num_test_groups
for i in [1, 3, 5]
    file_i = joinpath(@__DIR__, "group-$(i).jl")
    if test_group in ["all", string(i)]
        @info "Beginning to run test file" i file_i
        include(file_i)
        @info "Finished running test file" i file_i
    else
        @info "Skipping test file" i file_i
    end
end
for i in [2, 4] # these groups are currently broken on Julia nightly
    file_i = joinpath(@__DIR__, "group-$(i).jl")
    if test_group in ["all", string(i)]
        if (get(ENV, "CI", "") == "true") && (!(Base.VERSION < v"1.6.0"))
            @info "Skipping test file because this is a CI job on nightly" i file_i
        else
            @info "Beginning to run test file" i file_i
            include(file_i)
            @info "Finished running test file" i file_i
        end
    else
        @info "Skipping test file" i file_i
    end
end
