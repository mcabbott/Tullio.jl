
using Tullio, Test, ForwardDiff
# using Tracker; _gradient(x...) = Tracker.gradient(x...)

# simple
@test _gradient(x -> sum(@tullio y[i] := 2*x[i]), rand(3))[1] == [2,2,2]
@test _gradient(x -> sum(@tullio y[i] := 2*x[i] + i), rand(3))[1] == [2,2,2]

# two contributions
g2(x) = @tullio y[i, j] := 1 * x[i] + 1000 * x[j]
mat = [1 1 3; 1 1 5; 7 7 7]
g_fd = ForwardDiff.gradient(x -> sum(mat .* g2(x)), rand(3))
@test g_fd ≈ _gradient(x -> sum(mat .* g2(x)), rand(3))[1]

# two arrays, and a sum
h2(x,y) = @tullio z[i] := x[i,j] + y[j,i]
@test _gradient(sum∘h2, rand(2,3), rand(3,2)) == (ones(2,3), ones(3,2))

# nontrivial function
flog(x,y) = @tullio z[i] := log(x[i,j]) / y[j,i]
r_x, r_y = rand(2,3), rand(3,2)
fx = ForwardDiff.gradient(x -> sum(flog(x, r_y)), r_x)
fy = ForwardDiff.gradient(y -> sum(flog(r_x, y)), r_y)
@test fx ≈ _gradient(sum∘flog, r_x, r_y)[1]
@test fy ≈ _gradient(sum∘flog, r_x, r_y)[2]

# classic
mm(x,y) = @tullio z[i,j] := 2 * x[i,k] * y[k,j]
x1 = rand(3,4);
y1 = rand(4,5);
z1 = x1 * y1
dx, dy = _gradient(sum∘mm, x1, y1)
@test dx ≈ 2 * ones(3,5) * y1'
@test dy ≈ 2 * x1' * ones(3,5)

# Using zero-dim arrays fails on ReverseDiff & Tracker
# Tracker.gradient(x -> x[], fill(1.0))
# ReverseDiff.gradient(x -> x[], fill(1.0)) # is ambiguous
@test_skip _gradient(x -> sum(@tullio y[] := log(x[i])), collect(1:3.0))[1] == 1 ./ (1:3)
# one-element vectors are fine:
@test _gradient(x -> sum(@tullio y[1] := log(x[i])), collect(1:3.0))[1] == 1 ./ (1:3)
# which is what's now used for this:
@test _gradient(x -> (@tullio y := log(x[i])), collect(1:3.0))[1] == 1 ./ (1:3)

#=
# shifts, etc
c1(N,K) = @tullio M[x,y,c] := N[x+i-1, y+j-1,c] * K[i,j]
m1 = rand(10,10,2)
k1 = rand(3,3)
g_m = ForwardDiff.gradient(N -> sum(sin, c1(N, k1)), m1)
g_k = ForwardDiff.gradient(K -> sum(sin, c1(m1, K)), k1)
@test g_m ≈ _gradient(N -> sum(sin, c1(N, k1)), m1)[1]  atol=0.01
@test g_k ≈ _gradient(K -> sum(sin, c1(m1, K)), k1)[1]  atol=0.01

c2(mat, kern) = @tullio out[x,y,n] := begin
        i = mod(x+a, axes(mat,1))
        j = mod(y+b, axes(mat,2))
        @inbounds mat[i,j,n] * abs(kern[a,b])
    end (x in axes(mat,1), y in axes(mat,2)) grad=Dual

if Tullio.GRAD[] == :Dual
    g_m = ForwardDiff.gradient(N -> sum(sin, c2(N, k1)), m1)
    g_k = ForwardDiff.gradient(K -> sum(sin, c2(m1, K)), k1)
    @test g_m ≈ _gradient(N -> sum(sin, c2(N, k1)), m1)[1]  atol=0.01
    @test g_k ≈ _gradient(K -> sum(sin, c2(m1, K)), k1)[1]  atol=0.01
end
=#

