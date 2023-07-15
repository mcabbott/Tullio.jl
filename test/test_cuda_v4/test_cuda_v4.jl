using Tullio, CUDA, KernelAbstractions

CUDA.allowscalar(false)
x = CUDA.ones(10)

@tullio y[i] := x[i]

A, B, C = CUDA.rand(2,2,2), CUDA.rand(2,2), CUDA.rand(2,2,2);
@tullio A[k,i,a] = tanh(B[i,a] + C[k,i,a])

