using .CUDA: CuArray, GPUArrays

Tullio.threader(fun!::F, ::Type{T},
    Z::AbstractArray, As::Tuple, Is::Tuple, Js::Tuple,
    redfun, block=0, keep=nothing) where {F<:Function, T<:CuArray} =
    fun!(T, Z, As..., Is..., Js..., keep)

Tullio.∇threader(fun!::F, ::Type{T},
    As::Tuple, Is::Tuple, Js::Tuple, block=0) where {F<:Function, T<:CuArray} =
    fun!(T, As..., Is..., Js...,)

# Tullio.thread_scalar ... ought to work? Was never fast.
