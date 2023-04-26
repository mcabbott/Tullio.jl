module TullioCUDAExt

if !isdefined(Base, :get_extension)
    using ..Tullio, ..CUDA
else
    using Tullio, CUDA
end

Tullio.threader(fun!::F, ::Type{T},
    Z::AbstractArray, As::Tuple, Is::Tuple, Js::Tuple,
    redfun, block=0, keep=nothing) where {F<:Function, T<:CUDA.CuArray} =
    fun!(T, Z, As..., Is..., Js..., keep)

Tullio.∇threader(fun!::F, ::Type{T},
    As::Tuple, Is::Tuple, Js::Tuple, block=0) where {F<:Function, T<:CUDA.CuArray} =
    fun!(T, As..., Is..., Js...,)

# Tullio.thread_scalar ... ought to work? Was never fast.

end
