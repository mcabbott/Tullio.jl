module TullioMetalExt
if !isdefined(Base, :get_extension)
    using ..Tullio, ..Metal
else
    using Tullio, Metal
end

if isdefined(@__MODULE__, :CUDA) && isdefined(@__MODULE__, :CuArray)
    @warn "Loading multiple GPU backends can lead to unexpected bugs"
end

Tullio.threader(fun!::F, ::Type{T},
    Z::AbstractArray, As::Tuple, Is::Tuple, Js::Tuple,
    redfun, block=0, keep=nothing) where {F<:Function, T<:Metal.MtlArray} =
    fun!(T, Z, As..., Is..., Js..., keep)

Tullio.∇threader(fun!::F, ::Type{T},
    As::Tuple, Is::Tuple, Js::Tuple, block=0) where {F<:Function, T<:Metal.MtlArray} =
    fun!(T, As..., Is..., Js...,)

end
