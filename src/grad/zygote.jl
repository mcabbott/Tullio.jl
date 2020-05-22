
using .Zygote

Zygote.@adjoint function (ev::Eval)(args...)
    ev.fwd(args...), Δ -> begin
        isnothing(ev.rev) && error("no gradient definition here!")
        tuple(nothing, ev.rev(Δ, args...)...)
    end
end

Tullio.promote_storage(T::Type, ::Type{<:Zygote.Fill}) = T
Tullio.promote_storage(::Type{<:Zygote.Fill}, T::Type) = T
