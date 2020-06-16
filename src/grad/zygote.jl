
using .Zygote

Zygote.@adjoint function (ev::Eval)(args...)
    Z = ev.fwd(args...)
    Z, Δ -> begin
        isnothing(ev.rev) && error("no gradient definition here!")
        tuple(nothing, ev.rev(Δ, Z, args...)...)
    end
end

Tullio.promote_storage(T::Type, ::Type{<:Zygote.Fill}) = T
Tullio.promote_storage(::Type{<:Zygote.Fill}, T::Type) = T
