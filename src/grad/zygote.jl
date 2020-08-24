
using .Zygote

Zygote.@adjoint function (ev::Eval)(args...)
    Z = ev.fwd(args...)
    Z, Δ -> begin
        isnothing(ev.rev) && error("no gradient definition here!")
        tuple(nothing, ev.rev(Δ, Z, args...)...)
    end
end

Tullio.promote_storage(::Type{T}, ::Type{F}) where {T, F<:Zygote.Fill} = T
Tullio.promote_storage(::Type{F}, ::Type{T}) where {T, F<:Zygote.Fill} = T
