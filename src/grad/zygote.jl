
using .Zygote

Zygote.@adjoint function (ev::Eval)(args...)
    Z = ev.fwd(args...)
    Z, Δ -> begin
        ev.rev===nothing && throw("No gradient definition found! Running `@tullio` with keyword `verbose=true` may print the reason")
        tuple(nothing, ev.rev(Δ, Z, args...)...)
    end
end

Tullio.promote_storage(::Type{T}, ::Type{F}) where {T, F<:Zygote.Fill} = T
Tullio.promote_storage(::Type{F}, ::Type{T}) where {T, F<:Zygote.Fill} = T
