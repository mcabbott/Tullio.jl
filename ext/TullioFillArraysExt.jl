module TullioFillArraysExt

if !isdefined(Base, :get_extension)
    using ..Tullio, ..FillArrays
else
    using Tullio, FillArrays
end

Tullio.promote_storage(::Type{T}, ::Type{F}) where {T, F<:FillArrays.Fill} = T
Tullio.promote_storage(::Type{F}, ::Type{T}) where {T, F<:FillArrays.Fill} = T

end
