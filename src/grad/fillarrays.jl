using .FillArrays: Fill # used by Zygote
Tullio.promote_storage(::Type{T}, ::Type{F}) where {T, F<:Fill} = T
Tullio.promote_storage(::Type{F}, ::Type{T}) where {T, F<:Fill} = T
