module TullioChainRulesCoreExt

using Tullio, ChainRulesCore

function ChainRulesCore.rrule(ev::Tullio.Eval, args...)
    Z = ev.fwd(args...)
    Z, function tullio_back(Δ)
        isnothing(ev.rev) && error("no gradient definition here!")
        dxs = map(ev.rev(Δ, Z, args...)) do dx
            dx === nothing ? ChainRulesCore.ZeroTangent() : dx
        end
        tuple(ChainRulesCore.ZeroTangent(), dxs...)
    end
end

end
