"""
    Tullio.@einsum  A[i,j] += B[i] * C[j]

Since this package is almost superset of `Einsum.jl`, you can probable drop that and
write `using Tullio: @einsum` to use the new macro under the old name.

Differences:
* Constants need dollar signs like `A[i,1,\$c] + \$d`, as the macro creates a function
  which may not run in the caller's scope.
* Updating `A` with weird things like `*=` uses an awful hack which may be less efficient,
  but does make tests pass!
* Options `threads=false, avx=false, grad=false` are selected for you.
"""
macro einsum(ex::Expr)
    if ex.head in [:(:=), :(=), :(+=)]
        _tullio(ex, :(avx=false), :(threads=false), :(grad=false); mod=__module__)

    elseif ex.head in [:(-=), :(*=), :(/=)]
        @gensym tmp

        if @capture_(ex.args[1], Z_[ijk__]) # array *= ...
            act = Expr(:(:=), :($tmp[$(ijk...)]), ex.args[2:end]...)
            res = _tullio(act, :(avx=false), :(threads=false), :(grad=false); mod=__module__).args[1]
            Expr(Symbol(string(".", ex.head)), Z, res) |> esc

        elseif ex.args[1] isa Symbol # scalar case
            Z = ex.args[1]
            act = Expr(:(:=), :($tmp), ex.args[2:end]...)
            res = _tullio(act, :(avx=false), :(threads=false), :(grad=false); mod=__module__).args[1]
            Expr(ex.head, Z, res) |> esc

        end
    end
end
