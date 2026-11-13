module BEASTInterfaceTest
using Test
using MLFMA
using BEAST

# MLFMA extends BEAST's quadrature seam for its far-field assembly. Two things have to stay true,
# and neither is visible from a load or from reading the source.
#
# WHY THIS EXISTS. These four methods were once written unqualified, as `function quaddata(...)`
# rather than `function BEAST.quaddata(...)`. BEAST exports both names, but a definition of a name
# that has only been made AVAILABLE by `using` creates a NEW binding rather than extending the
# imported one. So MLFMA silently owned a private `quaddata` carrying just these methods, while
# `BEAST.quaddata` had none of them. Nothing failed, because MLFMA's far-field assembler calls them
# unqualified too and reached its own copy, but `MLFMA.quaddata` shadowed BEAST's export and would
# have raised `MethodError` for any operator BEAST could otherwise have handled.
#
# This is the same shadowing mechanism that cost `Nθ` its methods when `Sampling` was split out;
# `test/Package/layers.jl` pins that case for the layer modules, and this file pins it for BEAST.

# A method defined inside `MLFMA.Operators` is still MLFMA's own.
function belongstomlfma(mod::Module)
    mod === MLFMA && return true
    parent = parentmodule(mod)
    return parent !== mod && belongstomlfma(parent)
end

@testset "BEAST quadrature seam" begin
    @testset "the generics are BEAST's, not forked copies" begin
        # ASKED OF THE LAYER, not of the top level. These used to read `getfield(MLFMA, :quaddata)`,
        # which resolved only because `src/MLFMA.jl` said `using BEAST`. Task #175 removed that line
        # (nothing outside `src` reaches a BEAST name through `MLFMA.`), and the top level has no
        # `quaddata` binding at all now.
        #
        # `Operators` is where the question belonged anyway: it owns `hhfarfields.jl` and
        # `mwfarfields.jl`, so it is the module whose methods could fork a rival generic. The
        # testset below already compares each method's defining module against this same layer.
        @test getfield(MLFMA.Operators, :quaddata) === BEAST.quaddata
        @test getfield(MLFMA.Operators, :quadrule) === BEAST.quadrule

        # And the top level does not carry them, so no call site can reach a quadrature generic
        # through `MLFMA.` and appear to be using MLFMA's own.
        @test !isdefined(MLFMA, :quaddata)
        @test !isdefined(MLFMA, :quadrule)
    end

    @testset "dispatch still reaches MLFMA's methods" begin
        # Identity alone is not enough: sharing BEAST's generic also exposes its ~50 other methods
        # as candidates. For the two far-field types MLFMA assembles, resolution must still land on
        # MLFMA's own methods rather than sliding onto a BEAST one.
        for (Operator, file) in
            ((MLFMA.HHFarField, "hhfarfields.jl"), (BEAST.MWFarField, "mwfarfields.jl"))
            quaddatamethod = which(
                BEAST.quaddata, Tuple{Operator,Any,Any,BEAST.SingleNumQStrat}
            )
            quadrulemethod = which(
                BEAST.quadrule, Tuple{Operator,Any,Any,Any,Any,BEAST.SingleNumQStrat}
            )

            @test basename(string(quaddatamethod.file)) == file
            @test basename(string(quadrulemethod.file)) == file
            # The defining module is whichever LAYER owns the file, `MLFMA.Operators` today, so
            # compare against the package root rather than against `MLFMA` itself. Asserting
            # `=== MLFMA` made this testset fail the moment the operator files moved into a layer,
            # which was a defect in the test and not in the code.
            @test belongstomlfma(quaddatamethod.module)
            @test belongstomlfma(quadrulemethod.module)
        end
    end

    @testset "no name collides with a BEAST export by accident" begin
        # `farfieldvaluetype` was called `valuetype`, which BEAST also exports. While both arrived
        # through `using`, the two tied and Julia left `MLFMA.valuetype` UNDEFINED rather than
        # picking one; before the layer split, MLFMA's own top-level definition simply won, so the
        # hazard was invisible.
        #
        # Extending BEAST's was NOT the fix. `BEAST.valuetype(refspace, ::Type)` is the value type
        # of a reference space; MLFMA's is the element type of an operator's far-field integrand,
        # a different concept at a different arity. Renaming removes the collision outright.
        @test isdefined(MLFMA, :farfieldvaluetype)
        @test parentmodule(MLFMA.farfieldvaluetype) === MLFMA.Operators
        @test MLFMA.farfieldvaluetype !== BEAST.valuetype

        # And MLFMA no longer has anything called `valuetype` AT ALL. This used to read
        # `MLFMA.valuetype === BEAST.valuetype`: the name resolved straight through to BEAST's,
        # unambiguously, because the top level said `using BEAST`. Task #175 removed that line, so
        # the name no longer resolves in `MLFMA` in any sense.
        #
        # The stronger statement, and the same guarantee. The hazard was two `using` lines tying on
        # one name; with BEAST no longer among them the tie is not resolved in MLFMA's favour, it
        # cannot occur. A `valuetype` defined in a layer later would make this fire.
        @test !isdefined(MLFMA, :valuetype)

        # Non-vacuity for the arity claim above, which is the reason the two are not one function.
        @test !hasmethod(BEAST.valuetype, Tuple{Any})
        @test hasmethod(MLFMA.farfieldvaluetype, Tuple{MLFMA.HHFarField})
    end

    @testset "the gap BEAST leaves, which is why these methods exist at all" begin
        # If any of these three stops holding, BEAST has grown its own coverage and MLFMA's methods
        # should be reconsidered rather than kept out of habit. The `MWFarField` pair in particular
        # is type piracy, justified only by this gap.
        @test MLFMA.HHFarField <: BEAST.FarField
        @test BEAST.MWFarField <: BEAST.FarField
        @test !(BEAST.FarField <: BEAST.Functional)
    end
end
end
