"""
apeGmsh.opensees — Pythonic, statically-typed wrapper around OpenSees.

This package is the architectural successor to ``apeGmsh.solvers``. It
is intentionally empty during the design phase: see
``architecture/README.md`` for the design charter, decisions, and
implementation roadmap.

Once skeletons land, the public surface will be:

    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    ...
    ops.tcl("model.tcl")
    ops.run()
"""
