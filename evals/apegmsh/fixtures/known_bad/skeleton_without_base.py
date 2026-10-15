from apeGmsh import apeGmsh

with apeGmsh(model_name="my_model", save_to="my_model.h5") as g:
    # 1-5: GEOMETRY (occ) → PHYSICAL GROUPS (dim-unique names) → pre-mesh
    #      LOADS/MASSES/CONSTRAINTS (reference labels/PGs) → MESH → SNAPSHOT.
    #      (workflows.md §1 has the fully-commented version.)
    g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="body")
    g.physical.add_volume("body", name="Body")
    with g.loads.case("dead"):
        g.loads.gravity("Body", g=(0, 0, -9.81), density=2400)
    g.masses.volume("Body", density=2400)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)   # the solver contract; autosaved to my_model.h5 on exit
    print(fem.info)                            # "N nodes, M elements, bandwidth=..."

# 6. OPENSEES (optional) — post-session bridge, typed primitives.
#    Masses/fixities are re-declared explicitly; loads are OPT-IN — a
#    g.loads.case reaches the deck only via p.from_model(case) inside a
#    pattern (ADR 0051). MP constraints + per-node ndf emit AUTOMATICALLY.
from apeGmsh.opensees import apeSees

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
ops.element.FourNodeTetrahedron(pg="Body", material=conc)
ops.fix(pg="Base", dofs=(1, 1, 1))
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.from_model("dead")                    # import the session's "dead" case
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))   # + an ad-hoc bridge load
ops.py("model.py")     # or ops.tcl(...), ops.h5(...), ops.run()
