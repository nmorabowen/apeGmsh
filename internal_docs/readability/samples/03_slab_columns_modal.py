"""One-storey 3D building: 8 x 6 m concrete shell slab on four corner columns.

- Slab: 0.20 m thick ShellMITC4 at z = 3.5 m (elastic membrane-plate section)
- Columns: 0.40 x 0.40 m elasticBeamColumn, fixed at the base
- Slab and column are separate Parts, assembled with g.parts; the column
  tops land on the slab corners and fragmenting merges them into shared nodes.
- Mass: slab self weight + 2 kPa superimposed dead load, lumped on slab nodes.
- Modal analysis: first 3 periods.

Units: N, m, kg, s.
"""
from pathlib import Path
import math

from apeGmsh import apeGmsh, Part
from apeGmsh.opensees import apeSees

OUT = Path(__file__).parent / "out_03"
OUT.mkdir(exist_ok=True)

# ---- data ------------------------------------------------------------------
LX, LY, H = 8.0, 6.0, 3.5          # plan dimensions and storey height [m]
T_SLAB = 0.20                      # slab thickness [m]
B_COL = 0.40                       # square column side [m]

E_C = 25e9                         # concrete modulus [Pa]
NU_C = 0.2
G_C = E_C / (2 * (1 + NU_C))
RHO_C = 2400.0                     # concrete density [kg/m3]
SDL = 2000.0                       # superimposed dead load [Pa]
G_ACC = 9.81

A_COL = B_COL**2
I_COL = B_COL**4 / 12
J_COL = 0.141 * B_COL**4           # torsion constant, square section

SLAB_AREAL_MASS = RHO_C * T_SLAB + SDL / G_ACC     # [kg/m2]

MESH_SIZE = 0.5

# ---- parts -----------------------------------------------------------------
with Part("slab") as slab:
    slab.model.geometry.add_rectangle(0.0, 0.0, H, LX, LY, label="slab")

with Part("column") as column:
    base = column.model.geometry.add_point(0.0, 0.0, 0.0)
    top = column.model.geometry.add_point(0.0, 0.0, H)
    column.model.geometry.add_line(base, top, label="column")

# ---- assembly --------------------------------------------------------------
corners = [(0.0, 0.0), (LX, 0.0), (LX, LY), (0.0, LY)]

with apeGmsh(model_name="slab_columns", save_to=str(OUT / "model.h5")) as g:
    g.parts.add(slab, label="slab")
    for i, (x, y) in enumerate(corners, start=1):
        g.parts.add(column, label=f"col{i}", translate=(x, y, 0.0))

    # Column tops coincide with the slab corners: fragment so they share a point.
    g.parts.fragment_all()

    g.physical.add_surface("slab", name="Slab")
    g.physical.add_curve([f"col{i}" for i in range(1, 5)], name="Columns")
    eps = 1e-6
    (g.model.select(None, dim=0)
        .in_box((-eps, -eps, -eps), (LX + eps, LY + eps, eps))
        .to_physical("Base"))

    g.masses.surface("slab", areal_density=SLAB_AREAL_MASS)

    # Structured quad grid on the slab (ShellMITC4 needs quads).
    g.mesh.recipe.structured(size=MESH_SIZE)
    fem = g.mesh.queries.get_fem_data(dim=None)
    print(fem.info)

# ---- OpenSees ----------------------------------------------------------------
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)

# rho=0: slab mass comes from g.masses (self weight + SDL), not the section.
slab_sec = ops.section.ElasticMembranePlateSection(E=E_C, nu=NU_C, h=T_SLAB, rho=0.0)
ops.element.ShellMITC4(pg="Slab", section=slab_sec)

col_transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
ops.element.elasticBeamColumn(
    pg="Columns", transf=col_transf,
    A=A_COL, E=E_C, G=G_C, J=J_COL, Iy=I_COL, Iz=I_COL,
)

ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
ops.mass_from_model()

ops.py(str(OUT / "slab_columns.py"))

modes = ops.eigen(3)
print("\nMode   T [s]    f [Hz]")
for i, (T, f) in enumerate(zip(modes.periods, modes.freq), start=1):
    print(f"{i:4d}  {T:7.4f}  {f:7.3f}")

total_mass = fem.nodes.masses.total_mass()
k_storey = 4 * 12 * E_C * I_COL / H**3
print(f"\nSlab mass = {total_mass / 1e3:.1f} t; "
      f"rigid-slab estimate T = {2 * math.pi * math.sqrt(total_mass / k_storey):.3f} s")
