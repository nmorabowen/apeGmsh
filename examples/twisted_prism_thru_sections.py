"""
Twisted prism via thru_sections
================================

Builds a twisted (helical) square prism by lofting through a series of
cross-sections that are progressively rotated about the sweep axis.

Why thru_sections and not sweep?
--------------------------------
``sweep`` (gmsh's ``addPipe``) translates a profile along a path.  With a
straight path the profile orientation does not change, so you cannot
get a twist out of it.  ``thru_sections`` (gmsh's ``addThruSections``)
lofts a smooth solid through an ordered list of wires — which is exactly
what we need: place each wire at a different Y *and* a different angle.
"""

import math
from apeGmsh import apeGmsh


# ---------------------------------------------------------------- params
a            = 1.0                  # half-width of the square section [m]
L            = 40.0                 # length of the prism along Y [m]
total_twist  = math.radians(90.0)   # end-to-end rotation about Y [rad]
n_stations   = 20                   # number of intermediate cross-sections
mesh_size    = 1.0                  # global mesh size [m]


# ---------------------------------------------------------------- model
g = apeGmsh(model_name="twisted_prism", verbose=True)
g.begin()


def build_section(y_pos: float, angle: float) -> int:
    """Square wire in the plane Y = y_pos, rotated by ``angle`` about Y."""
    c, s = math.cos(angle), math.sin(angle)
    # corners in the local XZ frame, then rotated about Y
    local_xz = [(-a, -a), (+a, -a), (+a, +a), (-a, +a)]
    pts = [
        g.model.geometry.add_point(c * lx + s * lz, y_pos, -s * lx + c * lz)
        for (lx, lz) in local_xz
    ]
    lines = [
        g.model.geometry.add_line(pts[i], pts[(i + 1) % 4])
        for i in range(4)
    ]
    return g.model.geometry.add_wire(lines, check_closed=True)


# ---------------------------------------------------------------- wires
wires = [
    build_section(y_pos=(i / n_stations) * L,
                  angle=(i / n_stations) * total_twist)
    for i in range(n_stations + 1)
]


# ---------------------------------------------------------------- loft
out    = g.model.transforms.thru_sections(wires,
                                          make_solid=True,
                                          label="twisted_prism")
volume = next(t for d, t in out if d == 3)


# ---------------------------------------------------------------- mesh
g.physical.add(3, [volume], name="Body")
g.mesh.sizing.set_global_size(max_size=mesh_size, min_size=mesh_size)
g.mesh.generation.generate(dim=3)
g.mesh.viewer()
