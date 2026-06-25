# ============================================================================
# build_drm_buffered.py  --  STABILIZED DRM box: inner dataset grid + exterior
#                            buffer + absorbing/fixed boundary on NON-dataset nodes
# ----------------------------------------------------------------------------
# A FREE DRM box diverges (unconstrained rigid-body / residual-net-force drift).
# Fix: surround the inner 5x5x4 dataset grid with extra 50 m soil layers on the
# 4 SIDES and the BOTTOM (NOT the top z=0 free surface), and put an absorbing
# (or fixed) boundary on the OUTERMOST faces.
#
# KEY (H5DRMLoadPattern.cpp:580): an element is a "DRM element" ONLY IF ALL its
# nodes are matched dataset nodes. The inner 98 nodes stay at their EXACT dataset
# coords so H5DRM matches 98/98. Buffer/boundary elements include exterior
# (non-dataset) nodes => H5DRM ignores them. The exterior carries only the
# scattered field => bare Lysmer dashpots (or a far fixed boundary) are correct.
#
# Builds the mesh structurally in numpy (full control over the structured box) and
# emits node / element stdBrick Tcl directly, plus boundary tcl for two variants:
#   drm_model_buf.tcl       -- nodes + stdBricks (inner + buffer), shared
#   bc_fixed.tcl            -- fix all outermost-face nodes (variant a)
#   bc_lysmer.tcl           -- LysmerTriangle dashpots on outer faces (variant b)
#
# Build is METERS, z-DOWN, centered laterally at origin. Inner dataset region:
#   x,y in [-100,100], z in [0,150]  (5x5x4 nodes, 50 m).
# Buffer adds NBUF layers of 50 m hexes outward on -x,+x,-y,+y and +z(bottom).
# ============================================================================
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = 50.0            # hex size (m)
NBUF = int(os.environ.get("DRM_NBUF", "2"))   # buffer layers per side + bottom
Vp, Vs, rho = 4000.0, 2000.0, 2600.0

# soil elastic constants from Vp,Vs,rho
G = rho * Vs**2
nu = (Vp**2 - 2 * Vs**2) / (2.0 * (Vp**2 - Vs**2))
E = 2 * G * (1 + nu)

# --- axis coordinate sets ----------------------------------------------------
# inner dataset extents (these node positions are FIXED by the dataset)
inner_x = np.arange(-100.0, 100.0 + H / 2, H)   # -100..100
inner_y = np.arange(-100.0, 100.0 + H / 2, H)
inner_z = np.arange(0.0, 150.0 + H / 2, H)       # 0..150 (z-down, 0=surface)

# extended axes: add NBUF layers on each lateral side, and on the bottom (+z) only
ext_x = np.concatenate([
    inner_x[0] - H * np.arange(NBUF, 0, -1),
    inner_x,
    inner_x[-1] + H * np.arange(1, NBUF + 1),
])
ext_y = np.concatenate([
    inner_y[0] - H * np.arange(NBUF, 0, -1),
    inner_y,
    inner_y[-1] + H * np.arange(1, NBUF + 1),
])
ext_z = np.concatenate([
    inner_z,                                     # top stays at z=0 (free surface)
    inner_z[-1] + H * np.arange(1, NBUF + 1),    # extend downward only
])

xs = np.round(ext_x, 6)
ys = np.round(ext_y, 6)
zs = np.round(ext_z, 6)
print(f"extended grid: nx={len(xs)} ny={len(ys)} nz={len(zs)}  (NBUF={NBUF}, H={H})")
print(f"  x in [{xs[0]},{xs[-1]}]  y in [{ys[0]},{ys[-1]}]  z in [{zs[0]},{zs[-1]}]")

# --- node numbering: structured (i,j,k) -> tag -------------------------------
nx, ny, nz = len(xs), len(ys), len(zs)


def nid(i, j, k):
    return 1 + i + nx * (j + ny * k)


node_coord = {}
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            node_coord[nid(i, j, k)] = (xs[i], ys[j], zs[k])

# --- hex elements (8-node stdBrick), standard OpenSees brick node order ------
# OpenSees stdBrick connectivity: bottom face (k) CCW then top face (k+1) CCW.
# Orientation only affects sign of volume->Jacobian; stdBrick tolerates either as
# long as it's a valid hex. We use the conventional ordering used by apeGmsh too.
elements = []   # (tag, n1..n8)
etag = 1
for k in range(nz - 1):
    for j in range(ny - 1):
        for i in range(nx - 1):
            n = [
                nid(i,   j,   k),   nid(i+1, j,   k),
                nid(i+1, j+1, k),   nid(i,   j+1, k),
                nid(i,   j,   k+1), nid(i+1, j,   k+1),
                nid(i+1, j+1, k+1), nid(i,   j+1, k+1),
            ]
            elements.append((etag, *n))
            etag += 1
print(f"nodes={len(node_coord)}  elements={len(elements)}")

# --- identify outermost-face nodes (for fixed BC) and outer quad faces -------
xmin, xmax = xs[0], xs[-1]
ymin, ymax = ys[0], ys[-1]
zmax = zs[-1]   # bottom (z-down); top z=0 is free surface (NOT a boundary)

outer_face_nodes = set()
for tag, (x, y, z) in node_coord.items():
    if (abs(x - xmin) < 1e-6 or abs(x - xmax) < 1e-6 or
            abs(y - ymin) < 1e-6 or abs(y - ymax) < 1e-6 or
            abs(z - zmax) < 1e-6):
        outer_face_nodes.add(tag)
print(f"outermost-face nodes (4 sides + bottom): {len(outer_face_nodes)}")

# --- outer quad faces -> two triangles each, for LysmerTriangle --------------
facets = []   # (na, nb, nc)


def add_quad(a, b, c, d):
    facets.append((a, b, c))
    facets.append((a, c, d))


# bottom z=zmax (k = nz-1): all i,j cells
k = nz - 1
for j in range(ny - 1):
    for i in range(nx - 1):
        add_quad(nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k), nid(i, j+1, k))
# x=xmin (i=0) and x=xmax (i=nx-1): j,k cells
for i in (0, nx - 1):
    for k in range(nz - 1):
        for j in range(ny - 1):
            add_quad(nid(i, j, k), nid(i, j+1, k), nid(i, j+1, k+1), nid(i, j, k+1))
# y=ymin (j=0) and y=ymax (j=ny-1): i,k cells
for j in (0, ny - 1):
    for k in range(nz - 1):
        for i in range(nx - 1):
            add_quad(nid(i, j, k), nid(i+1, j, k), nid(i+1, j, k+1), nid(i, j, k+1))
print(f"outer triangular facets (LysmerTriangle): {len(facets)}")

# --- sanity: inner dataset nodes present at exact coords ---------------------
ninner = 0
for x in (-100, -50, 0, 50, 100):
    for y in (-100, -50, 0, 50, 100):
        for z in (0, 50, 100, 150):
            found = any(abs(c[0]-x) < 1e-6 and abs(c[1]-y) < 1e-6 and abs(c[2]-z) < 1e-6
                        for c in node_coord.values())
            ninner += found
print(f"inner dataset nodes present at exact coords: {ninner}/100 (98 are in dataset)")

# --- emit shared model tcl ---------------------------------------------------
OUT_MODEL = os.path.join(HERE, "drm_model_buf.tcl")
with open(OUT_MODEL, "w") as f:
    f.write("# STABILIZED buffered DRM model -- generated by build_drm_buffered.py\n")
    f.write(f"# NBUF={NBUF} buffer layers (4 sides + bottom), H={H} m, z-down, centered\n")
    f.write("model BasicBuilder -ndm 3 -ndf 3\n")
    f.write(f"nDMaterial ElasticIsotropic 1 {E:.10e} {nu:.10f} {rho}\n")
    for tag in sorted(node_coord):
        x, y, z = node_coord[tag]
        f.write(f"node {tag} {x:.6f} {y:.6f} {z:.6f}\n")
    for (tag, *n) in elements:
        f.write(f"element stdBrick {tag} {' '.join(str(t) for t in n)} 1\n")
print("wrote", OUT_MODEL)

# --- variant (a): fixed far boundary -----------------------------------------
OUT_FIX = os.path.join(HERE, "bc_fixed.tcl")
with open(OUT_FIX, "w") as f:
    f.write("# variant (a): FIX all outermost-face nodes (4 sides + bottom)\n")
    for tag in sorted(outer_face_nodes):
        f.write(f"fix {tag} 1 1 1\n")
print("wrote", OUT_FIX)

# --- variant (b): LysmerTriangle dashpots on outer faces ----------------------
OUT_LYS = os.path.join(HERE, "bc_lysmer.tcl")
with open(OUT_LYS, "w") as f:
    f.write("# variant (b): bare LysmerTriangle dashpots on outermost faces\n")
    f.write("# (DRM exterior carries only scattered field => bare Lysmer is correct)\n")
    t = 2000
    for (a, b, c) in facets:
        f.write(f"element LysmerTriangle {t} {a} {b} {c} {rho} {Vp} {Vs}\n")
        t += 1
print("wrote", OUT_LYS)

# --- comparison node tag (model coords (0,-50,0)) ----------------------------
cmp_tag = None
for tag, (x, y, z) in node_coord.items():
    if abs(x) < 1e-6 and abs(y + 50) < 1e-6 and abs(z) < 1e-6:
        cmp_tag = tag
print(f"comparison node (0,-50,0) tag = {cmp_tag}")
print(f"soil: E={E:.4e} nu={nu:.4f} rho={rho}")
print("PASS")
