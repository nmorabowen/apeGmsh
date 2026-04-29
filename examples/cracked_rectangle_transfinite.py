"""
Cracked rectangle — transfinite quad mesh + Crack plugin
========================================================

Geometry: 500 (wide) x 2000 (tall) rectangle, XY plane.
Crack:    vertical slit from (250, 0) to (250, 100), bottom-centre.

This version uses Gmsh's built-in "Crack" plugin (wrapped by
``g.mesh.editing.crack``) to duplicate nodes along the crack curve
*after* meshing.  The geometry is therefore fully conformal — every
shared boundary uses a single curve entity — and the mesh comes out
as a clean transfinite quad grid before the plugin splits the nodes
on the crack curve only.

Domain decomposition (four conformal patches sharing every boundary):

    +-------+-------+   y = H = 2000
    |  TL   |  TR   |
    +-------+-------+   y = CH = 100   <-- crack tip lives here
    |  BL   |  BR   |
    +-------+-------+   y = 0
   x=0    x=CX     x=W
          =250     =500

The single shared curve from (CX, 0) to (CX, CH) is the crack.  The
crack-tip point at (CX, CH) is the open boundary — its node stays
shared after the plugin runs.  The centre line above the crack
(from (CX, CH) to (CX, H)) is a different curve and is not touched
by the plugin.
"""

from apeGmsh import apeGmsh

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
W  = 500.0    # total width
H  = 2000.0   # total height
CX = W / 2    # crack x-position (centre = 250)
CH = 100.0    # crack height

NX       = 11   # nodes per half-width   (10 elements, dx = 25)
NY_CRACK = 6    # nodes in crack zone    (5 elements,  dy = 20)
NY_TOP   = 21   # nodes in top zone      (20 elements, dy ~ 95)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
with apeGmsh(model_name="cracked_rectangle", verbose=True) as g:

    geo = g.model.geometry

    # -----------------------------------------------------------------------
    # Points (one per location — no duplication tricks)
    # -----------------------------------------------------------------------
    p_bl  = geo.add_point(0,   0,   0)   # bottom-left corner
    p_cb  = geo.add_point(CX,  0,   0)   # crack base (single point)
    p_br  = geo.add_point(W,   0,   0)   # bottom-right corner

    p_il  = geo.add_point(0,   CH,  0)   # left edge at y=CH
    p_tip = geo.add_point(CX,  CH,  0)   # crack tip (open boundary)
    p_ir  = geo.add_point(W,   CH,  0)   # right edge at y=CH

    p_tl  = geo.add_point(0,   H,   0)   # top-left corner
    p_tc  = geo.add_point(CX,  H,   0)   # top centre
    p_tr  = geo.add_point(W,   H,   0)   # top-right corner

    # -----------------------------------------------------------------------
    # Curves — every interface is a single shared curve
    # -----------------------------------------------------------------------
    L_bot_L   = geo.add_line(p_bl,  p_cb,  label="bottom_left")
    L_bot_R   = geo.add_line(p_cb,  p_br,  label="bottom_right")

    # THE crack curve (shared by BL and BR — single mesh node set
    # before the plugin runs).
    L_crack   = geo.add_line(p_cb,  p_tip, label="crack")

    # Above-crack centre line (shared by TL and TR — never duplicated).
    L_center  = geo.add_line(p_tip, p_tc,  label="center_upper")

    L_left_L  = geo.add_line(p_bl,  p_il,  label="left_lower")
    L_right_L = geo.add_line(p_br,  p_ir,  label="right_lower")
    L_left_U  = geo.add_line(p_il,  p_tl,  label="left_upper")
    L_right_U = geo.add_line(p_ir,  p_tr,  label="right_upper")

    L_mid_L   = geo.add_line(p_il,  p_tip, label="mid_left")     # BL top = TL bottom
    L_mid_R   = geo.add_line(p_tip, p_ir,  label="mid_right")    # BR top = TR bottom

    L_top_L   = geo.add_line(p_tl,  p_tc,  label="top_left")
    L_top_R   = geo.add_line(p_tc,  p_tr,  label="top_right")

    # -----------------------------------------------------------------------
    # Surfaces (CCW curve loops; BL and BR both reference L_crack so the
    # crack is one entity at mesh time)
    # -----------------------------------------------------------------------
    CL_BL = geo.add_curve_loop([ L_bot_L,  L_crack,    -L_mid_L,  -L_left_L])
    S_BL  = geo.add_plane_surface(CL_BL, label="surf_BL")

    CL_BR = geo.add_curve_loop([ L_bot_R,  L_right_L,  -L_mid_R,  -L_crack])
    S_BR  = geo.add_plane_surface(CL_BR, label="surf_BR")

    CL_TL = geo.add_curve_loop([ L_mid_L,  L_center,   -L_top_L,  -L_left_U])
    S_TL  = geo.add_plane_surface(CL_TL, label="surf_TL")

    CL_TR = geo.add_curve_loop([ L_mid_R,  L_right_U,  -L_top_R,  -L_center])
    S_TR  = geo.add_plane_surface(CL_TR, label="surf_TR")

    # -----------------------------------------------------------------------
    # Physical groups — Crack and CrackTip are required by the plugin
    # -----------------------------------------------------------------------
    g.physical.add_surface([S_BL, S_BR, S_TL, S_TR], name="Domain")
    g.physical.add_curve([L_bot_L, L_bot_R],  name="Bottom")
    g.physical.add_curve([L_top_L, L_top_R],  name="Top")
    g.physical.add_curve([L_left_L, L_left_U], name="Left")
    g.physical.add_curve([L_right_L, L_right_U], name="Right")
    g.physical.add_curve([L_crack],            name="Crack")
    g.physical.add_point([p_cb],               name="CrackBase")  # crack mouth (free surface) — gets duplicated
    g.physical.add_point([p_tip],              name="CrackTip")   # interior tip — stays shared (default)

    # -----------------------------------------------------------------------
    # Transfinite structured mesh
    # -----------------------------------------------------------------------
    st = g.mesh.structured

    for L in (L_bot_L, L_bot_R, L_mid_L, L_mid_R, L_top_L, L_top_R):
        st.set_transfinite_curve(L, NX)

    for L in (L_crack, L_left_L, L_right_L):
        st.set_transfinite_curve(L, NY_CRACK)

    for L in (L_center, L_left_U, L_right_U):
        st.set_transfinite_curve(L, NY_TOP)

    st.set_transfinite_surface(S_BL, corners=[p_bl,  p_cb,  p_tip, p_il])
    st.set_transfinite_surface(S_BR, corners=[p_cb,  p_br,  p_ir,  p_tip])
    st.set_transfinite_surface(S_TL, corners=[p_il,  p_tip, p_tc,  p_tl])
    st.set_transfinite_surface(S_TR, corners=[p_tip, p_ir,  p_tr,  p_tc])

    for S in (S_BL, S_BR, S_TL, S_TR):
        st.set_recombine(S, dim=2)

    # -----------------------------------------------------------------------
    # Mesh + Crack plugin
    # -----------------------------------------------------------------------
    g.mesh.generation.generate(dim=2)

    # Plugin call — duplicates nodes along the "Crack" PG.  By default
    # the two endpoint nodes of the crack curve stay shared; naming
    # CrackBase as the open boundary forces the crack mouth (y=0, on the
    # free bottom surface) to also be duplicated.  The tip at (CX, CH)
    # remains a single shared node — the right convention for a crack
    # that terminates inside the material.
    g.mesh.editing.crack(
        "Crack",
        dim=1,
        open_boundary="CrackBase",
    )

    # -----------------------------------------------------------------------
    # FEM data
    # -----------------------------------------------------------------------
    fem = g.mesh.queries.get_fem_data(dim=2)
    print(fem.inspect.summary())

    import numpy as np

    # All nodes
    print("\n--- All nodes (id, x, y, z) ---")
    for nid, xyz in fem.nodes.get(pg="Domain"):
        print(f"  node {int(nid):5d}  x={xyz[0]:8.2f}  y={xyz[1]:8.2f}  z={xyz[2]:6.2f}")

    # All elements: type + tag + connectivity
    print("\n--- All elements (type, tag, connectivity) ---")
    for group in fem.elements.get(pg="Domain"):
        print(f"  element type : {group.element_type}")
        for eid, conn in zip(group.ids, group.connectivity):
            print(f"  elem {int(eid):5d}  nodes = {[int(n) for n in conn]}")

    # Find duplicate-coordinate node pairs to confirm the crack split.
    print("\n--- Duplicate-coordinate node pairs (crack signature) ---")
    coords = fem.nodes.coords
    ids    = fem.nodes.ids
    seen: dict[tuple, list[int]] = {}
    for nid, xyz in zip(ids, coords):
        key = (round(float(xyz[0]), 6), round(float(xyz[1]), 6), round(float(xyz[2]), 6))
        seen.setdefault(key, []).append(int(nid))
    n_pairs = 0
    for key, nlist in sorted(seen.items(), key=lambda kv: kv[0][1]):
        if len(nlist) > 1:
            n_pairs += 1
            print(f"  coords=({key[0]:6.1f}, {key[1]:6.1f})  nodes={nlist}")
    print(f"Total duplicate locations: {n_pairs}")

    try:
        g.mesh.viewer(fem=fem)
    except ModuleNotFoundError:
        pass  # pyvista not installed
