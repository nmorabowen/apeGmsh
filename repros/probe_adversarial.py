r"""Adversarial probes for the scoping report's load-bearing claims.

P1. contact() in a from_h5 chain session -> silent no-op? (claimed gap:
    contact bypasses _add_def AND the router entirely)
P2. LIVE session: bc() with a misspelled target after extraction is
    silent, but the NEXT get_fem_data() re-extracts and fails loud?
    (the report claims live sessions are eventually-loud)
P3. LIVE session: displacements.surface typo -> same question.
P4. from_h5 session: g.save() after a silently-dropped def — does the
    saved file at least round-trip what the broker has (no corruption)?
"""
import tempfile
from pathlib import Path

import apeGmsh as _pkg
print(f"apeGmsh from: {_pkg.__file__}")
from apeGmsh import apeGmsh

TMP = Path(tempfile.mkdtemp(prefix="apeGmsh_probe_"))


def build_module(fname, name, z0, size):
    path = TMP / fname
    with apeGmsh(model_name=name, save_to=str(path)) as g:
        g.model.geometry.add_box(0, 0, z0, 100, 100, 50, label="blk")
        g.model.sync()
        g.model.select("blk").to_physical(f"{name}_body")
        g.model.select(dim=2).on_plane(
            (0, 0, z0 + 50), (0, 0, 1), tol=1e-3).to_physical(f"{name}_top")
        g.model.select(dim=2).on_plane(
            (0, 0, z0), (0, 0, 1), tol=1e-3).to_physical(f"{name}_bot")
        g.mesh.sizing.set_size("blk", size)
        g.mesh.generation.generate(3)
        g.mesh.queries.get_fem_data(dim=None)
    return path


pa = build_module("mod_a.h5", "A", 0.0, 25.0)
pb = build_module("mod_b.h5", "B", 50.5, 20.0)

# ── P1: contact() in chain phase ────────────────────────────────────
g = apeGmsh.from_h5(pa)
g.compose(str(pb), label="B")
try:
    d = g.constraints.contact("A_top", "B.B_bot", formulation="mortar",
                              tie=True, outward=(0, 0, 1))
    n_defs = len(g.constraints.contact_defs)
    n_recs = len(getattr(g.constraints, "contact_records", []))
    broker_contacts = getattr(g._fem.elements, "contacts", None)
    n_broker = len(list(broker_contacts)) if broker_contacts is not None else "n/a"
    print(f"\n[P1 contact in chain phase] NO EXCEPTION — defs={n_defs}, "
          f"resolved records={n_recs}, broker contact records={n_broker}")
except Exception as exc:  # noqa: BLE001
    print(f"\n[P1 contact in chain phase] raised {type(exc).__name__}: {str(exc)[:120]}")

# ── P2/P3: live session eventual-loudness ───────────────────────────
with apeGmsh(model_name="live") as g2:
    g2.model.geometry.add_box(0, 0, 0, 10, 10, 10, label="blk")
    g2.model.sync()
    g2.model.select("blk").to_physical("Body")
    g2.model.select(dim=2).on_plane((0, 0, 0), (0, 0, 1),
                                    tol=1e-3).to_physical("Base")
    g2.mesh.sizing.set_size("blk", 5.0)
    g2.mesh.generation.generate(3)
    fem1 = g2.mesh.queries.get_fem_data()          # -> live chain phase

    g2.constraints.bc("Base_TYPO", dofs=[1, 1, 1])  # silent at declaration?
    print("\n[P2 live bc typo] declaration raised nothing:",
          len(list(g2._fem.nodes.sp)) == 0)
    try:
        fem2 = g2.mesh.queries.get_fem_data()       # re-extraction
        print(f"[P2 live bc typo] re-extraction NO EXCEPTION — "
              f"sp={len(list(fem2.nodes.sp))}")
    except Exception as exc:  # noqa: BLE001
        print(f"[P2 live bc typo] re-extraction raised "
              f"{type(exc).__name__}: {str(exc)[:120]}")
    g2.constraints._bc_defs.clear()                 # reset for P3

    with g2.displacements.case("push"):
        g2.displacements.surface("Top_TYPO", disp_xyz=(0, 0, -1.0))
    try:
        fem3 = g2.mesh.queries.get_fem_data()
        print(f"[P3 live disp typo] re-extraction NO EXCEPTION — "
              f"sp={len(list(fem3.nodes.sp))}")
    except Exception as exc:  # noqa: BLE001
        print(f"[P3 live disp typo] re-extraction raised "
              f"{type(exc).__name__}: {str(exc)[:120]}")
    g2.displacements.disp_defs.clear()              # let the session close

# ── P4: save after silent drop, from_h5 chain session ───────────────
g3 = apeGmsh.from_h5(pa)
g3.constraints.bc("A_bot_TYPO", dofs=[1, 1, 1])     # silently dropped
out = TMP / "resaved.h5"
try:
    g3.save(str(out))
    from apeGmsh.mesh.FEMData import FEMData
    back = FEMData.from_h5(str(out))
    print(f"\n[P4 save after silent drop] NO EXCEPTION — reloaded "
          f"{back.info.n_nodes} nodes, sp={len(list(back.nodes.sp))}")
except Exception as exc:  # noqa: BLE001
    print(f"\n[P4 save after silent drop] raised {type(exc).__name__}: "
          f"{str(exc)[:120]}")

print(f"\ntmp dir: {TMP}")
