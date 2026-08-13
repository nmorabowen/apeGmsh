"""ADR 0093 S9 — the interface numeric twin: RUN the partitioned decks.

S8/S9 proved the emitted decks' SHAPE; this suite proves their NUMBERS,
on the #944 harness (``test_contact_partitioned_numeric_twin.py``): the
same model definition emits both twins — the serial deck (the
sanctioned ``flat=True`` lane) and the partitioned 2-rank deck — and
both are actually executed, serial under ``OpenSees.exe``, partitioned
under ``mpiexec -n 2 OpenSeesMP.exe``. Every printed quantity must
agree to ``REL_TOL = 1e-10`` (the harness's established gate; the fork
P0 harness measured 1.4e-14 on a deck of this shape).

Two cases, the two halves of ADR 0093's partitioned story:

* **Case A — unclaimed, plain partitioned (S8 at runtime).** The
  S5-probe mechanics on the two-squares model: ENT normal + EPP
  tangential, rock on rank 0 / liner on rank 1 (every slave FOREIGN to
  the owner rank, so the INV-5 ghosting is load-bearing), a PUSH+SHEAR
  deck (interface closed, tangential sheared) and a PULL deck
  (separation — ENT carries nothing and the liner anchor's reaction
  vanishes).

* **Case B — the campaign scenario (S9 at runtime).** Staged
  liner-install under MPI: the ground stage loads the partitioned rock
  (split ACROSS the rock, so both ranks own ground elements in every
  stage), the install stage activates the liner PG and claims the
  interface by name (the whole per-pair unit emits inside the owner
  rank's stage block, slaves ghost-declared across the cut), a
  service stage applies increments. Asserted: the install is
  stress-free (the liner's displacement right after install is orders
  below the ground's — it carries only post-install increments), the
  service increment engages the interface, and every quantity matches
  the serial-staged twin at twin tolerance.

Deck handling follows the template exactly: everything model-side goes
through the real apeGmsh API; the harness only APPENDS a measurement
fragment (``puts`` + ``wipe``; Case A also appends its ``analyze``) to
each emitted deck. Requires the same environment as the template
(``APEGMSH_OPENSEES_BIN`` + Intel MPI) and skips loudly otherwise.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw
from apeGmsh.opensees import apeSees

REL_TOL = 1.0e-10      # the #944 harness's established gate
RUN_TIMEOUT_S = 300
THICKNESS = 0.5

# Stiff normal law (interface deformation << continuum deformation),
# elastic-range tangential (EPP with tau_b far above the demand — the
# yield plateau is S10 acceptance territory, not twin territory).
NORMAL = NormalLaw(kind="ent", k_per_area=1.0e12)
TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e11, tau_b=1.0e9)

F_PUSH = 1.0e6         # per face node, +x (compression through the pairs)
F_SHEAR = 2.0e5        # per face node, +y (tangential, elastic range)
F_GROUND = -1.0e6      # per face node, -x (ground pulls AWAY: separation)
F_SERVICE = 3.0e6      # per face node, +x on top of the frozen ground load


# ---------------------------------------------------------------------------
# Environment gating — identical to the #944 template
# ---------------------------------------------------------------------------


def _dist_bin() -> "Path | None":
    d = os.environ.get("APEGMSH_OPENSEES_BIN")
    if not d:
        return None
    p = Path(d)
    if (p / "OpenSees.exe").is_file() and (p / "OpenSeesMP.exe").is_file():
        return p
    return None


def _mpi_root() -> Path:
    oneapi = os.environ.get(
        "ONEAPI_ROOT", r"C:\Program Files (x86)\Intel\oneAPI")
    return Path(os.environ.get("I_MPI_ROOT") or Path(oneapi) / "mpi" / "latest")


def _mpiexec() -> "Path | None":
    c = _mpi_root() / "bin" / "mpiexec.exe"
    return c if c.is_file() else None


pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.slow,
    pytest.mark.skipif(
        _dist_bin() is None,
        reason=(
            "APEGMSH_OPENSEES_BIN unset or does not hold OpenSees.exe + "
            "OpenSeesMP.exe — point it at a Ladruno-fork dist\\bin to run "
            "the ADR 0093 S9 numeric-twin suite"
        ),
    ),
    pytest.mark.skipif(
        _mpiexec() is None,
        reason=(
            "Intel MPI mpiexec.exe not found (set I_MPI_ROOT, or install "
            "oneAPI at the default location) — needed for the 2-rank lane"
        ),
    ),
]


def _run_env(dist_bin: Path) -> "dict[str, str]":
    env = dict(os.environ)
    oneapi = Path(os.environ.get(
        "ONEAPI_ROOT", r"C:\Program Files (x86)\Intel\oneAPI"))
    mpiroot = _mpi_root()
    dirs = [
        dist_bin,
        mpiroot / "bin",
        mpiroot / "opt" / "mpi" / "libfabric" / "bin",
        oneapi / "compiler" / "latest" / "bin",
    ]
    env["PATH"] = os.pathsep.join(
        [str(d) for d in dirs if d.is_dir()] + [env.get("PATH", "")])
    tcl = dist_bin.parent / "lib" / "tcl8.6"
    if tcl.is_dir():
        env["TCL_LIBRARY"] = str(tcl)
    return env


def _launch(cmd: "list[str]", deck: Path, env: "dict[str, str]") -> str:
    """Run one deck and return its combined output, or fail with the
    ACTUAL cause.

    Both twin lanes are driven from a module-scoped fixture, so anything
    that goes wrong here errors every test in the module at once.  That
    makes diagnosability the whole game: without a returncode check a
    launcher failure (a contended or half-installed MPI runtime, a
    missing DLL, a rank that died mid-solve) surfaces only as the
    downstream "printed no S9TWIN line" parse assert, which reads like a
    product defect and sends the next reader hunting through emit code.
    Check the exit status here and say what actually happened, once.
    """
    try:
        r = subprocess.run(
            cmd, cwd=deck.parent, env=env,
            capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"{deck.name}: {cmd[0]} did not finish within "
            f"{RUN_TIMEOUT_S}s — these are 8-node models, so a timeout "
            f"means a hang or a pathologically loaded machine, not a slow "
            f"solve.\ncmd: {cmd}\n"
            f"output tail:\n{(exc.output or b'')[-2000:]!r}"
        ) from exc
    out = r.stdout + r.stderr
    if r.returncode != 0:
        raise AssertionError(
            f"{deck.name}: {Path(cmd[0]).name} exited {r.returncode} — the "
            f"deck never ran to completion, so the twin comparison below "
            f"would be meaningless.\ncmd: {cmd}\n"
            f"output tail:\n{out[-2000:]}"
        )
    return out


def _run_serial(deck: Path, env: "dict[str, str]") -> str:
    dist = _dist_bin()
    assert dist is not None
    return _launch([str(dist / "OpenSees.exe"), str(deck)], deck, env)


def _run_mp(deck: Path, env: "dict[str, str]") -> str:
    dist, mpi = _dist_bin(), _mpiexec()
    assert dist is not None and mpi is not None
    return _launch(
        [str(mpi), "-n", "2", str(dist / "OpenSeesMP.exe"), str(deck)],
        deck, env)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------------------
# Model — the ADR 0093 two-squares shape through the real apeGmsh API
# ---------------------------------------------------------------------------


def _curve_at_x(surface: int, x: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < tol and abs(bb[3] - x) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _surface_of(dim: int, tag: int) -> int:
    while dim < 2:
        up, _down = gmsh.model.getAdjacencies(dim, tag)
        dim, tag = dim + 1, int(up[0])
    return tag


def _two_squares(g, n: int):
    left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
    right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
    g.model.sync()
    g.mesh.structured.set_transfinite([(2, left), (2, right)], n=n)
    g.mesh.generation.generate(2)
    g.physical.add(2, [left], name="rock")
    g.physical.add(2, [right], name="liner")
    g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
    g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
    g.physical.add(1, [_curve_at_x(left, 0.0)], name="base")
    g.physical.add(1, [_curve_at_x(right, 2.0)], name="anchor")
    return left, right


def _partition_left_right(g) -> None:
    """Case A cut: whole rock body (boundary entities included) on
    partition 1 / rank 0, whole liner body on partition 2 / rank 1 —
    every slave (wire) node is FOREIGN to the owner rank."""
    elem_tags: "list[int]" = []
    parts: "list[int]" = []
    for dim, tag in gmsh.model.getEntities():
        if dim > 2:
            continue
        surf = _surface_of(dim, tag)
        part = 1 if gmsh.model.occ.getCenterOfMass(2, surf)[0] < 1.0 else 2
        _etypes, etags_list, _enodes = gmsh.model.mesh.getElements(
            dim=dim, tag=tag)
        for etags in etags_list:
            for t in etags:
                elem_tags.append(int(t))
                parts.append(part)
    g.mesh.partitioning.partition_explicit(2, elem_tags, parts)


def _partition_rock_split(g) -> None:
    """Case B cut (the campaign shape): the ROCK is split across the
    two ranks at x = 0.5 — both ranks own ground elements in every
    stage (no empty rank during the ground stage) — while the whole
    liner side lands on partition 1 / rank 0. The interface pairs'
    backing rock quads sit at x in (0.5, 1) → owner rank 1, and the
    slave (wire) nodes are native to rank 0 only → ghosted across the
    cut, inside the install stage's owner block."""
    elem_tags: "list[int]" = []
    parts: "list[int]" = []
    for dim, tag in gmsh.model.getEntities():
        if dim > 2:
            continue
        surf = _surface_of(dim, tag)
        liner_side = gmsh.model.occ.getCenterOfMass(2, surf)[0] > 1.0
        _etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(
            dim=dim, tag=tag)
        for etags, enodes in zip(etags_list, enodes_list):
            if len(etags) == 0:
                continue
            nper = len(enodes) // len(etags)
            for k, t in enumerate(etags):
                if liner_side:
                    part = 1
                else:
                    nds = enodes[k * nper:(k + 1) * nper]
                    x = float(np.mean([
                        gmsh.model.mesh.getNode(int(n))[0][0] for n in nds
                    ]))
                    part = 1 if x < 0.5 else 2
                elem_tags.append(int(t))
                parts.append(part)
    g.mesh.partitioning.partition_explicit(2, elem_tags, parts)


def _build_fem(model_name: str, *, cut: str, n: int = 3):
    with apeGmsh(model_name=model_name, verbose=False) as g:
        _two_squares(g, n=n)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, name="RockLiner")
        if cut == "left_right":
            _partition_left_right(g)
        else:
            _partition_rock_split(g)
        fem = g.mesh.queries.get_fem_data()
        assert len(fem.partitions) == 2
        assert fem.elements.interfaces
        return fem


def _probes(fem) -> "dict[str, object]":
    """Mid-height (y = 0.5) probe nodes + per-edge id lists + the two
    runtime ranks (rank_rock ⊇ base, the other holds the liner side
    natively)."""
    coords_of = {
        int(nid): tuple(map(float, xyz[:2]))
        for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords)
    }

    def ids(pg: str) -> "list[int]":
        return sorted(int(n) for n in fem.nodes.select(pg=pg).ids)

    def mid(pg: str) -> int:
        return min(ids(pg), key=lambda n: abs(coords_of[n][1] - 0.5))

    parts = sorted(fem.partitions, key=lambda p: p.id)
    owned = [set(int(n) for n in p.node_ids) for p in parts]
    base = ids("base")
    return {
        "base": base, "anchor": ids("anchor"),
        "face": ids("face"), "wire": ids("wire"),
        "n_face": mid("face"), "n_slave": mid("wire"),
        "rank_rock": next(r for r, o in enumerate(owned)
                          if set(base) <= o),
        "rank_liner_native": next(
            r for r, o in enumerate(owned) if set(ids("wire")) <= o),
    }


def _chain(ops, *, parallel: bool) -> None:
    ops.constraints.Transformation()
    if parallel:
        ops.numberer.ParallelPlain()
        ops.system.Mumps()
    else:
        ops.numberer.RCM()
        ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()


# ---------------------------------------------------------------------------
# Case A — unclaimed, plain partitioned (push+shear / pull)
# ---------------------------------------------------------------------------


def _build_ops_a(fem, *, parallel: bool, fx: float, fy: float) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
    for pg in ("rock", "liner"):
        ops.element.FourNodeQuad(
            pg=pg, thickness=THICKNESS, material=mat,
            plane_type="PlaneStrain")
    ops.fix(pg="base", dofs=(1, 1))
    ops.fix(pg="anchor", dofs=(1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="face", forces=(fx, fy))
    _chain(ops, parallel=parallel)
    return ops


def _fragment_a_serial(p: "dict[str, object]") -> str:
    b = " ".join(str(n) for n in p["base"])       # type: ignore[union-attr]
    a = " ".join(str(n) for n in p["anchor"])     # type: ignore[union-attr]
    return f"""
set ok [analyze 1]
reactions
set Rb 0.0
foreach n {{{b}}} {{ set Rb [expr {{$Rb + [nodeReaction $n 1]}}] }}
set Ra 0.0
foreach n {{{a}}} {{ set Ra [expr {{$Ra + [nodeReaction $n 1]}}] }}
puts [format "S9TWINA lane=serial ok=%d uxm=%.16e uym=%.16e uxs=%.16e uys=%.16e Rb=%.16e Ra=%.16e" $ok [nodeDisp {p['n_face']} 1] [nodeDisp {p['n_face']} 2] [nodeDisp {p['n_slave']} 1] [nodeDisp {p['n_slave']} 2] $Rb $Ra]
wipe
"""


def _fragment_a_mp(p: "dict[str, object]") -> str:
    b = " ".join(str(n) for n in p["base"])       # type: ignore[union-attr]
    a = " ".join(str(n) for n in p["anchor"])     # type: ignore[union-attr]
    return f"""
set ok [analyze 1]
if {{[getPID] == {p['rank_rock']}}} {{
    reactions
    set Rb 0.0
    foreach n {{{b}}} {{ set Rb [expr {{$Rb + [nodeReaction $n 1]}}] }}
    puts [format "S9TWINA lane=mp who=owner ok=%d uxm=%.16e uym=%.16e gxs=%.16e gys=%.16e Rb=%.16e" $ok [nodeDisp {p['n_face']} 1] [nodeDisp {p['n_face']} 2] [nodeDisp {p['n_slave']} 1] [nodeDisp {p['n_slave']} 2] $Rb]
}}
if {{[getPID] == {p['rank_liner_native']}}} {{
    reactions
    set Ra 0.0
    foreach n {{{a}}} {{ set Ra [expr {{$Ra + [nodeReaction $n 1]}}] }}
    puts [format "S9TWINA lane=mp who=liner ok=%d uxs=%.16e uys=%.16e Ra=%.16e" $ok [nodeDisp {p['n_slave']} 1] [nodeDisp {p['n_slave']} 2] $Ra]
}}
wipe
"""


_A_SER_RE = re.compile(
    r"S9TWINA lane=serial ok=(-?\d+) uxm=(\S+) uym=(\S+) uxs=(\S+) "
    r"uys=(\S+) Rb=(\S+) Ra=(\S+)")
_A_MP_OWNER_RE = re.compile(
    r"S9TWINA lane=mp who=owner ok=(-?\d+) uxm=(\S+) uym=(\S+) gxs=(\S+) "
    r"gys=(\S+) Rb=(\S+)")
_A_MP_LINER_RE = re.compile(
    r"S9TWINA lane=mp who=liner ok=(-?\d+) uxs=(\S+) uys=(\S+) Ra=(\S+)")


def _parse_a_serial(out: str) -> "dict[str, float] | None":
    m = _A_SER_RE.search(out)
    if not m:
        return None
    keys = ("ok", "uxm", "uym", "uxs", "uys", "Rb", "Ra")
    return {k: float(m.group(i + 1)) for i, k in enumerate(keys)}


def _parse_a_mp(out: str) -> "dict[str, float] | None":
    mo, ml = _A_MP_OWNER_RE.search(out), _A_MP_LINER_RE.search(out)
    if not (mo and ml):
        return None
    return {
        "ok_owner": float(mo.group(1)), "uxm": float(mo.group(2)),
        "uym": float(mo.group(3)), "gxs": float(mo.group(4)),
        "gys": float(mo.group(5)), "Rb": float(mo.group(6)),
        "ok_liner": float(ml.group(1)), "uxs": float(ml.group(2)),
        "uys": float(ml.group(3)), "Ra": float(ml.group(4)),
    }


def _run_case_a(tmp_dir: Path, env: "dict[str, str]", *,
                fx: float, fy: float, tag: str) -> "dict[str, object]":
    fem = _build_fem(f"s9_twin_a_{tag}", cut="left_right")
    p = _probes(fem)
    serial_deck = tmp_dir / f"a_{tag}_serial.tcl"
    mp_deck = tmp_dir / f"a_{tag}_mp.tcl"
    _build_ops_a(fem, parallel=False, fx=fx, fy=fy).tcl(
        str(serial_deck), flat=True)
    _build_ops_a(fem, parallel=True, fx=fx, fy=fy).tcl(str(mp_deck))
    with open(serial_deck, "a", encoding="utf-8") as f:
        f.write(_fragment_a_serial(p))
    with open(mp_deck, "a", encoding="utf-8") as f:
        f.write(_fragment_a_mp(p))
    serial_out = _run_serial(serial_deck, env)
    mp_out = _run_mp(mp_deck, env)
    serial = _parse_a_serial(serial_out)
    mp = _parse_a_mp(mp_out)
    assert serial is not None, (
        f"case A ({tag}) serial twin printed no S9TWINA line:\n"
        f"{serial_out[-2000:]}")
    assert mp is not None, (
        f"case A ({tag}) mp twin printed no S9TWINA lines:\n"
        f"{mp_out[-2000:]}")
    return {"probes": p, "serial": serial, "mp": mp}


# ---------------------------------------------------------------------------
# Case B — staged liner install (the campaign scenario)
# ---------------------------------------------------------------------------


def _build_ops_b(fem, *, parallel: bool, service: bool) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
    for pg in ("rock", "liner"):
        ops.element.FourNodeQuad(
            pg=pg, thickness=THICKNESS, material=mat,
            plane_type="PlaneStrain")
    ops.fix(pg="base", dofs=(1, 1))

    def chain(s) -> None:
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-10, max_iter=50),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Transformation(),
            numberer=(ops.numberer.ParallelPlain() if parallel
                      else ops.numberer.RCM()),
            system=(ops.system.Mumps() if parallel
                    else ops.system.UmfPack()),
            analysis=ops.analysis.Static(),
        )

    with ops.stage(name="ground") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(pg="face", forces=(F_GROUND, 0.0))
        chain(s)
        s.run(n_increments=1, dt=1.0)
    with ops.stage(name="install") as s:
        s.activate(pgs=["liner"])
        s.interface(name="RockLiner")
        s.fix(pg="anchor", dofs=(1, 1))
        chain(s)
        s.run(n_increments=1, dt=1.0)
    if service:
        with ops.stage(name="service") as s:
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                p.load(pg="face", forces=(F_SERVICE, 0.0))
            chain(s)
            s.run(n_increments=1, dt=1.0)
    return ops


def _fragment_b_serial(p: "dict[str, object]",
                       zl_tag: "int | None" = None) -> str:
    b = " ".join(str(n) for n in p["base"])       # type: ignore[union-attr]
    # ``def`` = the mid pair's zeroLength NORMAL deformation via
    # eleResponse — the born-strain-free claim measured AT SOURCE (and
    # a dead spring cannot answer eleResponse, so the alternative that
    # the element simply never engaged dies here too).
    if zl_tag is None:
        fmt, extra = "", ""
    else:
        fmt = " def=%.16e"
        extra = f" [lindex [eleResponse {zl_tag} deformation] 0]"
    return f"""
reactions
set Rb 0.0
foreach n {{{b}}} {{ set Rb [expr {{$Rb + [nodeReaction $n 1]}}] }}
puts [format "S9TWINB lane=serial uxm=%.16e uxs=%.16e uys=%.16e Rb=%.16e{fmt}" [nodeDisp {p['n_face']} 1] [nodeDisp {p['n_slave']} 1] [nodeDisp {p['n_slave']} 2] $Rb{extra}]
wipe
"""


def _fragment_b_mp(p: "dict[str, object]",
                   zl_tag: "int | None" = None) -> str:
    """Rank 0 (liner-native side, also holds base + anchor) prints the
    slave's native value + the base reaction sum; rank 1 (the pairs'
    owner — it holds the backing rock quads, the slave GHOSTS and the
    zeroLength elements themselves) prints the face value, the ghost
    copy of the slave, and the mid pair's eleResponse deformation."""
    b = " ".join(str(n) for n in p["base"])       # type: ignore[union-attr]
    if zl_tag is None:
        owner_fmt = "uxm=%.16e gxs=%.16e"
        owner_args = (f"[nodeDisp {p['n_face']} 1] "
                      f"[nodeDisp {p['n_slave']} 1]")
    else:
        owner_fmt = "uxm=%.16e gxs=%.16e def=%.16e"
        owner_args = (f"[nodeDisp {p['n_face']} 1] "
                      f"[nodeDisp {p['n_slave']} 1] "
                      f"[lindex [eleResponse {zl_tag} deformation] 0]")
    return f"""
if {{[getPID] == {p['rank_rock']}}} {{
    reactions
    set Rb 0.0
    foreach n {{{b}}} {{ set Rb [expr {{$Rb + [nodeReaction $n 1]}}] }}
    puts [format "S9TWINB lane=mp who=linernative uxs=%.16e uys=%.16e Rb=%.16e" [nodeDisp {p['n_slave']} 1] [nodeDisp {p['n_slave']} 2] $Rb]
}}
if {{[getPID] == {1 - p['rank_rock']}}} {{
    puts [format "S9TWINB lane=mp who=owner {owner_fmt}" {owner_args}]
}}
wipe
"""


_B_SER_RE = re.compile(
    r"S9TWINB lane=serial uxm=(\S+) uxs=(\S+) uys=(\S+) Rb=(\S+)"
    r"(?: def=(\S+))?")
_B_MP_NATIVE_RE = re.compile(
    r"S9TWINB lane=mp who=linernative uxs=(\S+) uys=(\S+) Rb=(\S+)")
_B_MP_OWNER_RE = re.compile(
    r"S9TWINB lane=mp who=owner uxm=(\S+) gxs=(\S+)(?: def=(\S+))?")


def _parse_b_serial(out: str) -> "dict[str, float | None] | None":
    m = _B_SER_RE.search(out)
    if not m:
        return None
    return {"uxm": float(m.group(1)), "uxs": float(m.group(2)),
            "uys": float(m.group(3)), "Rb": float(m.group(4)),
            "def0": float(m.group(5)) if m.group(5) else None}


def _parse_b_mp(out: str) -> "dict[str, float | None] | None":
    mn = _B_MP_NATIVE_RE.search(out)
    mo = _B_MP_OWNER_RE.search(out)
    if not (mn and mo):
        return None
    return {"uxs": float(mn.group(1)), "uys": float(mn.group(2)),
            "Rb": float(mn.group(3)),
            "uxm": float(mo.group(1)), "gxs": float(mo.group(2)),
            "def0": float(mo.group(3)) if mo.group(3) else None}


def _run_case_b(tmp_dir: Path, env: "dict[str, str]", *,
                service: bool, tag: str) -> "dict[str, object]":
    fem = _build_fem(f"s9_twin_b_{tag}", cut="rock_split")
    p = _probes(fem)
    # The campaign cut sanity: the slave side is native to exactly one
    # rank, and the owner (backing-element) side is the OTHER one.
    assert p["rank_liner_native"] == p["rank_rock"], (
        "fixture: base and wire should share rank 0 under the "
        "rock-split cut")
    serial_deck = tmp_dir / f"b_{tag}_serial.tcl"
    mp_deck = tmp_dir / f"b_{tag}_mp.tcl"
    _build_ops_b(fem, parallel=False, service=service).tcl(
        str(serial_deck), flat=True)
    _build_ops_b(fem, parallel=True, service=service).tcl(str(mp_deck))

    # The install lane also reads the mid pair's zeroLength deformation
    # via eleResponse (F5): recover its element tag from each emitted
    # deck (equal-ndf: iNode = mid master, jNode = mid slave).
    zl_serial = zl_mp = None
    if not service:
        mid_master = next(
            int(r.master_node) for r in fem.elements.interfaces
            if int(r.slave_node) == p["n_slave"])

        def _zl_tag(path: Path) -> int:
            m = re.search(
                rf"element zeroLength (\d+) {mid_master} {p['n_slave']} ",
                path.read_text(encoding="utf-8"))
            assert m, f"mid pair zeroLength not found in {path.name}"
            return int(m.group(1))

        zl_serial, zl_mp = _zl_tag(serial_deck), _zl_tag(mp_deck)
        # Serial-staged ↔ partitioned-staged tag identity (the S9 tag
        # pre-pass) — the same element answers in both lanes.
        assert zl_serial == zl_mp

    with open(serial_deck, "a", encoding="utf-8") as f:
        f.write(_fragment_b_serial(p, zl_serial))
    with open(mp_deck, "a", encoding="utf-8") as f:
        f.write(_fragment_b_mp(p, zl_mp))
    serial_out = _run_serial(serial_deck, env)
    mp_out = _run_mp(mp_deck, env)
    serial = _parse_b_serial(serial_out)
    mp = _parse_b_mp(mp_out)
    assert serial is not None, (
        f"case B ({tag}) serial twin printed no S9TWINB line "
        f"(a failed stage errors the deck before the fragment):\n"
        f"{serial_out[-2000:]}")
    assert mp is not None, (
        f"case B ({tag}) mp twin printed no S9TWINB lines:\n"
        f"{mp_out[-2000:]}")
    return {"probes": p, "serial": serial, "mp": mp}


# ---------------------------------------------------------------------------
# The twins — built + run ONCE per module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def twins(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, object]":
    env = _run_env(_dist_bin())  # type: ignore[arg-type]
    d = tmp_path_factory.mktemp("s9_twin")
    return {
        "push": _run_case_a(d, env, fx=F_PUSH, fy=F_SHEAR, tag="push"),
        "pull": _run_case_a(d, env, fx=-F_PUSH, fy=0.0, tag="pull"),
        "install": _run_case_b(d, env, service=False, tag="install"),
        "service": _run_case_b(d, env, service=True, tag="service"),
    }


# ---------------------------------------------------------------------------
# Case A gates
# ---------------------------------------------------------------------------


def test_case_a_both_twins_converge(twins) -> None:
    for tag in ("push", "pull"):
        s, m = twins[tag]["serial"], twins[tag]["mp"]
        assert s["ok"] == 0, f"case A {tag}: serial did not converge: {s}"
        assert m["ok_owner"] == 0 and m["ok_liner"] == 0, (
            f"case A {tag}: 2-rank twin did not converge: {m}")


def test_case_a_push_matches_serial(twins) -> None:
    s, m = twins["push"]["serial"], twins["push"]["mp"]
    deltas = {k: _rel(s[k], m[k])
              for k in ("uxm", "uym", "uxs", "uys", "Rb", "Ra")}
    print(f"\nS9 case A (push+shear) serial-vs-2-rank deltas: {deltas}")
    bad = {k: v for k, v in deltas.items() if v > REL_TOL}
    assert not bad, (
        f"case A push twin diverged beyond {REL_TOL:.0e}: {bad}\n"
        f"serial={s}\nmp={m}")


def test_case_a_push_physics(twins) -> None:
    """Compression flows through the pairs; the tangential shear slips
    elastically (master leads the slave in both directions); the two
    fixed edges carry the whole load."""
    s = twins["push"]["serial"]
    n_face = 3
    assert s["uxs"] > 0 and s["uxm"] > s["uxs"]          # closed + compliant
    assert s["uys"] > 0 and s["uym"] > s["uys"]          # sheared, elastic
    assert _rel(s["Rb"] + s["Ra"], -n_face * F_PUSH) < 1e-9


def test_case_a_pull_matches_serial_and_separates(twins) -> None:
    s, m = twins["pull"]["serial"], twins["pull"]["mp"]
    deltas = {k: _rel(s[k], m[k]) for k in ("uxm", "uxs", "Rb")}
    print(f"\nS9 case A (pull) serial-vs-2-rank deltas: {deltas}")
    bad = {k: v for k, v in deltas.items() if v > REL_TOL}
    assert not bad, f"case A pull twin diverged: {bad}\nserial={s}\nmp={m}"
    # Separation signature: ENT carries nothing across the open pairs,
    # so the WHOLE load returns through the rock base and the liner
    # anchor reaction vanishes — in both lanes. (The liner still shows
    # a small residual x-motion: the bilateral tangential springs at
    # the face CORNERS pick up the corners' antisymmetric u_y and drag
    # the liner second-order — measured ~2% of the master's motion —
    # so the honest zero here is the anchor FORCE, not the liner
    # displacement.)
    assert s["uxm"] < 0
    assert abs(s["Ra"]) < 1e-9 * abs(s["Rb"]), f"serial anchor loaded: {s}"
    assert abs(m["Ra"]) < 1e-9 * abs(m["Rb"]), f"mp anchor loaded: {m}"
    assert abs(s["uxs"]) < 0.05 * abs(s["uxm"])


def test_case_a_ghost_equals_native(twins) -> None:
    """The owner rank's ghost of the slave node reports the same value
    the slave's native rank reports (the #944 measured-bit-identical
    property, now for interface ghosts)."""
    for tag in ("push", "pull"):
        m = twins[tag]["mp"]
        assert _rel(m["gxs"], m["uxs"]) < 1e-14, (
            f"case A {tag}: ghost/native mismatch: "
            f"gxs={m['gxs']!r} uxs={m['uxs']!r}")
        if tag == "push":
            assert _rel(m["gys"], m["uys"]) < 1e-14


# ---------------------------------------------------------------------------
# Case B gates — the campaign scenario
# ---------------------------------------------------------------------------


def test_case_b_install_is_stress_free_and_matches_serial(twins) -> None:
    """The S7-probe behaviour, now under MPI: right after the install
    stage the liner has essentially not moved — the ground displacement
    accumulated BEFORE the install is not transferred to it (the ground
    pulled away, the ENT pairs are open, and the tangential drag is
    orders below the ground motion). Serial-staged and
    partitioned-staged agree at twin tolerance."""
    s, m = twins["install"]["serial"], twins["install"]["mp"]
    deltas = {"uxm": _rel(s["uxm"], m["uxm"]), "Rb": _rel(s["Rb"], m["Rb"])}
    print(f"\nS9 case B (install) serial-vs-2-rank deltas: {deltas} "
          f"eleResponse def0: serial={s['def0']!r} mp={m['def0']!r}")
    bad = {k: v for k, v in deltas.items() if v > REL_TOL}
    assert not bad, (
        f"case B install twin diverged: {bad}\nserial={s}\nmp={m}")
    scale = abs(s["uxm"])
    assert s["uxm"] < 0                       # ground moved away, and…
    for lane, got in (("serial", s), ("mp", m)):
        assert abs(got["uxs"]) < 1e-3 * scale, (
            f"{lane}: liner moved at install — the install is NOT "
            f"stress-free: uxs={got['uxs']!r} vs ground {scale!r}")
    # The near-zero liner numbers themselves agree between lanes
    # (absolute, at twin scale).
    assert abs(s["uxs"] - m["uxs"]) < REL_TOL * scale
    assert abs(s["uys"] - m["uys"]) < REL_TOL * scale
    # F5 — the born-strain-free claim AT SOURCE: the mid pair's own
    # eleResponse deformation is (numerically exactly) zero right
    # after install, against a raw relative displacement of ~2.3e-4 —
    # so the zero is an engineered reference offset, not a dead
    # spring (a dead spring could not answer eleResponse at all, and
    # the parse above would have failed).
    raw_rel = abs(s["uxs"] - s["uxm"])
    assert raw_rel > 1e-5, f"fixture lost its ground gap: {s}"
    for lane, got in (("serial", s), ("mp", m)):
        assert got["def0"] is not None, f"{lane}: no def= printed"
        assert abs(got["def0"]) < 1e-12 * raw_rel, (
            f"{lane}: install-stage zeroLength deformation is not "
            f"born-zero: def={got['def0']!r} vs raw relative "
            f"displacement {raw_rel!r}")


def test_case_b_service_increment_engages_and_matches_serial(twins) -> None:
    """The post-install increment: the service load reverses the
    ground pull and immediately loads the liner THROUGH the
    stage-claimed interface — the staged zeroLength is born
    STRAIN-FREE on the equilibrated ground (measured: the compression
    ordering holds in the DISPLACEMENT INCREMENTS from the install
    state, not in total displacements — the ground displacement
    accumulated before the install never strains the pairs, the
    INV-6 install-on-equilibrated-ground semantics). Every quantity
    matches the serial-staged twin."""
    s, m = twins["service"]["serial"], twins["service"]["mp"]
    deltas = {k: _rel(s[k], m[k]) for k in ("uxm", "uxs", "Rb")}
    print(f"\nS9 case B (service) serial-vs-2-rank deltas: {deltas}")
    bad = {k: v for k, v in deltas.items() if v > REL_TOL}
    assert not bad, (
        f"case B service twin diverged: {bad}\nserial={s}\nmp={m}")
    # Engaged, in increment space: from the install state (the B1 twin
    # — same mesh, same stage history through install) the face moved
    # +x and the liner tracked it, lagging by the finite interface
    # compliance — orders above the install-stage residual.
    s_inst = twins["install"]["serial"]
    d_face = s["uxm"] - s_inst["uxm"]
    d_liner = s["uxs"] - s_inst["uxs"]
    assert d_face > 0
    assert 0 < d_liner <= d_face, (
        f"post-install increment ordering broken: d_liner={d_liner!r} "
        f"d_face={d_face!r}")
    assert abs(s["uxs"]) > 100 * max(abs(s_inst["uxs"]), 1e-16)


def test_case_b_ghost_equals_native(twins) -> None:
    """The stage-declared ghost (minted inside the install stage's
    owner block, ADR 0093 S9) tracks the native value bit-close."""
    for tag in ("install", "service"):
        m = twins[tag]["mp"]
        scale = max(abs(m["uxs"]), abs(m["uxm"]), 1e-16)
        assert abs(m["gxs"] - m["uxs"]) < 1e-12 * scale, (
            f"case B {tag}: stage-ghost/native mismatch: "
            f"gxs={m['gxs']!r} uxs={m['uxs']!r}")
