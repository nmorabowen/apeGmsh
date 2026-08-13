"""ADR 0092 S5 — the numeric twin: RUN the partitioned contact deck.

S4 proved the emitted deck's SHAPE; this suite proves its NUMBERS. The
same model definition emits both twins — the serial deck (the sanctioned
``flat=True`` lane) and the partitioned 2-rank deck — and both are
actually executed: serial under ``OpenSees.exe``, partitioned under
``mpiexec -n 2 OpenSeesMP.exe``. The printed quantities (top-block tip
displacement, slave-interface displacement, master-face displacement,
base-reaction sum) must agree to ``REL_TOL = 1e-10`` (the fork P0 harness
measured 1.4e-14 on the hand-written deck of this exact shape,
``OpenSees/Ladruno_files/testbed/contact_parallel/mp.tcl``).

The model mirrors fork ADR-78 P0: two stacked 1x1x1 elastic blocks
(one ``stdBrick`` each, via transfinite hex meshing), 1e-3 initial
penetration so contact is active at the first Newton iteration, lateral
DOFs fixed everywhere (a 1-D column), base fixed, 4x(-2500) on the top
face, ``g.constraints.contact(kn="auto", outward=(0,0,1))`` — the
explicit outward is the documented requirement for an initially-
coincident flat interface. METIS at ``partition(2)`` naturally puts one
body per rank, so the S4 owner resolution + ghosting engage for real.

Negative controls (the comparison must be able to detect a wrong deck):

(a) the same model WITHOUT the contact declaration — the top block has
    no vertical support, so the serial run must fail / diverge (contact
    is load-bearing in the comparison);
(b) the ghost ``fix`` replay lines stripped from the owner's block
    (text surgery on the emitted deck) — cross-rank DOF handling breaks
    and the run must not reproduce the serial answer (measured: Mumps
    "Matrix is Singular Numerically", ADR 0027 INV-2);
(c) the contact verb duplicated into the second rank's block (master
    interface ghost-declared there by the mutation, the replayed verb
    carrying an EXPLICIT kn — see ``_duplicate_verb_on_other_rank`` for
    why the ``auto`` token would be silently inert there) — the ADR-78
    P0.d double-enforcement class. Measured on the current fork: the
    two would-be owners desynchronize the handler's collectives and the
    run DEADLOCKS (handled by the tree-killing ``_run_mp_may_hang``);
    a numeric completion or a loud teardown would equally count as
    detected.

Deck handling: everything model-side goes through the real apeGmsh API
(no hand-patching); the harness only APPENDS a measurement fragment
(``analyze`` + ``puts`` + ``wipe``) to each emitted deck, the same
append-only convention as ``test_tcl_invocation.py``. The reference
twins declare the full analysis chain (``LadrunoContact``/numberer/
system/test/algorithm/integrator/analysis) explicitly through
``ops.<family>``. The formerly-hazardous auto-emit lane — relying on
the auto-emitted handler while declaring ``analysis Static``, which
used to place the handler AFTER the ``analysis`` line where the Tcl
engine ignores it (the ADR 0092 S5 open item; measured: the serial
twin diverged, the 2-rank twin converged plausible-wrong) — is now
fixed (auto-emits hoisted before the user ``analysis`` directive) and
pinned numerically by the ``*_auto_chain`` regression tests below.

Requires (loud skip otherwise — CI has neither, so CI skips):

* ``APEGMSH_OPENSEES_BIN`` pointing at a fork ``dist\\bin`` that holds
  ``OpenSees.exe`` + ``OpenSeesMP.exe`` (the ``run_ladruno_integration``
  convention; the same directory the live lane uses for the pyd);
* Intel MPI's ``mpiexec`` (``I_MPI_ROOT``, or the default oneAPI
  location). The launch PATH recipe — libfabric under
  ``%I_MPI_ROOT%\\opt\\mpi\\libfabric\\bin``, the compiler runtime, and
  the binaries dir itself (MKL) — is the fork's proven one.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

REL_TOL = 1.0e-10   # P0 measured 1.4e-14; margin for model differences.
PENETRATION = 1.0e-3   # initial overlap, P0's 0.999 plane.
RUN_TIMEOUT_S = 300

# ---------------------------------------------------------------------------
# Environment gating — loud skips, never hardcoded paths
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
            "the ADR 0092 S5 numeric-twin suite"
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
    """The proven fork MPI launch environment (auto_penalty_mpi recipe).

    The libfabric bin is ``%I_MPI_ROOT%\\opt\\mpi\\libfabric\\bin`` — NOT
    ``mpi\\latest\\libfabric`` — missing it kills every rank inside
    ``MPIDI_OFI_mpi_init_hook`` with an error that names nothing.
    """
    env = dict(os.environ)
    oneapi = Path(os.environ.get(
        "ONEAPI_ROOT", r"C:\Program Files (x86)\Intel\oneAPI"))
    mpiroot = _mpi_root()
    dirs = [
        dist_bin,                                    # MKL / libiomp DLLs
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


# ---------------------------------------------------------------------------
# Model — the fork P0 shape through the real apeGmsh API
# ---------------------------------------------------------------------------


def _face_at_z(volume_tag: int, z: float, tol: float = 1e-4) -> int:
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of vol {volume_tag} at z={z}")


def _build_fem(model_name: str, *, contact: bool = True):
    """Two stacked single-hex blocks, 1e-3 initial penetration, np=2."""
    with apeGmsh(model_name=model_name, verbose=False) as g:
        z0 = 1.0 - PENETRATION
        box1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        box2 = g.model.geometry.add_box(0, 0, z0, 1, 1, 1)
        g.model.sync()
        master = _face_at_z(box1, 1.0)
        slave = _face_at_z(box2, z0)
        base = _face_at_z(box1, 0.0)
        top = _face_at_z(box2, z0 + 1.0)
        g.mesh.structured.set_transfinite_box(box1, n=2)   # 1 hex / block
        g.mesh.structured.set_transfinite_box(box2, n=2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box1, box2], name="solid")
        g.physical.add(2, [master], name="master")
        g.physical.add(2, [slave], name="slave")
        g.physical.add(2, [base], name="base")
        g.physical.add(2, [top], name="top")
        if contact:
            # outward= is required here: initially-coincident FLAT
            # interface (the one documented case for an explicit normal).
            g.constraints.contact(
                "master", "slave", formulation="nts", kn="auto",
                outward=(0.0, 0.0, 1.0), name="s5twin",
            )
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=3)
        assert len(fem.partitions) == 2
        return fem


def _build_ops(fem, *, parallel_chain: bool, contact: bool = True) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=2.0e7, nu=0.0)
    ops.element.stdBrick(pg="solid", material=mat)
    ops.fix(pg="base", dofs=(1, 1, 1))
    for pg in ("master", "slave", "top"):   # lateral-only: a 1-D column
        ops.fix(pg=pg, dofs=(1, 1, 0))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="top", forces=(0.0, 0.0, -2500.0))   # 4 nodes -> -1e4
    # The FULL chain is declared through the API — the explicit-chain
    # reference lane (see module docstring; the auto-emit lane is
    # covered by the *_auto_chain regression tests).
    if contact:
        ops.constraints.LadrunoContact()
    else:
        ops.constraints.Plain()
    if parallel_chain:
        ops.numberer.ParallelPlain()
        ops.system.Mumps()
    else:
        ops.numberer.RCM()
        ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


# ---------------------------------------------------------------------------
# Probes + the appended measurement fragments (append-only, never edits)
# ---------------------------------------------------------------------------


def _probes(fem) -> "dict[str, object]":
    def ids(pg: str) -> "list[int]":
        return sorted(int(n) for n in fem.nodes.select(pg=pg).ids)

    parts = sorted(fem.partitions, key=lambda p: p.id)
    owned = [set(int(n) for n in p.node_ids) for p in parts]
    base, top = ids("base"), ids("top")
    slave, master = ids("slave"), ids("master")
    return {
        "base": base, "top": top, "slave": slave, "master": master,
        "n_top": top[0], "n_slave": slave[0], "n_master": master[0],
        "rank_bottom": next(r for r, o in enumerate(owned)
                            if set(base) <= o),
        "rank_top": next(r for r, o in enumerate(owned) if set(top) <= o),
    }


def _serial_fragment(p: "dict[str, object]") -> str:
    b = " ".join(str(n) for n in p["base"])   # type: ignore[union-attr]
    return f"""
set ok [analyze 1]
reactions
set R 0.0
foreach n {{{b}}} {{ set R [expr {{$R + [nodeReaction $n 3]}}] }}
puts [format "S5TWIN lane=serial ok=%d wtop=%.16e wslave=%.16e wmaster=%.16e R=%.16e" $ok [nodeDisp {p['n_top']} 3] [nodeDisp {p['n_slave']} 3] [nodeDisp {p['n_master']} 3] $R]
wipe
"""


def _mp_fragment(p: "dict[str, object]") -> str:
    b = " ".join(str(n) for n in p["base"])   # type: ignore[union-attr]
    return f"""
set ok [analyze 1]
if {{[getPID] == {p['rank_bottom']}}} {{
    reactions
    set R 0.0
    foreach n {{{b}}} {{ set R [expr {{$R + [nodeReaction $n 3]}}] }}
    puts [format "S5TWIN lane=mp who=owner ok=%d wmaster=%.16e wghost=%.16e R=%.16e" $ok [nodeDisp {p['n_master']} 3] [nodeDisp {p['n_slave']} 3] $R]
}}
if {{[getPID] == {p['rank_top']}}} {{
    puts [format "S5TWIN lane=mp who=slaveowner ok=%d wtop=%.16e wslave=%.16e" $ok [nodeDisp {p['n_top']} 3] [nodeDisp {p['n_slave']} 3]]
}}
wipe
"""


_SER_RE = re.compile(
    r"S5TWIN lane=serial ok=(-?\d+) wtop=(\S+) wslave=(\S+) "
    r"wmaster=(\S+) R=(\S+)")
_MP_OWNER_RE = re.compile(
    r"S5TWIN lane=mp who=owner ok=(-?\d+) wmaster=(\S+) wghost=(\S+) "
    r"R=(\S+)")
_MP_SLAVE_RE = re.compile(
    r"S5TWIN lane=mp who=slaveowner ok=(-?\d+) wtop=(\S+) wslave=(\S+)")


def _parse_serial(out: str) -> "dict[str, float] | None":
    m = _SER_RE.search(out)
    if not m:
        return None
    return {"ok": float(m.group(1)), "wtop": float(m.group(2)),
            "wslave": float(m.group(3)), "wmaster": float(m.group(4)),
            "R": float(m.group(5))}


def _parse_mp(out: str) -> "dict[str, float] | None":
    mo, ms = _MP_OWNER_RE.search(out), _MP_SLAVE_RE.search(out)
    if not (mo and ms):
        return None
    return {"ok_owner": float(mo.group(1)), "wmaster": float(mo.group(2)),
            "wghost": float(mo.group(3)), "R": float(mo.group(4)),
            "ok_slave": float(ms.group(1)), "wtop": float(ms.group(2)),
            "wslave": float(ms.group(3))}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _run_serial(deck: Path, env: "dict[str, str]") -> str:
    dist = _dist_bin()
    assert dist is not None
    r = subprocess.run(
        [str(dist / "OpenSees.exe"), str(deck)], cwd=deck.parent, env=env,
        capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
    return r.stdout + r.stderr


def _run_mp(deck: Path, env: "dict[str, str]") -> str:
    dist, mpi = _dist_bin(), _mpiexec()
    assert dist is not None and mpi is not None
    r = subprocess.run(
        [str(mpi), "-n", "2", str(dist / "OpenSeesMP.exe"), str(deck)],
        cwd=deck.parent, env=env,
        capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
    return r.stdout + r.stderr


def _run_mp_may_hang(
    deck: Path, env: "dict[str, str]", timeout_s: int = 90,
) -> "tuple[str, bool]":
    """``_run_mp`` for decks that may DEADLOCK (the dup-verb control:
    two ranks both enforcing one interaction desynchronizes the fork
    handler's collectives and the run never completes).

    Windows-specific traps the plain runner cannot survive:
    ``subprocess.run(capture_output=True)`` blocks past its timeout
    because the orphaned rank processes inherit the pipe handles;
    killing ``mpiexec`` alone leaves the ranks alive; and even
    ``taskkill /T`` misses them (measured — the ``hydra_pmi_proxy``
    layer survives the tree walk).  So: stdout to a FILE, then a
    targeted sweep that stops every ``OpenSeesMP.exe`` whose command
    line names THIS deck, plus each one's ``hydra_pmi_proxy`` parent.
    Returns ``(output_text, timed_out)``."""
    dist, mpi = _dist_bin(), _mpiexec()
    assert dist is not None and mpi is not None
    out_file = deck.with_suffix(".out")
    with open(out_file, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [str(mpi), "-n", "2", str(dist / "OpenSeesMP.exe"), str(deck)],
            cwd=deck.parent, env=env,
            stdout=fh, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True)
            _kill_ranks_running_deck(deck)
            proc.wait(timeout=30)
            timed_out = True
    return out_file.read_text(encoding="utf-8", errors="replace"), timed_out


def _kill_ranks_running_deck(deck: Path) -> None:
    """Stop every ``OpenSeesMP.exe`` whose command line references
    ``deck`` (match by the unique tmp-dir'd file name), then its
    ``hydra_pmi_proxy`` parent.  Scoped to this deck so concurrent MPI
    runs are untouched."""
    ps = (
        "$ranks = Get-CimInstance Win32_Process "
        "-Filter \"Name='OpenSeesMP.exe'\" | "
        f"Where-Object {{ $_.CommandLine -match [regex]::Escape('{deck.name}') }}; "
        "$parents = $ranks | ForEach-Object { $_.ParentProcessId } | "
        "Select-Object -Unique; "
        "$ranks | ForEach-Object { "
        "try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } "
        "catch {} }; "
        "foreach ($p in $parents) { "
        "$par = Get-CimInstance Win32_Process -Filter \"ProcessId=$p\"; "
        "if ($par -and $par.Name -eq 'hydra_pmi_proxy.exe') { "
        "try { Stop-Process -Id $p -Force -ErrorAction Stop } catch {} } }"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps], capture_output=True,
        timeout=60,
    )


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------------------
# The good twins — built + run ONCE per module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def twin(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, object]":
    fem = _build_fem("s5_numeric_twin")
    p = _probes(fem)
    d = tmp_path_factory.mktemp("s5_twin")
    serial_deck, mp_deck = d / "serial.tcl", d / "mp.tcl"

    _build_ops(fem, parallel_chain=False).tcl(str(serial_deck), flat=True)
    _build_ops(fem, parallel_chain=True).tcl(str(mp_deck))
    mp_deck_text = mp_deck.read_text(encoding="utf-8")   # pre-fragment

    with open(serial_deck, "a", encoding="utf-8") as f:
        f.write(_serial_fragment(p))
    with open(mp_deck, "a", encoding="utf-8") as f:
        f.write(_mp_fragment(p))

    env = _run_env(_dist_bin())  # type: ignore[arg-type]
    serial_out = _run_serial(serial_deck, env)
    mp_out = _run_mp(mp_deck, env)
    serial = _parse_serial(serial_out)
    mp = _parse_mp(mp_out)
    assert serial is not None, f"serial twin printed no S5TWIN line:\n{serial_out[-2000:]}"
    assert mp is not None, f"mp twin printed no S5TWIN lines:\n{mp_out[-2000:]}"
    return {"fem": fem, "probes": p, "env": env, "dir": d,
            "mp_deck_text": mp_deck_text,
            "serial": serial, "mp": mp,
            "serial_out": serial_out, "mp_out": mp_out}


def test_both_twins_converge(twin) -> None:
    s, m = twin["serial"], twin["mp"]
    assert s["ok"] == 0, f"serial twin did not converge: {s}"
    assert m["ok_owner"] == 0 and m["ok_slave"] == 0, (
        f"2-rank twin did not converge: {m}")


def test_partitioned_twin_matches_serial(twin) -> None:
    """The S5 gate: every printed quantity agrees to REL_TOL = 1e-10."""
    s, m = twin["serial"], twin["mp"]
    deltas = {
        "wtop": _rel(s["wtop"], m["wtop"]),
        "wslave": _rel(s["wslave"], m["wslave"]),
        "wmaster": _rel(s["wmaster"], m["wmaster"]),
        "R": _rel(s["R"], m["R"]),
    }
    print(f"\nS5 measured serial-vs-2-rank relative deltas: {deltas}")
    bad = {k: v for k, v in deltas.items() if v > REL_TOL}
    assert not bad, (
        f"partitioned twin diverged from serial beyond {REL_TOL:.0e}: "
        f"{bad}\nserial={s}\nmp={m}"
    )


def test_physics_sanity(twin) -> None:
    """Anchor against the analytic column, not just self-consistency.

    Base reaction balances the full load (1e4); the master face carries
    exactly the bottom block's elastic shortening P*h/(E*A) = 5e-4.
    """
    s = twin["serial"]
    assert _rel(s["R"], 1.0e4) < 1e-9
    assert _rel(s["wmaster"], -5.0e-4) < 1e-6
    # Ordering: the top block sits below the master face by the residual
    # interface approach; everything moves down.
    assert s["wtop"] < s["wslave"] < s["wmaster"] < 0


def test_ghost_displacement_equals_native_owner_value(twin) -> None:
    """The owner rank's ghost of the slave node reports the SAME value the
    slave's native rank reports (P0 measured bit-identical)."""
    m = twin["mp"]
    assert _rel(m["wghost"], m["wslave"]) < 1e-14, (
        f"ghost/native mismatch: wghost={m['wghost']!r} "
        f"wslave={m['wslave']!r}"
    )


# ---------------------------------------------------------------------------
# Negative control (a) — contact removed: the comparison's subject is
# load-bearing
# ---------------------------------------------------------------------------


def test_control_no_contact_fails_or_diverges(
    twin, tmp_path: Path,
) -> None:
    fem = _build_fem("s5_no_contact", contact=False)
    p = _probes(fem)
    deck = tmp_path / "serial_nocontact.tcl"
    _build_ops(fem, parallel_chain=False, contact=False).tcl(
        str(deck), flat=True)
    with open(deck, "a", encoding="utf-8") as f:
        f.write(_serial_fragment(p))
    out = _run_serial(deck, twin["env"])
    got = _parse_serial(out)
    assert got is not None, f"control printed nothing:\n{out[-2000:]}"
    s = twin["serial"]
    detected = got["ok"] != 0 or _rel(got["wtop"], s["wtop"]) > 1e-3
    assert detected, (
        "the no-contact control reproduced the contact answer — contact "
        f"is not load-bearing in this comparison: control={got} twin={s}"
    )


# ---------------------------------------------------------------------------
# Negative controls (b)/(c) — deck mutations must be detected
# ---------------------------------------------------------------------------

_RANK_OPEN_RE = re.compile(r"^\s*if \{\[getPID\] == (\d+)\} \{\s*$")


def _rank_spans(text: str) -> "tuple[list[str], list[tuple[int, int, int]]]":
    """(lines, [(rank, open_idx, close_idx)]) for each getPID block."""
    lines = text.splitlines()
    spans: "list[tuple[int, int, int]]" = []
    rank: "int | None" = None
    depth, start = 0, 0
    for i, raw in enumerate(lines):
        if rank is None:
            m = _RANK_OPEN_RE.match(raw)
            if m:
                rank, depth, start = int(m.group(1)), 1, i
            continue
        depth += raw.count("{") - raw.count("}")
        if depth <= 0:
            spans.append((rank, start, i))
            rank = None
    return lines, spans


def _strip_ghost_fix_replay(text: str, owner_rank: int,
                            ghost_tags: "set[int]") -> str:
    lines, spans = _rank_spans(text)
    lo, hi = next((s, e) for r, s, e in spans if r == owner_rank)
    out = []
    for i, ln in enumerate(lines):
        if lo < i < hi:
            m = re.match(r"\s*fix (\d+) ", ln)
            if m and int(m.group(1)) in ghost_tags:
                continue
        out.append(ln)
    return "\n".join(out) + "\n"


def _duplicate_verb_on_other_rank(
    text: str, owner_rank: int, other_rank: int,
    extra_nodes: "dict[int, tuple[float, float, float]]",
) -> str:
    """ADR-78 P0.d surgery: replay the owner's contact block on the
    other rank, ghost-declaring the master nodes it lacks.

    The replayed ``contact`` verb gets an EXPLICIT kn (the positional
    ``auto`` token is replaced) — with ``auto`` the fork's D5.2 rule
    silently SKIPS every facet whose backing solid is off-rank (none of
    the master's solids live on the other rank), so the duplicate would
    contribute nothing and the mutant would converge to the correct
    answer.  (Before the ADR 0092 emit-order fix the duplicate happened
    to fail loudly instead: the pre-fix deck re-constructed the handler
    AFTER the contact verbs, so ``auto`` resolution blew up at
    construction.  Post-fix every handler line executes at the top of
    the deck, before any verb.)  An explicit kn re-creates the true
    P0.d class: both ranks enforce, the answer is plausible and WRONG."""
    def _force_explicit_kn(line: str) -> str:
        toks = line.split()
        if toks[0] == "contact" and "auto" in toks:
            toks[toks.index("auto")] = "1.0e9"
        return " ".join(toks)

    lines, spans = _rank_spans(text)
    lo, hi = next((s, e) for r, s, e in spans if r == owner_rank)
    contact_lines = [
        _force_explicit_kn(lines[i].strip())
        for i in range(lo + 1, hi)
        if lines[i].strip().startswith(("contactSurface", "contact "))
    ]
    assert contact_lines, "owner block carries no contact lines?"
    assert any("1.0e9" in ln for ln in contact_lines), (
        "kn surgery was a no-op — the verb grammar moved?")
    close = next(e for r, _, e in spans if r == other_rank)
    insert = []
    for tag, (x, y, z) in sorted(extra_nodes.items()):
        insert.append(f"    node {tag} {x} {y} {z}")
        insert.append(f"    fix {tag} 1 1 0")
    insert += [f"    {ln}" for ln in contact_lines]
    return "\n".join(lines[:close] + insert + lines[close:]) + "\n"


def test_control_stripped_ghost_fix_replay_detected(
    twin, tmp_path: Path,
) -> None:
    p = twin["probes"]
    deck = tmp_path / "mp_no_fix_replay.tcl"
    mutated = _strip_ghost_fix_replay(
        twin["mp_deck_text"], p["rank_bottom"], set(p["slave"]))
    assert mutated.count("\n") < twin["mp_deck_text"].count("\n"), (
        "mutation was a no-op")
    deck.write_text(mutated + _mp_fragment(p), encoding="utf-8")
    out = _run_mp(deck, twin["env"])
    got = _parse_mp(out)
    s = twin["serial"]
    if got is None:
        return   # died before printing — loud failure, detected.
    detected = (
        got["ok_owner"] != 0 or got["ok_slave"] != 0
        or _rel(got["wtop"], s["wtop"]) > REL_TOL
    )
    assert detected, (
        "stripping the ghost SP replay went UNDETECTED — the twin "
        f"comparison cannot see a wrong deck: control={got} twin={s}"
    )


def test_control_duplicated_contact_verb_detected(
    twin, tmp_path: Path,
) -> None:
    fem, p = twin["fem"], twin["probes"]
    master_xyz = {
        int(nid): (float(x[0]), float(x[1]), float(x[2]))
        for nid, x in zip(fem.nodes.ids, fem.nodes.coords)
        if int(nid) in set(p["master"])
    }
    deck = tmp_path / "mp_dup_verb.tcl"
    mutated = _duplicate_verb_on_other_rank(
        twin["mp_deck_text"], p["rank_bottom"], p["rank_top"], master_xyz)
    deck.write_text(mutated + _mp_fragment(p), encoding="utf-8")
    out, timed_out = _run_mp_may_hang(deck, twin["env"])
    if timed_out:
        # Measured on the current fork (ADR-78 P1/P2 parallel contact):
        # two ranks each enforcing the interaction desynchronizes the
        # handler's collectives and the run DEADLOCKS — it never
        # produces the P0.d plausible-wrong answer. Loud, detected.
        return
    got = _parse_mp(out)
    s = twin["serial"]
    if got is None:
        # A loud death (e.g. a fork-side contact error tearing down the
        # ranks) is also detection — just require it to name contact.
        assert "LadrunoContact" in out or "contact" in out.lower(), (
            f"duplicate-verb control died silently:\n{out[-2000:]}"
        )
        return
    detected = (
        got["ok_owner"] != 0 or got["ok_slave"] != 0
        or _rel(got["wtop"], s["wtop"]) > REL_TOL
    )
    assert detected, (
        "duplicating the contact verb on a second rank went UNDETECTED — "
        "this is exactly the ADR-78 P0.d silent-wrong deck the twin "
        f"comparison must catch: control={got} twin={s}"
    )


# ---------------------------------------------------------------------------
# ADR 0092 S5 open item (fixed) — the auto-chain twins.  A deck that
# declares ``analysis Static`` but relies on the AUTO-emitted handler
# (and, partitioned, numberer/system) must run with an EFFECTIVE
# LadrunoContact handler.  Pre-fix, the auto-emits landed after the
# ``analysis`` line where the engine ignores them: measured, this exact
# serial deck diverged and the 2-rank deck converged plausible-wrong
# (w_top -5.714e-4 vs the true -5.625e-3) with balanced base reactions.
# ---------------------------------------------------------------------------


def _build_ops_auto_chain(fem, *, parallel_chain: bool) -> apeSees:
    """The S5 hazard shape: user-declared ``analysis`` + auto-emitted
    handler.  The serial lane keeps numberer/system explicit (the flat
    path auto-emits only the handler); the partitioned lane omits all
    three so the INV-5 auto-emits engage too."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=2.0e7, nu=0.0)
    ops.element.stdBrick(pg="solid", material=mat)
    ops.fix(pg="base", dofs=(1, 1, 1))
    for pg in ("master", "slave", "top"):
        ops.fix(pg=pg, dofs=(1, 1, 0))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="top", forces=(0.0, 0.0, -2500.0))
    # NO ops.constraints declaration — the handler must be auto-emitted
    # BEFORE the user analysis directive to be effective.
    if not parallel_chain:
        ops.numberer.RCM()
        ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def test_serial_auto_chain_matches_explicit_twin(twin, tmp_path: Path) -> None:
    fem, p = twin["fem"], twin["probes"]
    deck = tmp_path / "serial_auto_chain.tcl"
    _build_ops_auto_chain(fem, parallel_chain=False).tcl(
        str(deck), flat=True)
    text = deck.read_text(encoding="utf-8")
    assert text.index("constraints LadrunoContact") < text.index(
        "analysis Static"), "auto handler landed after the analysis line"
    with open(deck, "a", encoding="utf-8") as f:
        f.write(_serial_fragment(p))
    out = _run_serial(deck, twin["env"])
    got = _parse_serial(out)
    assert got is not None, (
        f"auto-chain serial deck printed nothing:\n{out[-2000:]}")
    assert got["ok"] == 0, (
        f"auto-chain serial deck did not converge (the pre-fix measured "
        f"failure mode): {got}")
    s = twin["serial"]
    bad = {k: _rel(got[k], s[k]) for k in ("wtop", "wslave", "wmaster", "R")
           if _rel(got[k], s[k]) > REL_TOL}
    assert not bad, (
        f"auto-chain serial deck disagrees with the explicit-chain twin "
        f"beyond {REL_TOL:.0e}: {bad}\nauto={got}\ntwin={s}")


def test_mp_auto_chain_matches_explicit_twin(twin, tmp_path: Path) -> None:
    fem, p = twin["fem"], twin["probes"]
    deck = tmp_path / "mp_auto_chain.tcl"
    _build_ops_auto_chain(fem, parallel_chain=True).tcl(str(deck))
    text = deck.read_text(encoding="utf-8")
    analysis_at = text.index("analysis Static")
    for token in ("constraints LadrunoContact",
                  "if {[catch {numberer ", "if {[catch {system "):
        assert text.index(token) < analysis_at, (
            f"auto-emitted {token!r} landed after the analysis line")
    with open(deck, "a", encoding="utf-8") as f:
        f.write(_mp_fragment(p))
    out = _run_mp(deck, twin["env"])
    got = _parse_mp(out)
    assert got is not None, (
        f"auto-chain mp deck printed nothing:\n{out[-2000:]}")
    assert got["ok_owner"] == 0 and got["ok_slave"] == 0, (
        f"auto-chain mp deck did not converge: {got}")
    s = twin["serial"]
    # w_top is the quantity the pre-fix deck got plausibly WRONG
    # (-5.714e-4 vs -5.625e-3) while R still balanced — gate all four.
    bad = {k: _rel(got[k], s[k]) for k in ("wtop", "wslave", "wmaster", "R")
           if _rel(got[k], s[k]) > REL_TOL}
    assert not bad, (
        f"auto-chain 2-rank deck disagrees with the explicit-chain serial "
        f"twin beyond {REL_TOL:.0e}: {bad}\nauto={got}\ntwin={s}")
