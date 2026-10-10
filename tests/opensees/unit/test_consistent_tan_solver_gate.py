"""The ``consistent_tan`` contact / symmetric-solver gate (#1273, B4-d).

``contact(..., consistent_tan=True)`` emits ``-consistanttan`` and
``edge_consistent_tan=True`` emits ``-edgeConsistentTan``: the fork's
NON-symmetric consistent friction tangent. A half-storage solver reads only
the ``col >= row`` half of each element matrix and returns a plausible but
wrong solve with rc 0, so :func:`validate_consistent_tan_solver` refuses
with a :class:`BridgeError` naming the allowed solvers.

Scope mirrors the ADR-0074 D4 u-p gate exactly — see
``test_ladruno_up_build_gates.py::TestSolverGate`` — including the
"missing ``system`` is a violation" branch (#1273's recommendation).

``TestThroughTheBuild`` is the #1273 repro reached the way a user does:
a real two-body Gmsh mesh, ``g.constraints.contact(..., consistent_tan=True)``,
``ops.system.ProfileSPD()`` and ``ops.tcl(...)``. On ``main`` before this
slice that deck builds silently; a gate wired to nothing fails here even
with every unit case green.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace as NS

import pytest

from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees._internal.contact_solver_gate import (
    validate_consistent_tan_solver,
)
from apeGmsh.opensees.analysis.system import _MATRIX_TYPES

CONFIGURABLE = ["Pardiso", "Mumps"]
NOT_UNSYMMETRIC = sorted(t for t in _MATRIX_TYPES if t != "unsymmetric")

SYMMETRIC = [
    "ProfileSPD", "SProfileSPD", "ParallelProfileSPD", "BandSPD",
    "SparseSYM", "Diagonal", "MPIDiagonal",
]
GENERAL = [
    "UmfPack", "SparseGeneral", "FullGeneral", "BandGeneral", "Mumps",
    "Pardiso",
]
ALLOWED_IN_MESSAGE = (
    "UmfPack", "Pardiso", "SparseGeneral", "FullGeneral", "BandGeneral",
    "Mumps",
)


def _sys(name: str, **kw):
    import apeGmsh.opensees.analysis.system as sysmod

    return getattr(sysmod, name)(**kw)


def _rec(name="joint", **over):
    base = dict(name=name, consistent_tan=False, edge_consistent_tan=False)
    base.update(over)
    return NS(**base)


def _fem(*recs):
    return NS(elements=NS(contacts=list(recs), contact_planes=[]))


CT = _fem(_rec(consistent_tan=True))
EDGE = _fem(_rec(edge_consistent_tan=True))
PLAIN = _fem(_rec())


def _flat(fem, systems, *, enforce=True, partitioned=False):
    validate_consistent_tan_solver(
        fem, enforce=enforce, staged=False, partitioned=partitioned,
        flat_systems=systems, stage_systems=[],
    )


def _staged(fem, stage_systems, *, enforce=True, partitioned=False):
    validate_consistent_tan_solver(
        fem, enforce=enforce, staged=True, partitioned=partitioned,
        flat_systems=[], stage_systems=stage_systems,
    )


class TestFlat:
    @pytest.mark.parametrize("sys_name", SYMMETRIC)
    @pytest.mark.parametrize("fem", [CT, EDGE], ids=["consistent_tan", "edge"])
    def test_symmetric_or_diagonal_system_raises(self, fem, sys_name) -> None:
        with pytest.raises(BridgeError, match="UNSYMMETRIC") as exc:
            _flat(fem, [_sys(sys_name)])
        msg = str(exc.value)
        assert "joint" in msg and sys_name in msg
        for allowed in ALLOWED_IN_MESSAGE:
            assert allowed in msg          # the error names the allowed solvers

    @pytest.mark.parametrize("sys_name", GENERAL)
    @pytest.mark.parametrize("fem", [CT, EDGE], ids=["consistent_tan", "edge"])
    def test_general_system_passes(self, fem, sys_name) -> None:
        _flat(fem, [_sys(sys_name)])

    @pytest.mark.parametrize("cls_name", CONFIGURABLE)
    @pytest.mark.parametrize("mtype", sorted(_MATRIX_TYPES))
    def test_configured_symmetry_decides(self, cls_name, mtype) -> None:
        """Pardiso / Mumps are judged by their configured matrix_type."""
        system = _sys(cls_name, matrix_type=mtype)
        if mtype == "unsymmetric":
            _flat(CT, [system])
            return
        with pytest.raises(BridgeError, match=f"matrix_type={mtype!r}"):
            _flat(CT, [system])

    @pytest.mark.parametrize("cls_name", CONFIGURABLE)
    def test_unreadable_matrix_type_fails_closed(self, cls_name) -> None:
        """A Pardiso/Mumps with no readable matrix_type is refused, never
        assumed unsymmetric."""
        system = type(cls_name, (), {})()           # the class name, nothing else
        assert not hasattr(system, "matrix_type")
        with pytest.raises(BridgeError, match="matrix_type=<unreadable>"):
            _flat(CT, [system])

    def test_no_system_declared_raises(self) -> None:
        """The OpenSees no-``system`` default is ProfileSPD (#1273 ruling),
        and the refusal spells out the unsymmetric options."""
        with pytest.raises(BridgeError, match="ProfileSPD.*undeclared") as exc:
            _flat(CT, [])
        msg = str(exc.value)
        assert 'ops.system.Pardiso(matrix_type="unsymmetric")' in msg
        assert 'ops.system.Mumps(matrix_type="unsymmetric")' in msg
        assert "ops.system.UmfPack()" in msg

    def test_last_declared_system_is_the_effective_one(self) -> None:
        _flat(CT, [_sys("ProfileSPD"), _sys("UmfPack")])
        with pytest.raises(BridgeError):
            _flat(CT, [_sys("UmfPack"), _sys("ProfileSPD")])

    def test_partitioned_no_system_is_allowed(self) -> None:
        """ADR 0027 INV-5 auto-emits a general Mumps/UmfPack fallback."""
        _flat(CT, [], partitioned=True)

    def test_missing_system_skipped_when_not_enforced(self) -> None:
        _flat(CT, [], enforce=False)

    def test_declared_symmetric_raises_even_when_not_enforced(self) -> None:
        """A model-only Tcl export still catches ops.system.ProfileSPD()."""
        with pytest.raises(BridgeError):
            _flat(CT, [_sys("ProfileSPD")], enforce=False)

    def test_both_flags_named(self) -> None:
        fem = _fem(_rec(consistent_tan=True, edge_consistent_tan=True))
        with pytest.raises(BridgeError, match="consistent_tan, edge_consistent_tan"):
            _flat(fem, [_sys("ProfileSPD")])


class TestStaged:
    def test_symmetric_stage_raises_naming_the_stage(self) -> None:
        with pytest.raises(BridgeError, match="stage 'push'"):
            _staged(CT, [("'hold'", _sys("UmfPack")), ("'push'", _sys("BandSPD"))])

    def test_undeclared_stage_raises_when_enforced(self) -> None:
        with pytest.raises(BridgeError, match="stage 'push', undeclared"):
            _staged(CT, [("'hold'", _sys("UmfPack")), ("'push'", None)])

    def test_undeclared_stage_skipped_when_not_enforced(self) -> None:
        _staged(CT, [("'push'", None)], enforce=False)

    def test_every_stage_general_passes(self) -> None:
        _staged(CT, [("'a'", _sys("UmfPack")), ("'b'", _sys("Pardiso"))])


class TestUnaffected:
    """A deck with contacts but none using the flag is untouched (seam)."""

    @pytest.mark.parametrize("sys_name", SYMMETRIC)
    def test_plain_contact_passes_on_any_system(self, sys_name) -> None:
        _flat(PLAIN, [_sys(sys_name)])

    def test_plain_contact_no_system_passes(self) -> None:
        _flat(PLAIN, [])
        _staged(PLAIN, [("'push'", None)])

    def test_no_contacts_at_all(self) -> None:
        _flat(_fem(), [])
        _flat(NS(), [])                       # no ``elements`` either

    def test_contact_planes_are_not_contacts(self) -> None:
        fem = NS(elements=NS(contacts=[], contact_planes=[_rec(consistent_tan=True)]))
        _flat(fem, [_sys("ProfileSPD")])


# ---------------------------------------------------------------------------
# The #1273 repro through ``ops.tcl``: builds silently on main.
# ---------------------------------------------------------------------------
def _two_body_contact_fem(partitions=0, **contact_kw):
    gmsh = pytest.importorskip("gmsh")
    from apeGmsh import apeGmsh

    def face_at_z(volume_tag, z, tol=1e-3):
        for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
            if dim == 2:
                com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
                if abs(com[2] - z) < tol:
                    return abs(tag)
        raise AssertionError(f"no face of vol {volume_tag} at z={z}")

    with warnings.catch_warnings(), \
            apeGmsh(model_name="ct_gate", verbose=False) as g:
        if partitions:
            # The session's exit autosave skips a partitioned model and
            # says so (a UserWarning, not this test's subject); scoped to
            # this session so -W error stays meaningful elsewhere.
            warnings.filterwarnings(
                "ignore", message=r".*partitioned run", category=UserWarning)
        box1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        box2 = g.model.geometry.add_box(0, 0, 1.05, 1, 1, 1)
        g.model.sync()
        master = face_at_z(box1, 1.0)
        slave = face_at_z(box2, 1.05)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box1, box2], name="solid")
        g.physical.add(2, [master], name="master")
        g.physical.add(2, [slave], name="slave")
        g.constraints.contact("master", "slave", **contact_kw)
        if partitions:
            g.mesh.partitioning.partition(partitions)
        return g.mesh.queries.get_fem_data(dim=3)


def _bridge(fem):
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem, _artifacts=False)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.FourNodeTetrahedron(pg="solid", material=mat)
    ops.analysis.Static()
    return ops


def _bridge_no_chain(fem):
    """Model + elements only: the stages own their analysis chains."""
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem, _artifacts=False)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.FourNodeTetrahedron(pg="solid", material=mat)
    return ops


def _chain(ops, system):
    """A stage chain. ``system=None`` is how a stage reaches the gate's
    "undeclared" branch: ``_StageBuilder.analysis`` annotates ``system``
    as required but only stores the reference, and ``StageRecord.system``
    is Optional, so the record the gate reads carries ``None``."""
    return dict(
        test=ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        algorithm=ops.algorithm.Newton(),
        integrator=ops.integrator.LoadControl(dlam=0.1),
        constraints=ops.constraints.LadrunoContact(),   # contacts need it
        numberer=ops.numberer.RCM(),
        system=system,
        analysis=ops.analysis.Static(),
    )


_MORTAR_CT = dict(formulation="mortar", eps_n="auto", mu=0.3,
                  consistent_tan=True)
_MORTAR_EDGE_CT = dict(formulation="mortar", eps_n="auto", edge_edge=True,
                       edge_mu=0.3, edge_consistent_tan=True)


class TestThroughTheBuild:
    def test_profilespd_refuses(self, tmp_path) -> None:
        """The #1273 repro. Built silently on main before this slice."""
        fem = _two_body_contact_fem(**_MORTAR_CT)
        assert fem.elements.contacts[0].consistent_tan
        ops = _bridge(fem)
        ops.system.ProfileSPD()
        with pytest.raises(BridgeError, match="consistent_tan.*ProfileSPD"):
            ops.tcl(str(tmp_path / "deck.tcl"))
        assert not (tmp_path / "deck.tcl").exists()   # raised before any write

    def test_edge_consistent_tan_refuses_the_same(self, tmp_path) -> None:
        fem = _two_body_contact_fem(**_MORTAR_EDGE_CT)
        assert fem.elements.contacts[0].edge_consistent_tan
        ops = _bridge(fem)
        ops.system.ProfileSPD()
        with pytest.raises(BridgeError, match="edge_consistent_tan.*ProfileSPD"):
            ops.tcl(str(tmp_path / "deck.tcl"))

    def test_no_system_refuses(self, tmp_path) -> None:
        ops = _bridge(_two_body_contact_fem(**_MORTAR_CT))
        with pytest.raises(BridgeError, match="undeclared"):
            ops.tcl(str(tmp_path / "deck.tcl"))

    def test_symmetric_pardiso_refuses(self, tmp_path) -> None:
        ops = _bridge(_two_body_contact_fem(**_MORTAR_CT))
        ops.system.Pardiso(matrix_type="symmetric")
        with pytest.raises(BridgeError, match="matrix_type='symmetric'"):
            ops.tcl(str(tmp_path / "deck.tcl"))

    # -- the configured-symmetry rider (#1273 ruling) ----------------------
    # Pardiso runs on the serial deck. Mumps needs a PARTITIONED mesh: on a
    # serial deck the ADR 0106 D5 gate downstream of this one refuses every
    # Mumps, which would mask the verdict under test.

    @pytest.mark.parametrize("mtype", sorted(_MATRIX_TYPES))
    def test_pardiso_judged_by_matrix_type(self, tmp_path, mtype) -> None:
        ops = _bridge(_two_body_contact_fem(**_MORTAR_CT))
        ops.system.Pardiso(matrix_type=mtype)
        out = tmp_path / "deck.tcl"
        if mtype == "unsymmetric":
            ops.tcl(str(out))
            assert "-consistanttan" in out.read_text()
            return
        with pytest.raises(BridgeError, match=f"Pardiso.*matrix_type={mtype!r}"):
            ops.tcl(str(out))

    @pytest.mark.parametrize("mtype", sorted(_MATRIX_TYPES))
    def test_mumps_judged_by_matrix_type(self, tmp_path, mtype) -> None:
        fem = _two_body_contact_fem(partitions=2, **_MORTAR_CT)
        assert len(fem.partitions) == 2
        ops = _bridge(fem)
        ops.numberer.ParallelRCM()      # else the ADR 0027 auto-emit warns
        ops.system.Mumps(matrix_type=mtype)
        out = tmp_path / "deck.tcl"
        if mtype == "unsymmetric":
            ops.tcl(str(out))
            assert "-consistanttan" in out.read_text()
            return
        with pytest.raises(BridgeError, match=f"Mumps.*matrix_type={mtype!r}"):
            ops.tcl(str(out))

    def test_staged_unsymmetric_mumps_stage_passes(self, tmp_path) -> None:
        fem = _two_body_contact_fem(partitions=2, **_MORTAR_CT)
        ops = _bridge_no_chain(fem)
        with ops.stage(name="hold") as s:
            s.analysis(**_chain(ops, ops.system.Mumps(matrix_type="unsymmetric")))
            s.run(n_increments=1)
        with ops.stage(name="push") as s:
            s.analysis(**_chain(ops, ops.system.Mumps()))      # the default
            s.run(n_increments=1)
        out = tmp_path / "deck.tcl"
        ops.tcl(str(out), flat=True)
        assert "-consistanttan" in out.read_text()

    @pytest.mark.parametrize("mtype", NOT_UNSYMMETRIC)
    def test_staged_symmetric_mumps_stage_refuses(self, tmp_path, mtype) -> None:
        fem = _two_body_contact_fem(partitions=2, **_MORTAR_CT)
        ops = _bridge_no_chain(fem)
        with ops.stage(name="hold") as s:
            s.analysis(**_chain(ops, ops.system.Mumps()))
            s.run(n_increments=1)
        with ops.stage(name="push") as s:
            s.analysis(**_chain(ops, ops.system.Mumps(matrix_type=mtype)))
            s.run(n_increments=1)
        with pytest.raises(BridgeError,
                           match=f"Mumps.*matrix_type={mtype!r}.*stage 'push'"):
            ops.tcl(str(tmp_path / "deck.tcl"), flat=True)

    @pytest.mark.parametrize(
        "make",
        [lambda s: s.UmfPack(), lambda s: s.FullGeneral(),
         lambda s: s.BandGeneral(), lambda s: s.Pardiso()],
        ids=["UmfPack", "FullGeneral", "BandGeneral", "Pardiso"],
    )
    def test_general_builds(self, tmp_path, make) -> None:
        ops = _bridge(_two_body_contact_fem(**_MORTAR_CT))
        make(ops.system)
        out = tmp_path / "deck.tcl"
        ops.tcl(str(out))
        assert "-consistanttan" in out.read_text()

    def test_staged_profilespd_refuses_naming_the_stage(self, tmp_path) -> None:
        ops = _bridge_no_chain(_two_body_contact_fem(**_MORTAR_CT))
        with ops.stage(name="hold") as s:
            s.analysis(**_chain(ops, ops.system.UmfPack()))
            s.run(n_increments=1)
        with ops.stage(name="push") as s:
            s.analysis(**_chain(ops, ops.system.ProfileSPD()))
            s.run(n_increments=1)
        with pytest.raises(BridgeError, match="stage 'push'.*ProfileSPD"):
            ops.tcl(str(tmp_path / "deck.tcl"))

    def test_staged_no_system_refuses_as_undeclared(self, tmp_path) -> None:
        ops = _bridge_no_chain(_two_body_contact_fem(**_MORTAR_CT))
        with ops.stage(name="hold") as s:
            s.analysis(**_chain(ops, ops.system.UmfPack()))
            s.run(n_increments=1)
        with ops.stage(name="push") as s:
            s.analysis(**_chain(ops, None))
            s.run(n_increments=1)
        with pytest.raises(BridgeError, match="stage 'push', undeclared"):
            ops.tcl(str(tmp_path / "deck.tcl"))

    def test_archival_h5_stamps_the_refusal_and_replay_fails_closed(
        self, tmp_path,
    ) -> None:
        """The archival emit never raises: it records the verdict (ADR 0114
        D6), and the replay fails closed on it. A direct
        ``validate_consistent_tan_solver`` call in place of the
        ``_solve_gate`` wrapper would stamp nothing and replay silently."""
        import h5py

        from apeGmsh.opensees import OpenSeesModel

        import warnings

        from apeGmsh.opensees.emitter.h5 import H5LedgerWarning

        ops = _bridge(_two_body_contact_fem(**_MORTAR_CT))   # no system
        out = tmp_path / "m.h5"
        with warnings.catch_warnings():
            # The contact verbs are ledgered, not archived (ADR 0114 Q3):
            # expected on every contact archive, and not this test's subject.
            warnings.simplefilter("ignore", H5LedgerWarning)
            ops.h5(str(out))                                 # no BridgeError
        with h5py.File(out, "r") as f:
            stored = [str(t) for t in f["opensees"].attrs["solve_refusals"]]
        assert "consistent_tan_solver" in stored
        model = OpenSeesModel.from_h5(out)
        assert "consistent_tan_solver" in model.solve_stamp.solve_refusals
        with pytest.raises(BridgeError, match="'consistent_tan_solver'.*fails closed"):
            model.build("tcl")

    def test_plain_contact_on_profilespd_is_unaffected(self, tmp_path) -> None:
        """Seam: no flag, no refusal and no new warning (run with -W error)."""
        import warnings

        ops = _bridge(_two_body_contact_fem(formulation="mortar",
                                            eps_n="auto", mu=0.3))
        ops.system.ProfileSPD()
        out = tmp_path / "deck.tcl"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ops.tcl(str(out))
        assert "-consistanttan" not in out.read_text()
