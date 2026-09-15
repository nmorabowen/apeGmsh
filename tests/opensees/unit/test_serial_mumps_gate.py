"""ADR 0106 D5 — refuse an explicit ``system Mumps`` on a serial deck.

The Ladruno fork's desktop targets (``OpenSees.exe``, the desktop
openseespy build) never compile the serial ``MumpsSolver``: a declared
``system Mumps`` answers *unknown system type* at runtime, and a rejected
``system`` command does not abort the deck, so the model silently solves
on whatever SOE was already in place instead of stopping.

Scope mirrors the ADR-0074 D4 u-p gate's flat/staged/partitioned seam
exactly — see ``test_ladruno_up_build_gates.py::TestSolverGate`` — but
this gate has no "missing system" branch: only an EXPLICIT ``Mumps``
declaration is refused.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees._internal.build import (
    BridgeError,
    validate_serial_mumps,
)


def _sys(name: str, **kw):
    import apeGmsh.opensees.analysis.system as sysmod

    return getattr(sysmod, name)(**kw)


def _flat(systems, *, enforce=True, partitioned=False):
    validate_serial_mumps(
        enforce=enforce, staged=False, partitioned=partitioned,
        flat_systems=systems, stage_systems=[],
    )


def _staged(stage_systems, *, enforce=True, partitioned=False):
    validate_serial_mumps(
        enforce=enforce, staged=True, partitioned=partitioned,
        flat_systems=[], stage_systems=stage_systems,
    )


class TestSerialMumpsGate:
    # -- flat ----------------------------------------------------------------
    def test_serial_explicit_mumps_raises(self) -> None:
        with pytest.raises(BridgeError, match="unknown system type"):
            _flat([_sys("Mumps")])

    def test_partitioned_mumps_is_accepted(self) -> None:
        """That is what Mumps is for."""
        _flat([_sys("Mumps")], partitioned=True)

    def test_adr0027_auto_emitted_fallback_is_accepted(self) -> None:
        """The ADR 0027 INV-5 auto-emitted 'system Mumps'/'UmfPack' runtime
        fallback (``emitter.parallel_runtime_fallback_system``) never
        creates a ``LinearSystem`` primitive in ``ordered`` -- flat_systems
        stays empty even though the emitted partitioned deck runs
        ``system Mumps`` at runtime."""
        _flat([], partitioned=True)

    def test_general_serial_systems_are_accepted(self) -> None:
        for name in (
            "UmfPack", "Pardiso", "SparseGeneral", "FullGeneral",
            "BandGeneral", "ProfileSPD",
        ):
            _flat([_sys(name)])

    def test_last_declared_system_is_the_effective_one(self) -> None:
        """OpenSees uses the most-recently-declared system at analyze; a
        superseded Mumps before a legal UmfPack must NOT falsely reject,
        and vice versa."""
        _flat([_sys("Mumps"), _sys("UmfPack")])
        with pytest.raises(BridgeError, match="unknown system type"):
            _flat([_sys("UmfPack"), _sys("Mumps")])

    # -- staged ---------------------------------------------------------------
    def test_staged_deck_one_stage_declares_mumps_raises_naming_stage(
        self,
    ) -> None:
        with pytest.raises(BridgeError, match=r"stage 'push'"):
            _staged([
                ("'gravity'", _sys("UmfPack")),
                ("'push'", _sys("Mumps")),
            ])

    def test_staged_deck_no_mumps_passes(self) -> None:
        _staged([
            ("'gravity'", _sys("UmfPack")),
            ("'push'", _sys("Pardiso")),
        ])

    def test_staged_deck_with_no_declared_system_is_not_this_gates_concern(
        self,
    ) -> None:
        """A stage with no system runs on ProfileSPD by wipeAnalysis
        default -- a real footgun, but ``validate_ladruno_up_solver``'s
        missing-system branch, not this gate's."""
        _staged([("'push'", None)])

    def test_staged_partitioned_mumps_is_accepted(self) -> None:
        _staged([("'push'", _sys("Mumps"))], partitioned=True)

    # -- enforce gate (archival / eigen / model-only) --------------------------
    def test_eigen_only_h5_emit_is_skipped(self) -> None:
        """Archival / eigen-only / model-only-skeleton emits never solve,
        so an explicit Mumps is not (yet) a footgun."""
        _flat([_sys("Mumps")], enforce=False)
        _staged([("'push'", _sys("Mumps"))], enforce=False)

    # -- message wording --------------------------------------------------------
    def test_message_wording(self) -> None:
        with pytest.raises(BridgeError) as ei:
            _flat([_sys("Mumps")])
        msg = str(ei.value)
        assert "system Mumps" in msg
        assert "serial deck" in msg
        assert "len(fem.partitions) <= 1" in msg
        assert "MumpsSolver" in msg
        assert "unknown system type" in msg
        assert "OpenSees.exe" in msg
        assert "ProfileSPD" in msg
        assert "ops.system.Pardiso()" in msg
        assert "g.mesh.partitioning" in msg
        assert "OpenSeesMP" in msg
