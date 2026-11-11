"""An MPCO gauss strain record that names no element class refuses (#2083).

Tri31 has no element-level ``strains`` branch; it answers strain only as
``material.strain``. ``collect_mpco_tokens`` picks that token from the
record's ``element_class_name``. Before #2083 a strain record that named no
class emitted ``-E strains``, and ``Results.from_mpco`` then returned no
Tri31 rows and no warning.

A :class:`ResolvedRecorderRecord` carries element ids, not a physical group
or a bridge, so the emit has nothing to resolve the class from: it refuses
by name, pointing at ``element_class_name=``. Every route into the MPCO
emit takes such records (``emit_mpco_tcl`` / ``emit_mpco_python`` /
``mpco_ops_args``, ``spec.emit_mpco``, ``spec.emit_default``), and the
refusal comes before any ``ops.recorder`` call.

The cases in ``TestScopeUnchanged`` pin what the refusal leaves alone. They
pass with or without the fix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results.spec._emit import (
    collect_mpco_tokens,
    emit_mpco_python,
    emit_mpco_tcl,
    mpco_ops_args,
)
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


def _gauss(components, cls=None, name="g") -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="gauss", name=name, components=tuple(components),
        dt=None, n_steps=None, element_ids=np.array([1, 2]),
        element_class_name=cls,
    )


class _FakeOps:
    """Records every ``recorder`` call; never a real domain."""

    def __init__(self) -> None:
        self.recorder_calls: list[tuple] = []

    def recorder(self, *args) -> int:
        self.recorder_calls.append(tuple(args))
        return 1

    def remove(self, *args) -> None:  # pragma: no cover - not reached
        pass


# The refusal names the record, the component, the missing keyword and the
# class whose rows would read back empty.
_REFUSAL = r"(?s)'tri'.*'strain_xx'.*element_class_name.*Tri31"


class TestRefusesUnknownClass:
    def test_continuum_strain_with_no_class_refuses(self) -> None:
        with pytest.raises(ValueError, match=_REFUSAL):
            collect_mpco_tokens([_gauss(("strain_xx", "strain_yy"),
                                        name="tri")])

    @pytest.mark.parametrize("emit", [emit_mpco_tcl, emit_mpco_python,
                                      mpco_ops_args])
    def test_every_emit_form_refuses(self, emit) -> None:
        with pytest.raises(ValueError, match=_REFUSAL):
            emit([_gauss(("strain_xx",), name="tri")])

    def test_derived_strain_scalar_with_no_class_refuses(self) -> None:
        # A derived scalar records the raw strain tensor, so the token is
        # class-dependent too.
        with pytest.raises(ValueError, match=r"'von_mises_strain'.*"
                                             r"element_class_name"):
            collect_mpco_tokens([_gauss(("von_mises_strain",))])

    def test_only_the_classless_record_is_named(self) -> None:
        # Several records: one names its class, one does not. The refusal
        # names the one that does not.
        records = [_gauss(("strain_xx",), cls="Tri31", name="named"),
                   _gauss(("strain_xx",), name="tri")]
        with pytest.raises(ValueError, match=_REFUSAL) as info:
            collect_mpco_tokens(records)
        assert "'named'" not in str(info.value)

    def test_refused_before_the_recorder_is_made(
        self, tmp_path: Path,
    ) -> None:
        ops = _FakeOps()
        spec = ResolvedRecorderSpec(
            fem_snapshot_id="s", records=(_gauss(("strain_xx",),
                                                 name="tri"),))
        with pytest.raises(ValueError, match=_REFUSAL):
            with spec.emit_mpco(tmp_path / "run.mpco", ops=ops):
                pass  # pragma: no cover - never entered
        assert ops.recorder_calls == []
        assert not (tmp_path / "run.mpco").exists()


class TestScopeUnchanged:
    """What the refusal leaves alone (passes before and after the fix)."""

    def test_tri31_named_records_material_strain(self) -> None:
        _, elem = collect_mpco_tokens([_gauss(("strain_xx",), cls="Tri31")])
        assert elem == ["material.strain"]

    def test_other_class_named_records_strains(self) -> None:
        _, elem = collect_mpco_tokens([_gauss(("strain_xx",),
                                              cls="FourNodeQuad")])
        assert elem == ["strains"]

    def test_stress_with_no_class_records_stresses(self) -> None:
        _, elem = collect_mpco_tokens([_gauss(("stress_xx",))])
        assert elem == ["stresses"]

    def test_shell_generalized_strain_with_no_class_records_strains(
        self,
    ) -> None:
        # Tri31 has no shell resultants, so a shell generalized strain's
        # token does not depend on the class.
        _, elem = collect_mpco_tokens([_gauss(("membrane_strain_xx",
                                               "curvature_xx"))])
        assert elem == ["strains"]
