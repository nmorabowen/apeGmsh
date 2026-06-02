"""
``section Aggregator`` — couple a uniaxial response to each DOF.

The OpenSees Tcl command::

    section Aggregator $tag $matTag1 $code1 $matTag2 $code2 ...
                       <-section $sectionTag>

stitches one :class:`UniaxialMaterial` per force/moment code, optionally
layered on top of a base :class:`Section` (whose other DOFs pass
through).  ``$code`` is one of:

* ``P`` — axial force / axial deformation
* ``Vy`` / ``Vz`` — shear forces along the local y / z axes
* ``T`` — torsion
* ``My`` / ``Mz`` — bending moments about the local y / z axes

For an aggregator built purely from uniaxials (no base section), every
non-zero DOF in the response must have a material assigned;
unassigned DOFs are infinitely stiff (OpenSees defaults to a
zero-displacement constraint at those DOFs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping

from .._internal.tag_resolution import resolve_tag
from .._internal.types import Primitive, Section, UniaxialMaterial

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = ["Aggregator", "AGGREGATOR_DOF_CODES"]


#: The six DOF codes accepted by ``section Aggregator``.  Anything
#: outside this set fails validation in :class:`Aggregator`'s
#: ``__post_init__``.
AGGREGATOR_DOF_CODES: tuple[str, ...] = ("P", "Vy", "Vz", "T", "My", "Mz")

_AggregatorDOF = Literal["P", "Vy", "Vz", "T", "My", "Mz"]


@dataclass(frozen=True, kw_only=True, slots=True)
class Aggregator(Section):
    """``section Aggregator`` — DOF-wise uniaxial coupling.

    Each DOF in ``materials_by_dof`` is coupled to its associated
    :class:`UniaxialMaterial`; when ``base_section`` is non-``None``,
    the aggregator layers these on top of the base section's response
    for the listed DOFs and falls back to the base section's response
    on every other DOF.

    Parameters
    ----------
    materials_by_dof
        Mapping from one of :data:`AGGREGATOR_DOF_CODES` to a typed
        :class:`UniaxialMaterial` primitive.  Insertion order is
        preserved in the emitted command (Python's dict order
        guarantee).  At least one entry is required.
    base_section
        Optional :class:`Section` providing the response on DOFs not
        covered by ``materials_by_dof``.  When ``None`` the aggregator
        is "pure" — unlisted DOFs are infinitely stiff.

    Notes
    -----
    The aggregator emits with the materials in the dict's insertion
    order — this is the order the user typed them.  Mixing keys
    (e.g. ``{"Mz": k_phi, "P": k_N}``) is legal and matches OpenSees
    semantics (the parser is order-independent on the code/tag
    pairs).

    Use case anchor — the S3-splice macro-model from Cerro Lindo
    Pasada 3a binding C.4 couples ``Mz`` (rotational spring) with
    ``P`` (axial slip-bearing) and ``Vy`` (frictional shear) in a
    single ``zeroLengthSection`` element via this primitive.
    """

    materials_by_dof: Mapping[_AggregatorDOF, UniaxialMaterial]
    base_section: Section | None = None

    def __post_init__(self) -> None:
        if not self.materials_by_dof:
            raise ValueError(
                "Aggregator: materials_by_dof must not be empty — at "
                "least one DOF/material pair is required."
            )
        for code, mat in self.materials_by_dof.items():
            if code not in AGGREGATOR_DOF_CODES:
                raise ValueError(
                    f"Aggregator: unknown DOF code {code!r}. Allowed: "
                    f"{AGGREGATOR_DOF_CODES}."
                )
            if not isinstance(mat, UniaxialMaterial):
                raise TypeError(
                    f"Aggregator: materials_by_dof[{code!r}] must be a "
                    f"UniaxialMaterial primitive, got "
                    f"{type(mat).__name__!r}."
                )
            if mat.is_rate_dependent:
                # section Aggregator feeds its uniaxials the 1-arg
                # setTrialStrain (no rate), and ZeroLengthSection never
                # passes a section strain rate — a rate-dependent
                # material here is SILENTLY inert as a dashpot. Fail loud.
                raise ValueError(
                    f"Aggregator: materials_by_dof[{code!r}] is a "
                    f"rate-dependent material ({type(mat).__name__!r}), "
                    f"which is silently inert inside a section "
                    f"Aggregator — neither the aggregator nor "
                    f"zeroLengthSection passes a strain rate, so it "
                    f"produces zero velocity-proportional force. Put "
                    f"the dashpot directly on a zeroLength/twoNodeLink "
                    f"-mat slot, or use a rate-independent material here."
                )
        if self.base_section is not None and not isinstance(
            self.base_section, Section,
        ):
            raise TypeError(
                "Aggregator: base_section must be a Section primitive "
                f"or None, got {type(self.base_section).__name__!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        deps: list[Primitive] = list(self.materials_by_dof.values())
        if self.base_section is not None:
            deps.append(self.base_section)
        return tuple(deps)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        params: list[int | float | str] = []
        for code, mat in self.materials_by_dof.items():
            params.append(resolve_tag(emitter, mat))
            params.append(code)
        if self.base_section is not None:
            base_tag = resolve_tag(emitter, self.base_section)
            params += ["-section", base_tag]
        emitter.section("Aggregator", tag, *params)
