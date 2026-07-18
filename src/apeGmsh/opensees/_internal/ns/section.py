"""
``_SectionNS`` — backs ``ops.section.<Type>(...)``.

Phase 1C populates this with one typed method per OpenSees section.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, cast

from ...section.aggregator import Aggregator
from ...section.beam import ElasticSection
from ...section.computed import ComputedSection as _ComputedSectionCls
from ...section.fiber import (
    Fiber as _FiberCls,
    FiberPoint,
    RectPatch,
    StraightLayer,
    W_fiber as _build_W_fiber,
)
from ...section.plate import (
    ElasticMembranePlateSection,
    LayeredShell,
    LayeredShellFiberSection,
    ShellLayer,
)
from ..types import Section, UniaxialMaterial
from ._base import _BridgeNamespace

if TYPE_CHECKING:
    from apeGmsh.sections._analysis import SectionProperties


__all__ = ["_SectionNS"]


class _SectionNS(_BridgeNamespace):
    """``ops.section.<Type>(...)`` — Phase 1C population.

    Each method constructs a typed section primitive, registers it
    with the bridge (allocating its tag), and returns the typed
    instance. Per the namespace contract in :mod:`api-design`,
    every signature is fully kw-only with explicit types — no
    ``**kwargs`` (P12).
    """

    # -- Beam-line sections ---------------------------------------------

    def Elastic(
        self,
        *,
        E: float,
        A: float,
        Iz: float,
        Iy: float | None = None,
        G: float | None = None,
        J: float | None = None,
        alphaY: float | None = None,
        alphaZ: float | None = None,
        name: str | None = None,
    ) -> ElasticSection:
        """``section Elastic`` — 2-D or 3-D linear-elastic beam section.

        Supplying any of ``Iy`` / ``J`` / ``alphaZ`` selects the 3-D
        variant; in that case ``Iy``, ``G``, and ``J`` are all
        required. See :class:`ElasticSection` for the full contract.
        """
        return self._bridge._register(
            ElasticSection(
                E=E, A=A, Iz=Iz,
                Iy=Iy, G=G, J=J,
                alphaY=alphaY, alphaZ=alphaZ,
            ),
            name=name,
        )

    def ComputedSection(
        self,
        *,
        analysis: "SectionProperties",
        E: float | None = None,
        G: float | None = None,
        ndm: Literal[2, 3] = 3,
        name: str | None = None,
    ) -> _ComputedSectionCls:
        """``section Elastic`` lowered lazily from a section analyzer
        (ADR 0078).

        ``analysis`` is a :class:`~apeGmsh.sections.SectionProperties`
        declaration; the axis mapping and reference-moduli rules run at
        emit through the shared lowering.  ``E`` / ``G`` default from
        the single material on a homogeneous analyzer and are required
        (fail-loud at emit) for composite and geometric-only analyzers.
        ``ndm`` selects the 2-D or 3-D ``ElasticSection`` form. See
        :class:`~apeGmsh.opensees.section.ComputedSection`.
        """
        return self._bridge._register(
            _ComputedSectionCls(analysis=analysis, E=E, G=G, ndm=ndm),
            name=name,
        )

    # -- Plate / shell sections -----------------------------------------

    def ElasticMembranePlateSection(
        self,
        *,
        E: float,
        nu: float,
        h: float,
        rho: float = 0.0,
        name: str | None = None,
    ) -> ElasticMembranePlateSection:
        """``section ElasticMembranePlateSection`` — single-layer plate."""
        return self._bridge._register(
            ElasticMembranePlateSection(E=E, nu=nu, h=h, rho=rho), name=name
        )

    def LayeredShell(
        self,
        *,
        layers: tuple[ShellLayer, ...],
        name: str | None = None,
    ) -> LayeredShell:
        """``section LayeredShell`` — stacked nDMaterial layers."""
        return self._bridge._register(LayeredShell(layers=layers), name=name)

    def LayeredShellFiberSection(
        self,
        *,
        layers: tuple[ShellLayer, ...],
        name: str | None = None,
    ) -> LayeredShellFiberSection:
        """``section LayeredShellFiberSection`` — fiber-based stacked
        layered plate section."""
        return self._bridge._register(
            LayeredShellFiberSection(layers=layers), name=name
        )

    # -- Fiber section ---------------------------------------------------

    def Fiber(
        self,
        *,
        patches: tuple[RectPatch, ...] = (),
        fibers:  tuple[FiberPoint, ...] = (),
        layers:  tuple[StraightLayer, ...] = (),
        GJ: float | None = None,
        name: str | None = None,
    ) -> _FiberCls:
        """``section Fiber`` — block-emit fiber section.

        At least one of ``patches`` / ``fibers`` / ``layers`` must be
        non-empty. See :class:`Fiber` for the full contract and the
        material-tag resolution open question.

        Materials embedded inside the ``patches`` / ``fibers`` /
        ``layers`` structs are passed as object handles (those structs
        are built outside the bridge); the name-alias channel applies
        to the direct reference kwargs only.
        """
        return self._bridge._register(
            _FiberCls(
                patches=patches,
                fibers=fibers,
                layers=layers,
                GJ=GJ,
            ),
            name=name,
        )

    # -- Parametric fiber-section builders ------------------------------

    def W_fiber(
        self,
        *,
        bf: float,
        tf: float,
        hw: float,
        tw: float,
        material: UniaxialMaterial | str,
        ny_flange: int = 2,
        nz_flange: int = 8,
        ny_web: int = 8,
        nz_web: int = 1,
        GJ: float | None = None,
        name: str | None = None,
    ) -> _FiberCls:
        """``section Fiber`` for a built-up W shape — parametric builder.

        Convenience wrapper around
        :func:`apeGmsh.opensees.section.fiber.W_fiber` that
        auto-registers the resulting :class:`Fiber` with the bridge.
        ``material`` accepts a UniaxialMaterial handle or its registered
        name. See the helper docstring for the full geometric contract.
        """
        material = self._bridge._resolve(material, base=UniaxialMaterial)
        return self._bridge._register(
            _build_W_fiber(
                bf=bf, tf=tf, hw=hw, tw=tw,
                material=material,
                ny_flange=ny_flange, nz_flange=nz_flange,
                ny_web=ny_web, nz_web=nz_web,
                GJ=GJ,
            ),
            name=name,
        )

    # -- Aggregator (composes other sections + uniaxials) ---------------

    def Aggregator(
        self,
        *,
        materials_by_dof: Mapping[
            Literal["P", "Vy", "Vz", "T", "My", "Mz"],
            UniaxialMaterial | str,
        ],
        base_section: Section | str | None = None,
        name: str | None = None,
    ) -> Aggregator:
        """``section Aggregator`` — DOF-wise uniaxial coupling.

        Maps each entry in ``materials_by_dof`` (DOF code →
        :class:`UniaxialMaterial`) onto a force/moment response code,
        optionally layered on top of ``base_section``.  Both the
        per-DOF materials and ``base_section`` accept object handles or
        registered names.  See
        :class:`apeGmsh.opensees.section.Aggregator` for the full
        contract.
        """
        resolved_mats: dict[
            Literal["P", "Vy", "Vz", "T", "My", "Mz"], UniaxialMaterial
        ] = {
            dof: cast(
                UniaxialMaterial,
                self._bridge._resolve(mat, base=UniaxialMaterial),
            )
            for dof, mat in materials_by_dof.items()
        }
        resolved_base = (
            self._bridge._resolve(base_section, base=Section)
            if base_section is not None else None
        )
        return self._bridge._register(
            Aggregator(
                materials_by_dof=resolved_mats,
                base_section=resolved_base,
            ),
            name=name,
        )
