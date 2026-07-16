"""``SectionMaterial`` — the per-region material spec for the
section-properties analyzer (ADR 0078).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class SectionMaterial:
    """Material assigned to one physical-group region of a cross-section.

    Parameters
    ----------
    E
        Young's modulus (> 0).  Weights the geometric integrals.
    nu
        Poisson's ratio (−1 < nu < 0.5).
    G
        Shear-modulus **override**; default is the isotropic
        ``E / (2 (1 + nu))``.  An independent ``G`` exists for
        *equivalent shear media* — smeared battens / lacing, corrugated
        webs: a strip with near-zero ``E`` and a calibrated ``G``
        transfers shear between parts without adding parasitic flexural
        area.  The solver assembles the E-field (geometric) and G-field
        (warping) separately, so the override is exact, not a fudge.
    fy
        Yield stress (> 0).  Required by ``plastic()``.
    density
        Mass density; when every material carries one, the analyzer
        reports mass per unit length.
    name
        Display-only label (tables, plots).  Falls back to the physical
        group name where one is needed.
    """

    E: float
    nu: float
    G: float | None = None
    fy: float | None = None
    density: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.E > 0.0:
            raise ValueError(f"SectionMaterial: E must be > 0, got {self.E}.")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(
                f"SectionMaterial: nu must lie in (-1, 0.5), got {self.nu}."
            )
        if self.G is not None and not self.G > 0.0:
            raise ValueError(f"SectionMaterial: G must be > 0, got {self.G}.")
        if self.fy is not None and not self.fy > 0.0:
            raise ValueError(f"SectionMaterial: fy must be > 0, got {self.fy}.")
        if self.density is not None and not self.density > 0.0:
            raise ValueError(
                f"SectionMaterial: density must be > 0, got {self.density}."
            )

    @property
    def shear_modulus(self) -> float:
        """Effective shear modulus: the ``G`` override when given, else
        the isotropic ``E / (2 (1 + nu))``."""
        if self.G is not None:
            return self.G
        return self.E / (2.0 * (1.0 + self.nu))


#: Placeholder material for geometric-only mode (no ``materials=``):
#: unit moduli make every rigidity-form value the classic geometric one.
GEOMETRIC_ONLY = SectionMaterial(E=1.0, nu=0.0, G=1.0, name="geometric-only")
