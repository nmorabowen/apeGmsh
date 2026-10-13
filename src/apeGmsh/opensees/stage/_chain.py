"""``StageChain`` -- the shared analysis chain behind ``ops.analysis.chain``.

A frozen value object holding six of a stage's seven analysis-chain
slots (all but the integrator, which usually differs per stage).  It is
NOT a primitive: it registers no tag and emits nothing.  Each stage that
receives it through ``s.analysis(chain, integrator=...)`` emits the
chain's primitives itself, exactly as the keyword form does, so a deck
built with a chain is byte-identical to one built with the slots spelled
out (R3-S1, #1785).
"""

from __future__ import annotations

from dataclasses import dataclass

from .._internal.types import (
    Analysis,
    ConstraintHandler,
    ConvergenceTest,
    LinearSystem,
    Numberer,
    Primitive,
    SolutionAlgorithm,
)

__all__ = ["InertStaticDtWarning", "StageChain", "algorithm_needs_test"]


class InertStaticDtWarning(UserWarning):
    """A ``Static`` stage's ``s.run(dt=)`` differs from its ``LoadControl``
    ``dlam`` (R3 ruling D1, #1785).

    OpenSees ignores ``analyze``'s ``dt`` under ``Static``; each increment
    advances the load factor by ``dlam``.  A differing ``dt`` never acted,
    so the stage warns when it closes.  Omit ``dt`` or set it equal to
    ``dlam`` to silence it.
    """


def algorithm_needs_test(algorithm: object) -> bool:
    """Whether ``algorithm`` iterates, so a stage must give it a test.

    Reads the algorithm class's ``needs_test`` attribute.  An algorithm
    class that does not declare it is treated as iterating (fail closed:
    the test stays required).
    """
    needs = getattr(type(algorithm), "needs_test", True)
    return bool(needs)


#: Slot name -> the primitive base its value must derive from.
_SLOT_TYPES: dict[str, type[Primitive]] = {
    "test": ConvergenceTest,
    "algorithm": SolutionAlgorithm,
    "constraints": ConstraintHandler,
    "numberer": Numberer,
    "system": LinearSystem,
    "analysis": Analysis,
}

#: The slots a chain carries, in the order ``s.analysis`` binds them.
CHAIN_SLOTS: tuple[str, ...] = tuple(_SLOT_TYPES)


@dataclass(frozen=True, kw_only=True, slots=True)
class StageChain:
    """A stage analysis chain shared by several stages.

    Build it with ``ops.analysis.chain(...)`` and pass it to each stage
    as ``s.analysis(chain, integrator=...)``.  ``test`` may be omitted
    only when the algorithm does not iterate (``ops.algorithm.Linear``).
    """

    test: ConvergenceTest | None = None
    algorithm: SolutionAlgorithm
    constraints: ConstraintHandler
    numberer: Numberer
    system: LinearSystem
    analysis: Analysis

    def __post_init__(self) -> None:
        for slot, base in _SLOT_TYPES.items():
            value = getattr(self, slot)
            if value is None and slot == "test":
                continue
            if not isinstance(value, base):
                raise TypeError(
                    f"ops.analysis.chain: {slot}= must be a "
                    f"{base.__name__} primitive (ops.{slot}.<Type>(...)), "
                    f"got {type(value).__name__}."
                )
        if self.test is None and algorithm_needs_test(self.algorithm):
            raise ValueError(
                "ops.analysis.chain: test= is required with algorithm "
                f"{type(self.algorithm).__name__}, which iterates; only "
                "a non-iterating algorithm (Linear) may omit it."
            )
