"""``results.particles`` — the particle-cloud composite (ADR 0134 D4).

A particle result is not a node, element or Gauss result: its carriers
move, appear and disappear. This composite therefore stands apart from
the node and element composites and does not mix in their selection:
particles are selected by body name or by explicit particle ids, and
outputs by index or ``step``, never by time (the Domain time may
restart inside a stage, contract 232a §2.5).

Only a ``.ladruno`` with a particle group answers it (ADR 0134 D3); any
other results file, a partitioned set, or a unit-converted view refuses
by name rather than reading nothing.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from ._slabs import ParticleSlab
    from .readers._ladruno_particles import (
        ParticleGrid,
        ParticleRegistry,
        ParticleStage,
        ParticleStageInfo,
    )
    from .Results import Results


#: ``time=`` for particles: an output index, a sequence of them, or a slice.
OutputSelect = Union[None, int, slice, Iterable[int]]


def _is_int(x: object) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


class ParticleResultsComposite:
    """``results.particles``: MPM particle outputs of a ``.ladruno``."""

    def __init__(self, results: "Results") -> None:
        self._r = results

    def __repr__(self) -> str:
        return f"<ParticleResultsComposite of {self._r._reader_path()}>"

    # -- stage resolution ---------------------------------------------

    def _source(self):
        """The reader, when it can answer particle reads; else a refusal."""
        from .readers._ladruno_particles import require_particle_reader

        return require_particle_reader(self._r._reader, self._r._reader_path())

    def stages(self) -> list[str]:
        """The stage ids that hold particles, in stage order."""
        return self._source().particle_stage_ids()

    def _stage(self, stage: Optional[str]) -> ParticleStage:
        reader = self._source()
        if stage is not None:
            sid = self._r._resolve_stage(stage)
        elif self._r._stage_id is not None:
            sid = self._r._stage_id
        else:
            ids = reader.particle_stage_ids()
            if not ids:
                raise ValueError(
                    f"{self._r._reader_path()}: no stage holds a particle "
                    "group (INFO/FORMAT_MINOR_VERSION >= 2 and "
                    "MODEL/PARTICLES, ADR 0134 D3)."
                )
            if len(ids) > 1:
                raise ValueError(
                    f"{self._r._reader_path()}: stages {ids} hold "
                    "particles. Pick one with stage= or results.stage(...)."
                )
            sid = ids[0]
        return reader.particle_stage(sid)

    # -- metadata -----------------------------------------------------

    def info(self, *, stage: Optional[str] = None) -> ParticleStageInfo:
        """Mode, epoch, pattern tag, units and source set of a stage."""
        return self._stage(stage).info()

    def bodies(self, *, stage: Optional[str] = None) -> "dict[int, str]":
        """``{body id: body name}`` of a stage."""
        return self._stage(stage).bodies()

    def registry(self, *, stage: Optional[str] = None) -> ParticleRegistry:
        """Every particle of a stage: id, body, material tag and mass."""
        return self._stage(stage).registry()

    def grid(self, *, stage: Optional[str] = None) -> ParticleGrid:
        """The stage's background grid."""
        return self._stage(stage).grid()

    def components(self, *, stage: Optional[str] = None) -> list[str]:
        """Every component as the file spells it, ``"<FIELD>/<label>"``."""
        return self._stage(stage).components()

    # -- values -------------------------------------------------------

    def get(
        self,
        *,
        component: str,
        body: "str | Iterable[str] | None" = None,
        ids: "Iterable[int] | ndarray | None" = None,
        time: OutputSelect = None,
        step: "int | Iterable[int] | None" = None,
        stage: Optional[str] = None,
    ) -> ParticleSlab:
        """Read one particle field (``"STRESS"``) or component
        (``"STRESS/sigma12"``) as a ragged :class:`ParticleSlab`.

        Parameters
        ----------
        component
            A field name, or ``"<FIELD>/<label>"`` as the file spells it
            (``POSITION/x``, ``STATE_<token>/<label>``).
        body, ids
            Keep the particles of these bodies (by name), or these
            particle ids. At most one of the two.
        time
            Output index (``-1`` is the last), a sequence of indices, or
            a slice. A float is refused: outputs are never keyed by time.
        step
            Keep the outputs with these ``STEP`` values (the Domain commit
            tag). At most one of ``time`` and ``step``.
        stage
            The stage; defaults to the scoped stage, else the only stage
            with particles.
        """
        ps = self._stage(stage)
        sel = self._select_ids(ps, body=body, ids=ids)
        outputs = self._select_outputs(ps, time=time, step=step)
        return ps.read(component, ids=sel, outputs=outputs)

    @staticmethod
    def _select_ids(
        ps: ParticleStage, *, body, ids,
    ) -> "Optional[ndarray]":
        if body is not None and ids is not None:
            raise ValueError("Pass body= or ids=, not both.")
        reg = ps.registry()
        if body is not None:
            names = [body] if isinstance(body, str) else list(body)
            by_name = {name: bid for bid, name in reg.bodies.items()}
            unknown = [n for n in names if n not in by_name]
            if unknown:
                raise KeyError(
                    f"No particle body {unknown}; the bodies are "
                    f"{sorted(by_name)}."
                )
            wanted = [by_name[n] for n in names]
            return reg.ids[np.isin(reg.body_ids, wanted)]
        if ids is not None:
            arr = np.unique(np.asarray(list(ids) if not isinstance(
                ids, np.ndarray) else ids).astype(np.int64, casting="safe"))
            missing = arr[~np.isin(arr, reg.ids)]
            if missing.size:
                raise KeyError(
                    f"Particle id(s) {missing[:10].tolist()} are not in the "
                    f"stage registry ({reg.ids.size} particles)."
                )
            return arr
        return None

    @staticmethod
    def _select_outputs(
        ps: ParticleStage, *, time, step,
    ) -> "Optional[ndarray]":
        if time is not None and step is not None:
            raise ValueError("Pass time= (output index) or step=, not both.")
        _, steps, _ = ps.committed()
        n = steps.size
        if step is not None:
            wanted = [step] if _is_int(step) else list(step)
            if not all(_is_int(s) for s in wanted):
                raise TypeError(f"step= takes integer STEP values, got {step!r}.")
            pos = np.searchsorted(steps, wanted)
            ok = (pos < n) & (steps[np.minimum(pos, max(n - 1, 0))] == wanted) \
                if n else np.zeros(len(wanted), dtype=bool)
            if not bool(np.all(ok)):
                bad = [w for w, good in zip(wanted, ok) if not good]
                raise KeyError(
                    f"No particle output at STEP {bad}; the outputs are at "
                    f"STEP {steps.tolist()}."
                )
            return np.asarray(pos, dtype=np.int64)
        if time is None:
            return None
        if isinstance(time, slice):
            return np.arange(n, dtype=np.int64)[time]
        wanted = [time] if not isinstance(time, Iterable) else list(time)
        if not all(_is_int(k) for k in wanted):
            raise TypeError(
                f"time= takes output indices for particles, got {time!r}. "
                "Particle outputs are keyed by index or step=, never by "
                "time: the Domain time may restart inside a stage "
                "(contract 232a §2.5)."
            )
        out = []
        for k in wanted:
            if not -n <= int(k) < n:
                raise IndexError(
                    f"Output index {k} is out of range; the stage holds "
                    f"{n} output(s)."
                )
            out.append(int(k) % n)
        return np.asarray(out, dtype=np.int64)
