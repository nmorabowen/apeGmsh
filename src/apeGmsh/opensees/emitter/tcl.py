"""
TclEmitter — accumulates a Tcl deck as a list of strings.

The Tcl emitter is a pure text builder. Every Protocol method appends
one (or, for block-emit primitives, a few) lines to ``_lines``. The
deck is retrieved via :meth:`lines` once :meth:`emit` has driven the
full BuiltModel.

Tcl-specific dialect choices:

* Sections that take blocks (``Fiber``) emit
  ``section Fiber 1 -GJ 1.0e9 \\{`` with the opening brace; subsequent
  ``patch`` / ``fiber`` / ``layer`` lines indent inside the block;
  ``section_close`` emits ``\\}``.

* ``pattern_open`` is dispatched on the type token: ``Plain`` and
  ``MultiSupport`` open a block (``pattern Plain 1 1 \\{``); single-line
  patterns like ``UniformExcitation`` emit a single line and
  ``pattern_close`` is a no-op.

* Floats render with Python's ``repr(float)`` precision — enough digits
  to round-trip through the parser without rewriting the user's
  numbers.
"""
from __future__ import annotations

import os
from typing import (
    IO, Any, Callable, Literal, NamedTuple, Sequence, SupportsIndex,
)

from .base import StrategySpec


__all__ = ["PartitionSpan", "TclEmitter"]


class PartitionSpan(NamedTuple):
    """Line-index span of one ``if {[getPID] == K} { ... }`` block.

    Recorded by ``partition_open`` / ``partition_close`` (ADR 0061) so
    the per-rank deck writer can slice rank-local fragments out of the
    monolithic line buffer. Recording is observation-only — the emitted
    lines are unchanged.

    ``header`` is the ``if {[getPID] == K} {`` line; the body occupies
    ``[body_start, body_end)`` (indented one level); ``end`` is the
    index just past the closing ``}`` and its trailing blank line, i.e.
    the full block is ``lines[header:end]``.
    """

    rank: int
    header: int
    body_start: int
    body_end: int
    end: int


#: Pattern type tokens that open a Tcl block (``\\{`` ... ``\\}``). Other
#: pattern type tokens emit a single line and ``pattern_close`` is a no-op.
_BLOCK_PATTERNS: tuple[str, ...] = ("Plain", "MultiSupport")


class _LineBuf(list[str]):
    """Line buffer with an optional partition-indent prefix.

    Behaves as a plain ``list[str]`` for ``insert``, iteration, slicing,
    and the ``lines()`` copy-out. When ``indent`` is non-empty, every
    ``append(line)`` prepends ``indent`` to ``line`` before storing.
    This keeps the existing ``self._lines.append(...)`` call sites in
    :class:`TclEmitter` working unchanged while
    ``partition_open(K)`` / ``partition_close()`` toggle the indent.

    **Dual mode** (ADR 0065 Tier 2 / plan_emit_memory_columnar.md
    A1–A3): the default is list accumulation, byte-identical to
    before. After :meth:`attach_sink`, ``append`` writes
    ``indent + line + "\\n"`` straight through to the sink and stores
    nothing — the buffer stays empty and peak emit memory stops
    scaling with deck size. The partition-indent logic is unchanged
    (it lives here in ``append``), so all emitter call sites are
    captured for free. In stream mode, ``insert`` (the ``preamble()``
    front-insert) fails loud — earlier lines are already on disk.

    ``strip_partition_indent`` (per-rank fragment routing, Decision
    §3) makes the sink write ``(indent + line).removeprefix("    ")``
    — the exact post-hoc transform ``_write_per_rank_tcl`` applies to
    body lines, so live-streamed fragments are byte-identical to the
    sliced ones (including the indent-0 override paths and blank
    lines inside partition blocks).
    """

    __slots__ = ("indent", "_sink_write", "_strip_partition_indent",
                 "_streamed")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.indent: str = ""
        # Stream-mode sink (ADR 0065 Tier 2). ``None`` = list mode.
        self._sink_write: Callable[[str], object] | None = None
        self._strip_partition_indent: bool = False
        # Lines already written through the sink (so ``line_count()``
        # stays monotone across modes).
        self._streamed: int = 0

    def append(self, line: str) -> None:
        write = self._sink_write
        if write is not None:
            if self.indent:
                line = self.indent + line
            if self._strip_partition_indent:
                line = line.removeprefix("    ")
            write(line + "\n")
            self._streamed += 1
            return
        if self.indent:
            super().append(self.indent + line)
        else:
            super().append(line)

    def insert(self, index: SupportsIndex, line: str) -> None:
        if self._sink_write is not None:
            raise RuntimeError(
                "_LineBuf.insert() is illegal in stream mode — earlier "
                "lines are already on disk, so a front-insert (e.g. "
                "preamble()) cannot be honored. Call it before "
                "stream_to() (ADR 0065 Tier 2 Decision §2)."
            )
        super().insert(index, line)

    @property
    def streaming(self) -> bool:
        """True while a write-through sink is attached (stream mode)."""
        return self._sink_write is not None

    def attach_sink(self, write: "Callable[[str], object]") -> None:
        """Enter stream mode: flush buffered lines (the banner and any
        ``preamble()`` header lines present at attach time), clear the
        buffer, and route every subsequent ``append`` to ``write``."""
        for ln in self:
            write(ln + "\n")
        self._streamed += len(self)
        self.clear()
        self._sink_write = write
        self._strip_partition_indent = False

    def switch_sink(
        self, write: "Callable[[str], object]",
        *, strip_partition_indent: bool = False,
    ) -> None:
        """Redirect the active sink (per-rank fragment routing)."""
        self._sink_write = write
        self._strip_partition_indent = strip_partition_indent

    def detach_sink(self) -> None:
        """Leave stream mode (stream_finish / stream_abort teardown)."""
        self._sink_write = None
        self._strip_partition_indent = False

    def total_line_count(self) -> int:
        """Lines emitted so far — streamed + still buffered."""
        return self._streamed + len(self)


class _TclStreamState:
    """Bookkeeping for one streaming emission (ADR 0065 Tier 2 /
    plan_emit_memory_columnar.md A1–A3).

    Everything is written to ``.tmp`` siblings and promoted with
    ``os.replace`` on clean completion (:meth:`TclEmitter.stream_finish`)
    — a mid-emit exception never leaves a half-written final deck
    (Decision §4); :meth:`TclEmitter.stream_abort` removes the temps.
    """

    __slots__ = (
        "path", "per_rank", "driver_tmp", "driver_handle", "ranks_dir",
        "seq", "fragment_handle", "fragment_tmp", "fragment_final",
        "fragment_rank", "fragment_name", "replacements",
        "fragments_written",
    )

    def __init__(self, path: str, per_rank: bool) -> None:
        self.path = path
        self.per_rank = per_rank
        self.driver_tmp = path + ".tmp"
        self.driver_handle: IO[str] | None = None
        # Per-rank fragment routing (Decision §3).
        self.ranks_dir = os.path.join(
            os.path.dirname(os.path.abspath(path)), "ranks",
        )
        # ``seq[rank]`` = the rank's next 0-based block counter —
        # identical numbering to ``_write_per_rank_tcl`` (which walks
        # the spans in emission order).
        self.seq: dict[int, int] = {}
        self.fragment_handle: IO[str] | None = None
        self.fragment_tmp: str | None = None
        self.fragment_final: str | None = None
        self.fragment_rank: int | None = None
        self.fragment_name: str | None = None
        # (tmp, final) pairs to promote at stream_finish.
        self.replacements: list[tuple[str, str]] = []
        self.fragments_written: int = 0


def _fmt_value(v: Any) -> str:
    """Render one positional argument for Tcl.

    Strings pass through unchanged (no quoting — OpenSees Tcl tokens
    don't normally need it for our vocabulary). Booleans are coerced
    to ``1`` / ``0`` (OpenSees doesn't speak Python ``True``).
    Integers and floats use their ``repr`` (which preserves enough
    digits for floats to round-trip).
    """
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    # Fallback — should not happen for the typed-emit boundary.
    return str(v)


def _join(*parts: Any) -> str:
    """Render a Tcl line from ``parts`` separated by single spaces.

    The int / float / str dispatch is inlined (rather than calling
    :func:`_fmt_value` per token) — one function call per LINE instead
    of per token; the bulk bands emit ~10 tokens a line and the
    per-token call overhead was a measured flat-emit hotspot. Output is
    identical: odd types still route through :func:`_fmt_value`.
    """
    out: list[str] = []
    append = out.append
    for p in parts:
        c = p.__class__
        if c is int:
            append(str(p))
        elif c is float:
            append(repr(p))
        elif c is str:
            append(p)
        else:
            append(_fmt_value(p))
    return " ".join(out)


class TclEmitter:
    """Accumulates an OpenSees Tcl deck as a list of strings.

    Construct with no arguments; drive it through ``BuiltModel.emit``;
    retrieve the deck via :meth:`lines`.

    The class also exposes :meth:`preamble` to inject a leading comment
    line. Most users will rely on the default banner emitted by
    ``__init__``.
    """

    def __init__(self) -> None:
        # ``_lines`` is a :class:`_LineBuf` so ``partition_open`` can
        # toggle a per-rank indent without every ``append`` call site
        # needing to know. All existing primitives keep calling
        # ``self._lines.append(...)`` unchanged.
        self._lines: _LineBuf = _LineBuf([
            "# auto-generated by apeGmsh.opensees; do not edit by hand",
        ])
        # Tracks the type token of the currently-open pattern (for
        # pattern_close to decide whether to emit ``\\}``). None means
        # no pattern open or the open pattern is single-line.
        self._open_block_pattern: str | None = None
        # Partition-emission state (ADR 0027 / P4). ``_in_partition``
        # is True between ``partition_open`` and ``partition_close``;
        # ``_partition_shim_emitted`` flips on the first
        # ``partition_open`` call so the ``proc getPID`` fallback is
        # emitted exactly once. Per-rank content is indented one level
        # (4 spaces) so the closing ``\\}`` lines up with the opening
        # ``if {[getPID] == K}``.
        self._partition_shim_emitted: bool = False
        # Per-rank block spans (ADR 0061). ``partition_open`` stashes
        # the open block's ``(rank, header_index)``; ``partition_close``
        # completes the :class:`PartitionSpan`. Observation-only — used
        # by ``apeSees.tcl(per_rank=True)`` to slice rank-local
        # fragments; never alters the emitted lines.
        self._partition_spans: list[PartitionSpan] = []
        self._open_partition: tuple[int, int] | None = None
        # Step-hook state (Phase SSI-1).  ``_step_hooks_registered``
        # flips to True the first time ``step_hook_ramp`` runs and is
        # checked by ``analyze`` to decide whether to wrap the analyze
        # loop with hook-dispatcher calls.
        # ``_hook_dispatcher_emitted`` toggles on the first
        # ``step_hook_ramp`` call so the dispatcher boilerplate (list
        # variables + dispatcher procs) emits exactly once.
        self._step_hooks_registered: bool = False
        self._hook_dispatcher_emitted: bool = False
        # Streaming sink state (ADR 0065 Tier 2 /
        # plan_emit_memory_columnar.md A1–A3). ``None`` = list mode
        # (the default, byte-identical to before).
        self._stream: _TclStreamState | None = None

    # -- Output --------------------------------------------------------------

    def _assert_not_streaming(self, what: str) -> None:
        if self._stream is not None:
            raise RuntimeError(
                f"TclEmitter.{what} is unavailable in stream mode — the "
                "deck was written through to disk, not accumulated "
                "(ADR 0065 Tier 2). Use the default list mode for "
                "in-memory introspection."
            )

    def lines(self) -> list[str]:
        """Return a copy of the accumulated Tcl lines.

        Fails loud in stream mode (ADR 0065 Tier 2) — streamed lines
        are on disk, not in memory.
        """
        self._assert_not_streaming("lines()")
        return list(self._lines)

    def line_count(self) -> int:
        """Number of accumulated lines, without copying the buffer.

        ``len(emitter.lines())`` clones the whole line list (multi-M
        entries on large models) just to read a length — span/module
        recording must use this instead (ADR 0065 A0). In stream mode
        this counts lines written through the sink.
        """
        return self._lines.total_line_count()

    def line_buffer(self) -> "list[str]":
        """The internal line buffer — READ-ONLY by contract.

        For O(1)-extra-memory consumers that only iterate or slice
        (the per-rank / split deck writers). Mutating the returned
        list corrupts the emitter; use :meth:`lines` for a defensive
        copy (ADR 0065 A0). Fails loud in stream mode (ADR 0065
        Tier 2).
        """
        self._assert_not_streaming("line_buffer()")
        return self._lines

    def write_to(self, f: "Any") -> None:
        """Stream the accumulated deck to an open text handle.

        Writes the internal buffer line-by-line, so neither the
        ``list(self._lines)`` copy of :meth:`lines` nor a single joined
        deck-sized string is materialized — peak write-time memory drops
        to the OS write buffer (ADR 0065 Tier 1). Output is byte-identical
        to ``f.write("\\n".join(self.lines()) + "\\n")``. Fails loud in
        stream mode (the deck already went to the sink).
        """
        self._assert_not_streaming("write_to()")
        write = f.write
        for line in self._lines:
            write(line)
            write("\n")

    def preamble(self, text: str) -> None:
        """Insert ``text`` as a leading comment line. Idempotent.

        In stream mode this fails loud once streaming has begun —
        earlier lines are already on disk (ADR 0065 Tier 2 Decision
        §2); call it before :meth:`stream_to`.
        """
        self._lines.insert(0, f"# {text}")

    # -- Streaming sink (ADR 0065 Tier 2 / plan A1–A3) -----------------------

    def stream_to(self, path: str, *, per_rank: bool = False) -> None:
        """Attach a write-through file sink — stream mode (ADR 0065
        Tier 2 / plan_emit_memory_columnar.md A1–A3).

        Every subsequent ``append`` writes straight to ``path + ".tmp"``
        and stores nothing, so peak emit memory stops scaling with deck
        size. The banner (and any ``preamble()`` lines) buffered at
        attach time flush to the sink first, preserving the
        leading-comment contract (Decision §2).

        ``per_rank=True`` additionally live-routes every
        ``partition_open(K)`` / ``partition_close()`` block body to
        ``ranks/rank<K>_<seq>.tcl`` (Decision §3) — naming and content
        byte-identical to the post-hoc ``_write_per_rank_tcl`` slicing,
        with the driver carrying the same one-line source guards.

        All files are written as ``.tmp`` siblings and promoted with
        ``os.replace`` by :meth:`stream_finish` (Decision §4); call
        :meth:`stream_abort` on failure to remove the temps.
        """
        if self._stream is not None:
            raise RuntimeError(
                "TclEmitter.stream_to: a streaming sink is already "
                "attached — one stream per emitter."
            )
        st = _TclStreamState(path, per_rank)
        if per_rank:
            os.makedirs(st.ranks_dir, exist_ok=True)
        st.driver_handle = open(st.driver_tmp, "w", encoding="utf-8")
        self._stream = st
        self._lines.attach_sink(st.driver_handle.write)

    def stream_fragment_count(self) -> int:
        """Per-rank fragments completed so far (stream mode only)."""
        if self._stream is None:
            raise RuntimeError(
                "TclEmitter.stream_fragment_count: not in stream mode."
            )
        return self._stream.fragments_written

    def stream_finish(self) -> None:
        """Close the sink(s) and atomically promote every ``.tmp`` file
        onto its final path (ADR 0065 Tier 2 Decision §4).

        Fragments are promoted before the driver, so the driver (the
        deck entry point) only appears once everything it sources
        exists.
        """
        st = self._stream
        if st is None:
            raise RuntimeError(
                "TclEmitter.stream_finish: not in stream mode — call "
                "stream_to() first."
            )
        if st.fragment_handle is not None:
            raise RuntimeError(
                "TclEmitter.stream_finish: a per-rank fragment is still "
                "open (unbalanced partition_open/partition_close)."
            )
        assert st.driver_handle is not None
        st.driver_handle.close()
        for tmp, final in st.replacements:
            os.replace(tmp, final)
        os.replace(st.driver_tmp, st.path)
        self._lines.detach_sink()
        self._stream = None

    def stream_abort(self) -> None:
        """Best-effort teardown after a mid-emit failure: close any open
        handles and remove every ``.tmp`` file, leaving no final deck
        (ADR 0065 Tier 2 Decision §4). Safe to call when not streaming.
        """
        st = self._stream
        if st is None:
            return
        for handle in (st.fragment_handle, st.driver_handle):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        temps = [tmp for tmp, _final in st.replacements]
        if st.fragment_tmp is not None:
            temps.append(st.fragment_tmp)
        temps.append(st.driver_tmp)
        for tmp in temps:
            try:
                os.remove(tmp)
            except OSError:
                pass
        self._lines.detach_sink()
        self._stream = None

    # -- Model ---------------------------------------------------------------

    def model(self, *, ndm: int, ndf: int) -> None:
        self._lines.append(f"model BasicBuilder -ndm {ndm} -ndf {ndf}")

    def node(
        self, tag: int, *coords: float, ndf: int | None = None,
    ) -> None:
        # Fast path for the dominant deck band (one line per mesh
        # node): plain-int tag + plain-float coords render via a single
        # f-string. ``{x!r}`` on an exact float is exactly what _join
        # emits, so the output is byte-identical; anything else falls
        # through to the generic path.
        if ndf is None:
            if tag.__class__ is int and len(coords) == 3:
                x, y, z = coords
                if (
                    x.__class__ is float
                    and y.__class__ is float
                    and z.__class__ is float
                ):
                    self._lines.append(f"node {tag} {x!r} {y!r} {z!r}")
                    return
            self._lines.append(_join("node", tag, *coords))
        else:
            # Per-node ``-ndf`` override (used for 6-DOF phantom nodes in
            # otherwise 3-DOF models; ADR 0022 INV-3).
            self._lines.append(
                _join("node", tag, *coords, "-ndf", ndf)
            )

    def fix(self, tag: int, *dofs: int) -> None:
        self._lines.append(_join("fix", tag, *dofs))

    def mass(self, tag: int, *values: float) -> None:
        self._lines.append(_join("mass", tag, *values))

    # -- MP constraints (ADR 0022, Phase 7b) -----------------------------

    def equalDOF(self, master: int, slave: int, *dofs: int) -> None:
        self._lines.append(_join("equalDOF", master, slave, *dofs))

    def equalDOF_mixed(
        self, master: int, slave: int,
        dof_pairs: "Sequence[tuple[int, int]]",
    ) -> None:
        flat = [int(d) for pair in dof_pairs for d in pair]
        self._lines.append(
            _join("equalDOF_Mixed", master, slave, len(dof_pairs), *flat)
        )

    def rigidLink(self, kind: str, master: int, slave: int) -> None:
        # ``rigidLink {beam|bar} $master $slave`` — kind is unquoted.
        self._lines.append(_join("rigidLink", kind, master, slave))

    def rigidDiaphragm(
        self, perp_dir: int, master: int, *slaves: int,
    ) -> None:
        self._lines.append(
            _join("rigidDiaphragm", perp_dir, master, *slaves)
        )

    def embeddedNode(
        self, ele_tag: int, cnode: int, *master_nodes: int,
        stiffness: float = 1.0e18,
        stiffness_p: float | None = None,
        rotational: bool = False,
        pressure: bool = False,
    ) -> None:
        # ASDEmbeddedNodeElement covers tie / tied_contact / mortar /
        # embedded surface-coupling primitives (ADR 0022).  Optional
        # flags (-rot / -p / -K / -KP) are exposed via kwargs and
        # serialised in parser order by `_build_embedded_flag_args`
        # (ADR 0035).
        from .base import _build_embedded_flag_args
        flag_args = _build_embedded_flag_args(
            stiffness, stiffness_p, rotational, pressure,
        )
        self._lines.append(
            _join(
                "element", "ASDEmbeddedNodeElement",
                ele_tag, cnode, *master_nodes, *flag_args,
            )
        )

    def embedded_rebar(
        self, ele_tag: int, *args: int | float | str,
    ) -> None:
        # LadrunoEmbeddedRebar coupling (g.reinforce, ADR 20). The full
        # positional arg list is pre-built by the R0 ``embedded_rebar_args``
        # grammar builder; we just front it with the element command +
        # token. Fork-only — bites at run time on a stock OpenSees, not
        # here at emit.
        self._lines.append(
            _join("element", "LadrunoEmbeddedRebar", ele_tag, *args))

    def equationConstraint(
        self, cnode: int, cdof: int, ccoef: float,
        retained: "Sequence[tuple[int, int, float]]",
    ) -> None:
        # EQ_Constraint (upstream): the exact / explicit-safe tie route
        # (ADR 0068). One line per tied DOF:
        #   equationConstraint $cnode $cdof $ccoef  $rn $rd $rc ...
        flat: list[int | float] = []
        for rn, rd, rc in retained:
            flat += [int(rn), int(rd), float(rc)]
        self._lines.append(
            _join("equationConstraint", int(cnode), int(cdof),
                  float(ccoef), *flat)
        )

    def embedded_node(
        self, ele_tag: int, *args: int | float | str,
    ) -> None:
        # LadrunoEmbeddedNode coupling (g.embed). Args pre-built by the
        # ``embedded_node_args`` grammar builder; fork-only at run time.
        self._lines.append(
            _join("element", "LadrunoEmbeddedNode", ele_tag, *args))

    def contact_surface(
        self, tag: int, *args: int | float | str,
    ) -> None:
        # Fork contact surface (g.constraints.contact). Args pre-built by
        # ``contact_surface_args``; fork-only at run time.
        self._lines.append(_join("contactSurface", tag, *args))

    def contact(
        self, tag: int, *args: int | float | str,
    ) -> None:
        # Fork contact verb. Args pre-built by ``contact_args``; fork-only.
        self._lines.append(_join("contact", tag, *args))

    def contact_plane(
        self, tag: int, *args: int | float | str,
    ) -> None:
        # Fork rigid-plane contact (g.constraints.contact_plane). Args pre-built
        # by ``contact_plane_args``; fork-only at run time.
        self._lines.append(_join("contactPlane", tag, *args))

    def mp_constraint_comment(self, name: str) -> None:
        # Round-trips the user's declaration label into the deck (INV-2).
        self._lines.append(f"# {name}")

    # -- Regions -------------------------------------------------------------

    def region(self, tag: int, *args: int | float | str) -> None:
        self._lines.append(_join("region", tag, *args))

    # -- Damping (ADR 0053) --------------------------------------------------

    def rayleigh(
        self,
        alpha_m: float,
        beta_k: float,
        beta_k_init: float,
        beta_k_comm: float,
    ) -> None:
        self._lines.append(
            _join("rayleigh", alpha_m, beta_k, beta_k_init, beta_k_comm),
        )

    def damping(
        self, damp_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._lines.append(_join("damping", damp_type, tag, *args))

    def modal_damping(self, *factors: float) -> None:
        self._lines.append(_join("modalDamping", *factors))

    # -- Constitutive --------------------------------------------------------

    def uniaxialMaterial(
        self, mat_type: str, tag: int, *params: float | str,
    ) -> None:
        self._lines.append(_join("uniaxialMaterial", mat_type, tag, *params))

    def nDMaterial(
        self, mat_type: str, tag: int, *params: float | str,
    ) -> None:
        self._lines.append(_join("nDMaterial", mat_type, tag, *params))

    def section(
        self, sec_type: str, tag: int, *params: float | str,
    ) -> None:
        self._lines.append(_join("section", sec_type, tag, *params))

    def geomTransf(self, t_type: str, tag: int, *vec: float) -> None:
        self._lines.append(_join("geomTransf", t_type, tag, *vec))

    # -- Sections that take blocks (Fiber) ----------------------------------

    def section_open(
        self, sec_type: str, tag: int, *params: float | str,
    ) -> None:
        # Open the block — Tcl's curly brace must be the LAST token on
        # the line for the parser to consume the body.
        head = _join("section", sec_type, tag, *params)
        self._lines.append(f"{head} {{")

    def section_close(self) -> None:
        self._lines.append("}")

    def patch(self, kind: str, *args: int | float) -> None:
        self._lines.append("    " + _join("patch", kind, *args))

    def fiber(
        self, y: float, z: float, area: float, mat_tag: int,
    ) -> None:
        self._lines.append("    " + _join("fiber", y, z, area, mat_tag))

    def layer(self, kind: str, *args: int | float) -> None:
        self._lines.append("    " + _join("layer", kind, *args))

    # -- Beam integration rules ---------------------------------------------

    def beamIntegration(
        self, rule_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._lines.append(_join("beamIntegration", rule_type, tag, *args))

    # -- Topology ------------------------------------------------------------

    def element(
        self, ele_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._lines.append(_join("element", ele_type, tag, *args))

    # -- Time series --------------------------------------------------------

    def timeSeries(
        self, ts_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._lines.append(_join("timeSeries", ts_type, tag, *args))

    # -- Patterns -----------------------------------------------------------

    def pattern_open(
        self, p_type: str, tag: int, *args: int | float | str,
    ) -> None:
        head = _join("pattern", p_type, tag, *args)
        if p_type in _BLOCK_PATTERNS:
            self._lines.append(f"{head} {{")
            self._open_block_pattern = p_type
        else:
            self._lines.append(head)
            self._open_block_pattern = None

    def pattern_close(self) -> None:
        if self._open_block_pattern is not None:
            self._lines.append("}")
            self._open_block_pattern = None
        # else: single-line pattern; pattern_close is a no-op.

    def load(self, tag: int, *forces: float) -> None:
        # Inside a Plain pattern: indent for readability.
        prefix = "    " if self._open_block_pattern else ""
        self._lines.append(prefix + _join("load", tag, *forces))

    def eleLoad(self, *args: int | float | str) -> None:
        prefix = "    " if self._open_block_pattern else ""
        self._lines.append(prefix + _join("eleLoad", *args))

    def sp(self, tag: int, dof: int, value: float) -> None:
        prefix = "    " if self._open_block_pattern else ""
        self._lines.append(prefix + _join("sp", tag, dof, value))

    def sp_hold(self, node: int, dof: int) -> None:
        # HOLD support (ADR 0052): pin the DOF at its current deformed
        # displacement, captured at runtime via ``nodeDisp``, with
        # ``-const`` so the value is never scaled by the load factor.
        prefix = "    " if self._open_block_pattern else ""
        self._lines.append(
            prefix + f"sp {node} {dof} [nodeDisp {node} {dof}] -const"
        )

    # -- Recorders ----------------------------------------------------------

    def recorder(self, kind: str, *args: int | float | str) -> None:
        self._lines.append(_join("recorder", kind, *args))

    def recorder_declaration_begin(
        self,
        *,
        declaration_name: str,
        record_name: str | None,
        category: str,
        components: tuple[str, ...],
        raw: tuple[str, ...] = (),
        pg: tuple[str, ...] = (),
        label: tuple[str, ...] = (),
        selection: tuple[str, ...] = (),
        ids: tuple[int, ...] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        file_root: str = ".",
    ) -> None:
        """Phase 9 schema 2.3.0 declaration metadata — no-op for Tcl."""
        del (
            declaration_name, record_name, category, components, raw,
            pg, label, selection, ids, dt, n_steps, file_root,
        )

    def recorder_declaration_end(self) -> None:
        """Phase 9 schema 2.3.0 declaration metadata — no-op for Tcl."""

    # -- Analysis chain -----------------------------------------------------

    def constraints(self, c_type: str, *args: int | float | str) -> None:
        self._lines.append(_join("constraints", c_type, *args))

    def numberer(self, n_type: str) -> None:
        self._lines.append(_join("numberer", n_type))

    def system(self, s_type: str, *args: int | float | str) -> None:
        self._lines.append(_join("system", s_type, *args))

    def test(self, t_type: str, *args: int | float | str) -> None:
        self._lines.append(_join("test", t_type, *args))

    def algorithm(self, a_type: str, *args: int | float | str) -> None:
        self._lines.append(_join("algorithm", a_type, *args))

    def integrator(self, i_type: str, *args: int | float | str) -> None:
        self._lines.append(_join("integrator", i_type, *args))

    def analysis(self, a_type: str) -> None:
        self._lines.append(_join("analysis", a_type))

    def analyze(
        self, *, steps: int, dt: float | None = None,
        label: str | None = None,
        strategy: StrategySpec | None = None,
    ) -> int:
        # Fail-loud per-increment loop (see the Emitter Protocol note):
        # a batched ``analyze N`` short-circuits internally on the first
        # failed increment and the deck would silently run on with the
        # stage partial (or not applied at all).  Every increment is
        # checked; the first failure aborts the deck (Tcl ``error``)
        # with a banner naming the loop, increment, and pseudo-time.
        # The private loop variable name avoids clashing with anything
        # the user might define in their own Tcl.
        #
        # ADR 0057 Phase A: with ``strategy``, the loop walks the rung
        # ladder on a failed increment (rung 0 = the chain's own
        # algorithm) and restores rung 0 after a rescue; exhaustion
        # aborts with the same banner naming the ladder.
        n = int(steps)
        where = f" of stage '{label}'".replace('"', "'") if label else ""
        call = "analyze 1" if dt is None else _join("analyze", 1, dt)
        if strategy is None:
            self._lines.append(
                f"for {{set _apesees_i 0}} {{$_apesees_i < {n}}} "
                f"{{incr _apesees_i}} {{"
            )
            prev_indent = self._lines.indent
            self._lines.indent = prev_indent + "    "
            if self._step_hooks_registered:
                self._lines.append("_apesees_call_before_step")
            self._lines.append(f"if {{[{call}] != 0}} {{")
            self._lines.indent = prev_indent + "        "
            self._lines.append(
                'error "apeGmsh: analyze FAILED at increment '
                f"[expr {{$_apesees_i + 1}}]/{n}{where} "
                '(pseudo-time [getTime]) -- aborting, the remaining deck '
                'would run on a partial state"'
            )
            self._lines.indent = prev_indent + "    "
            self._lines.append("}")
            if self._step_hooks_registered:
                self._lines.append("_apesees_call_after_step")
            self._lines.indent = prev_indent
            self._lines.append("}")
            return 0

        rungs_literal = "{" + " ".join(
            "{" + _join(*rung) + "}" for rung in strategy.rungs
        ) + "}"
        sname = strategy.name.replace('"', "'").replace("[", "(").replace("]", ")")
        self._lines.append(f"set _apesees_rungs {rungs_literal}")
        self._lines.append(
            f"for {{set _apesees_i 0}} {{$_apesees_i < {n}}} "
            f"{{incr _apesees_i}} {{"
        )
        prev_indent = self._lines.indent
        self._lines.indent = prev_indent + "    "
        if self._step_hooks_registered:
            self._lines.append("_apesees_call_before_step")
        self._lines.append("set _apesees_ok 0")
        self._lines.append("set _apesees_r 0")
        self._lines.append("foreach _apesees_rung $_apesees_rungs {")
        self._lines.indent = prev_indent + "        "
        self._lines.append("if {$_apesees_r > 0} {")
        self._lines.indent = prev_indent + "            "
        self._lines.append(
            f'puts "apeGmsh strategy \'{sname}\': increment '
            f"[expr {{$_apesees_i + 1}}]/{n}{where} -> rung "
            '$_apesees_r ($_apesees_rung)"'
        )
        self._lines.append("eval algorithm $_apesees_rung")
        self._lines.indent = prev_indent + "        "
        self._lines.append("}")
        self._lines.append(f"if {{[{call}] == 0}} {{ set _apesees_ok 1; break }}")
        self._lines.append("incr _apesees_r")
        self._lines.indent = prev_indent + "    "
        self._lines.append("}")
        self._lines.append("if {!$_apesees_ok} {")
        self._lines.indent = prev_indent + "        "
        self._lines.append(
            'error "apeGmsh: analyze FAILED at increment '
            f"[expr {{$_apesees_i + 1}}]/{n}{where} "
            "(pseudo-time [getTime]) -- aborting after exhausting "
            f"strategy ladder '{sname}' ({len(strategy.rungs)} rungs); "
            'the remaining deck would run on a partial state"'
        )
        self._lines.indent = prev_indent + "    "
        self._lines.append("}")
        self._lines.append(
            "if {$_apesees_r > 0} { eval algorithm [lindex $_apesees_rungs 0] }"
        )
        if self._step_hooks_registered:
            self._lines.append("_apesees_call_after_step")
        self._lines.indent = prev_indent
        self._lines.append("}")
        return 0

    def eigen(
        self, num_modes: int, *, solver: str = "-genBandArpack",
    ) -> list[float]:
        self._lines.append(_join("eigen", solver, num_modes))
        return []

    def profiler(self, *args: int | float | str) -> None:
        self._lines.append(_join("profiler", *args))

    # -- Stress control (Phase SSI-1: initial_stress + ramping hooks) -------

    def addToParameter(
        self, tag: int, ele_tag: int, response: str,
    ) -> None:
        self._lines.append(
            _join("addToParameter", tag, "element", ele_tag, response)
        )

    def flip_element_stage(
        self, pid: int, ele_tags: tuple[int, ...],
    ) -> None:
        self._lines.append(_join("parameter", int(pid)))
        for et in ele_tags:
            self._lines.append(
                _join("addToParameter", int(pid), "element", int(et), "stage")
            )
        self._lines.append(_join("updateParameter", int(pid), 1))
        self._lines.append(_join("remove", "parameter", int(pid)))

    def step_hook_ramp(
        self,
        name: str,
        *,
        targets: tuple[tuple[int, float], ...],
        n_steps_to_full: float,
        phase: Literal["before", "after"] = "before",
    ) -> None:
        # 1. Dispatcher boilerplate (idempotent — emitted on first call).
        if not self._hook_dispatcher_emitted:
            self._emit_hook_dispatcher_boilerplate()
            self._hook_dispatcher_emitted = True
        # 2. parameter declarations (one per ramped component).  These
        # are global — emit at the outer indent so partition_open does
        # NOT scope them per-rank.
        for tag, _target in targets:
            self._lines.append(_join("parameter", int(tag)))
        # 3. The per-step proc body.  Uses a Tcl array keyed by a name
        # that includes the hook name (so multiple ramps don't clash).
        self._emit_hook_ramp_proc(name, targets, n_steps_to_full)
        # 4. Register with the appropriate dispatcher list.
        list_var = (
            "_apesees_before_step_hooks"
            if phase == "before"
            else "_apesees_after_step_hooks"
        )
        self._lines.append(f"lappend {list_var} {name}")
        # 5. Flip the flag so future analyze() calls know to wrap.
        self._step_hooks_registered = True

    def _emit_hook_dispatcher_boilerplate(self) -> None:
        """Emit the once-per-deck hook dispatcher infrastructure.

        Mirrors STKO's ``STKO_CALL_OnBeforeAnalyze`` pattern but with
        apeSees-prefixed names so emitted decks don't collide with
        hand-written STKO blocks.
        """
        # Emit at indent 0 (the dispatcher is global, not per-rank).
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(
            "# apeSees per-step hook dispatcher (Phase SSI-1)"
        )
        self._lines.append("set _apesees_before_step_hooks {}")
        self._lines.append("set _apesees_after_step_hooks {}")
        self._lines.append("proc _apesees_call_before_step {} {")
        self._lines.append(
            "    global _apesees_before_step_hooks"
        )
        self._lines.append(
            "    foreach _f $_apesees_before_step_hooks { $_f }"
        )
        self._lines.append("}")
        self._lines.append("proc _apesees_call_after_step {} {")
        self._lines.append("    global _apesees_after_step_hooks")
        self._lines.append(
            "    foreach _f $_apesees_after_step_hooks { $_f }"
        )
        self._lines.append("}")
        self._lines.indent = prev_indent

    # -- Staged analysis (Phase SSI-2.A) ------------------------------------

    def stage_open(self, name: str) -> None:
        # Human-readable banner.  Emitted at the outer indent so it
        # remains visible even inside partition-open blocks.
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(f"# === Stage: {name} ===")
        self._lines.indent = prev_indent

    def domain_change(self) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append("domainChange")
        self._lines.indent = prev_indent

    def stage_close(self) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append("loadConst -time 0.0")
        self._lines.append("wipeAnalysis")
        if self._step_hooks_registered:
            # Clear the dispatcher lists so the next stage's hooks
            # don't inherit this stage's procs.  Proc bodies persist
            # but are unreachable until a future ``lappend`` puts
            # them back into the list.
            self._lines.append("set _apesees_before_step_hooks {}")
            self._lines.append("set _apesees_after_step_hooks {}")
            # Reset the "are hooks live?" flag so the next stage's
            # ``analyze`` emits a bare ``analyze N`` line UNLESS that
            # stage also registers a ramp (the orchestrator emits
            # ramp registrations BEFORE the next analyze).
            self._step_hooks_registered = False
        self._lines.indent = prev_indent

    # -- Staged-analysis mutators (Phase SSI-2.E) ---------------------------

    def set_time(self, t: float) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(f"setTime {repr(float(t))}")
        self._lines.indent = prev_indent

    def set_creep(self, on: bool) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(f"setCreep {1 if on else 0}")
        self._lines.indent = prev_indent

    def reset(self) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append("reset")
        self._lines.indent = prev_indent

    def remove_sp(self, node: int, dof: int) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(f"remove sp {int(node)} {int(dof)}")
        self._lines.indent = prev_indent

    def remove_element(self, tag: int) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(f"remove element {int(tag)}")
        self._lines.indent = prev_indent

    def _emit_hook_ramp_proc(
        self,
        name: str,
        targets: tuple[tuple[int, float], ...],
        n_steps_to_full: float,
    ) -> None:
        """Emit the per-step linear ramp procedure body for one hook.

        The proc tracks a private Tcl array ``${name}_state`` carrying
        a step counter plus one cumulative-value entry per parameter.
        Each call advances the counter, recomputes the factor (capped
        at 1.0), and emits one ``updateParameter $tag $delta`` per
        target where ``$delta = target * factor - previous_cumulative``.
        """
        # Emit at indent 0 — procs are global definitions.
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(f"proc {name} {{}} {{")
        self._lines.append(f"    global {name}_state")
        # First-call initialization of state array.
        self._lines.append(f"    if {{![info exists {name}_state(count)]}} {{")
        self._lines.append(f"        set {name}_state(count) 0")
        for tag, _target in targets:
            self._lines.append(
                f"        set {name}_state(cum_{int(tag)}) 0.0"
            )
        self._lines.append("    }")
        # Advance counter, compute capped factor.
        self._lines.append(
            f"    set {name}_state(count) "
            f"[expr {{${name}_state(count) + 1}}]"
        )
        self._lines.append(
            f"    set _factor [expr "
            f"{{${name}_state(count) / {repr(float(n_steps_to_full))}}}]"
        )
        self._lines.append("    if {$_factor > 1.0} { set _factor 1.0 }")
        # One updateParameter per target — delta = current - previous.
        for tag, target in targets:
            self._lines.append(
                f"    set _cur [expr {{{repr(float(target))} * $_factor}}]"
            )
            self._lines.append(
                f"    set _delta "
                f"[expr {{$_cur - ${name}_state(cum_{int(tag)})}}]"
            )
            self._lines.append(f"    updateParameter {int(tag)} $_delta")
            self._lines.append(
                f"    set {name}_state(cum_{int(tag)}) $_cur"
            )
        self._lines.append("}")
        self._lines.indent = prev_indent

    # -- Partition emission scoping (ADR 0027, P4) --------------------------

    def partition_open(self, rank: int) -> None:
        """Open ``if {[getPID] == K} \\{ ... \\}`` block; indent body 4 spaces.

        On the **first** call across the emitter's lifetime, emit the
        one-shot runtime shim ``if {[info commands getPID] == ""} { proc
        getPID {} { return 0 } }`` so single-process OpenSees (no
        OpenSeesMP loaded) still parses and runs the deck — the
        fallback returns 0 so only the rank-0 block executes.

        The guard MUST probe ``info commands``, not ``info procs``:
        OpenSeesMP registers ``getPID`` via ``Tcl_CreateCommand``
        (a C command, invisible to ``info procs``), so an
        ``info procs`` guard overrides the native command and every
        MPI rank evaluates ``getPID`` as 0 — all ranks silently build
        rank 0's submodel (run-verified under ``mpiexec -n 8``).
        """
        if not self._partition_shim_emitted:
            # Emit at indent 0 (the shim is global, not per-rank).
            # In stream mode the active sink is the driver here, so the
            # shim lands in the driver — same as the list-mode slicing,
            # where it precedes the span header.
            prev_indent = self._lines.indent
            self._lines.indent = ""
            self._lines.append(
                "if {[info commands getPID] == \"\"} "
                "{ proc getPID {} { return 0 } }"
            )
            self._lines.indent = prev_indent
            self._partition_shim_emitted = True
        st = self._stream
        if st is not None and st.per_rank:
            # Live fragment routing (ADR 0065 Tier 2 Decision §3): the
            # block header is never emitted — the driver gets the
            # one-line source guard at partition_close instead. Naming
            # (``rank<K>_<seq>.tcl``, seq = the rank's 0-based block
            # counter) and the fragment banner match
            # ``_write_per_rank_tcl`` exactly.
            n = st.seq.get(rank, 0)
            st.seq[rank] = n + 1
            fname = f"rank{rank}_{n}.tcl"
            final = os.path.join(st.ranks_dir, fname)
            tmp = final + ".tmp"
            f = open(tmp, "w", encoding="utf-8")
            f.write(
                f"# apeGmsh per-rank fragment (ADR 0061): "
                f"rank {rank}, block {n}\n"
            )
            st.fragment_handle = f
            st.fragment_tmp = tmp
            st.fragment_final = final
            st.fragment_rank = rank
            st.fragment_name = fname
            # Body lines go to the fragment with the partition-level
            # indent stripped — reproducing _write_per_rank_tcl's
            # per-body-line ``removeprefix("    ")`` live (the
            # intra-block nesting indents survive; only the first
            # 4 spaces go).
            self._lines.switch_sink(
                f.write, strip_partition_indent=True,
            )
            self._lines.indent = self._lines.indent + "    "
            return
        # Header line is emitted at the **outer** indent (not nested
        # inside the partition's own indent).
        self._lines.append(
            "if {[getPID] == " + str(rank) + "} {"
        )
        if st is None:
            # Span recording is list-mode only — in stream mode the
            # buffer holds nothing to slice (spans retire; ADR 0065
            # Tier 2 Decision §3).
            self._open_partition = (rank, len(self._lines) - 1)
        # Indent subsequent emit calls one level deeper.
        self._lines.indent = self._lines.indent + "    "

    def partition_close(self) -> None:
        """Close the current per-rank block; emit trailing blank line."""
        # Drop one indent level.
        if self._lines.indent.endswith("    "):
            self._lines.indent = self._lines.indent[:-4]
        st = self._stream
        if (
            st is not None and st.per_rank
            and st.fragment_handle is not None
        ):
            # Close the live fragment; route the sink back to the
            # driver and write the source guard + trailing blank —
            # byte-identical to _write_per_rank_tcl's driver lines
            # (written raw: the driver replaces the whole block, no
            # indent applies).
            st.fragment_handle.close()
            assert st.fragment_tmp is not None
            assert st.fragment_final is not None
            st.replacements.append((st.fragment_tmp, st.fragment_final))
            rank, fname = st.fragment_rank, st.fragment_name
            st.fragment_handle = None
            st.fragment_tmp = None
            st.fragment_final = None
            st.fragment_rank = None
            st.fragment_name = None
            st.fragments_written += 1
            assert st.driver_handle is not None
            self._lines.switch_sink(
                st.driver_handle.write, strip_partition_indent=False,
            )
            st.driver_handle.write(
                f"if {{[getPID] == {rank}}} {{ source [file join "
                f"[file dirname [info script]] ranks {fname}] }}\n"
            )
            st.driver_handle.write("\n")
            return
        body_end = len(self._lines)
        # Closing brace and trailing blank line at the outer indent.
        self._lines.append("}")
        self._lines.append("")
        if self._open_partition is not None:
            rank_h, header = self._open_partition
            self._partition_spans.append(PartitionSpan(
                rank=rank_h, header=header, body_start=header + 1,
                body_end=body_end, end=len(self._lines),
            ))
            self._open_partition = None

    def partition_spans(self) -> list[PartitionSpan]:
        """Return the recorded per-rank block spans (ADR 0061), in
        emission order. Empty for unpartitioned decks. Fails loud in
        stream mode — spans retire there (live routing replaces the
        post-hoc slicing; ADR 0065 Tier 2 Decision §3)."""
        self._assert_not_streaming("partition_spans()")
        return list(self._partition_spans)

    # -- Partition runtime-conditional fallback (ADR 0027 INV-5) ------------

    def parallel_runtime_fallback_numberer(
        self, primary: str, fallback: str,
    ) -> None:
        """Emit ``if {[catch {numberer $primary} _err]} { numberer $fallback }``.

        ``ParallelPlain`` (the typical primary) only exists in
        OpenSeesMP; under single-process OpenSees the ``numberer``
        command errors and the Tcl ``catch`` swallows it so the
        fallback (typically ``RCM``) runs instead.  Restores
        shim-consistency with :meth:`partition_open`'s ``proc getPID``
        fallback (ADR 0027 INV-5 amendment 2026-05-23).
        """
        self._lines.append(
            f"# {primary} only exists in OpenSeesMP; "
            f"fall back to {fallback} under single-process OpenSees."
        )
        self._lines.append(
            f"if {{[catch {{numberer {primary}}} _err]}} "
            f"{{ numberer {fallback} }}"
        )

    def parallel_runtime_fallback_system(
        self, primary: str, fallback: str,
    ) -> None:
        """Emit ``if {[catch {system $primary} _err]} { system $fallback }``.

        Mirror of :meth:`parallel_runtime_fallback_numberer` — the
        primary (``Mumps``) requires OpenSeesMP + MPI; the fallback
        (``UmfPack``) lets the deck still parse and run under
        single-process OpenSees.
        """
        self._lines.append(
            f"# {primary} requires OpenSeesMP + MPI; "
            f"fall back to {fallback} under single-process OpenSees."
        )
        self._lines.append(
            f"if {{[catch {{system {primary}}} _err]}} "
            f"{{ system {fallback} }}"
        )
