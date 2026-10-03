"""Regenerate the golden emit corpus: ``python -m tests.opensees.golden.regen``.

Run from the repository root.  It rewrites ``MANIFEST.json`` and every
golden under ``cells/``, deletes goldens no cell owns any more, and prints
one ``changed`` / ``unchanged`` / ``new`` / ``removed`` line per file plus
a count.  ``--exact`` also rewrites a deck that differs from its golden only
within the float tolerance: use it when ``src`` deliberately changes the
bits of an emitted float (the default keeps another host's last-ulp libm
difference from rewriting the corpus).  A regen is a maintainer-visible act: a PR that changes goldens
lists the changed cells in its body (see ``README.md``).  No test and no
workflow calls this module; ``test_golden_corpus.py`` asserts the latter.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Emit from THIS checkout's source even when the editable install points at
# another one (AGENTS.md "Build and test").
_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tests.opensees.golden import builder  # noqa: E402


def _write_if_changed(
    path: Path, text: str, *, deck: bool = False, exact: bool = False,
) -> str:
    """Write ``text`` unless the committed file already holds it.

    A deck that matches its golden within the float tolerance the test
    uses (``builder.first_deck_mismatch``) is left as committed, so a
    last-ulp libm difference on another platform never rewrites it,
    unless ``exact`` asks for the bytes.
    """
    data = text.encode("utf-8")
    if path.exists():
        old = path.read_bytes()
        if old == data:
            return "unchanged"
        if deck and not exact and builder.first_deck_mismatch(
            old.decode("utf-8").replace("\r\n", "\n"), text,
        ) is None:
            return "unchanged"
        status = "changed"
    else:
        status = "new"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return status


def regenerate(*, exact: bool = False) -> dict[str, str]:
    """Rewrite the corpus; return ``{cell id or file: status}``.

    A golden cell is ``changed`` when its deck or its h5 dump changed
    (the status names which), ``new`` when it had no golden yet, else
    ``unchanged``.  ``MANIFEST.json`` and removed stale files report as
    their own entries.
    """
    report: dict[str, str] = {}
    owned: set[Path] = {builder.MANIFEST_PATH}
    report["MANIFEST.json"] = _write_if_changed(
        builder.MANIFEST_PATH, builder.manifest_text(),
    )
    with tempfile.TemporaryDirectory(prefix="apegmsh_golden_") as tmp:
        scratch = Path(tmp)
        for n, (f, m, o) in enumerate(builder.all_cells()):
            if builder.applicability(f, m, o) is not None:
                continue
            deck_path, dump_path = builder.golden_paths(f, m, o)
            owned.update((deck_path, dump_path))
            parts = {
                "deck": _write_if_changed(
                    deck_path,
                    builder.render_deck(f, m, o, scratch / f"d{n}"),
                    deck=True, exact=exact,
                ),
                "h5dump": _write_if_changed(
                    dump_path,
                    builder.render_h5_dump(f, m, o, scratch / f"h{n}"),
                ),
            }
            if all(v == "unchanged" for v in parts.values()):
                status = "unchanged"
            elif all(v == "new" for v in parts.values()):
                status = "new"
            else:
                status = "changed (" + ", ".join(
                    k for k, v in parts.items() if v != "unchanged"
                ) + ")"
            report[builder.cell_id(f, m, o)] = status

    if builder.CELLS_DIR.exists():
        for stale in sorted(builder.CELLS_DIR.rglob("*")):
            if stale.is_file() and stale not in owned:
                stale.unlink()
                rel = stale.relative_to(builder.GOLDEN_DIR).as_posix()
                report[rel] = "removed"
    return report


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if any(a != "--exact" for a in args):
        print("usage: python -m tests.opensees.golden.regen [--exact]")
        return 2
    report = regenerate(exact="--exact" in args)
    for key, status in report.items():
        print(f"{status:24s} {key}")
    n_changed = sum(1 for s in report.values() if s != "unchanged")
    print(
        f"golden corpus: {n_changed} changed, "
        f"{len(report) - n_changed} unchanged "
        f"(of {len(report)} entries: MANIFEST.json + golden cells"
        " + removed files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
