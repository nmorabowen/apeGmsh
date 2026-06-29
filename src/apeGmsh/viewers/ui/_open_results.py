"""File → Open Results… support for the post-solve viewer.

Splits into two layers:

* **Pure, Qt-free helpers** — :func:`sniff_results_format`,
  :func:`model_requirement`, :func:`build_results`. These inspect the
  file with raw ``h5py`` and construct the :class:`Results` via the
  existing ``from_native`` / ``from_ladruno`` / ``from_mpco`` loaders.
  They import no Qt and are headless-testable.
* **Qt driver** — :func:`run_open_dialog`. Wires a ``QFileDialog`` for
  the results file, asks the pure layer whether a model file is
  needed, pops the conditional second dialog, and hands the built
  :class:`Results` to ``on_loaded``.

The "is a model file needed?" rule follows directly from how the
three loaders source their broker (see :class:`Results`):

==============  ==================  ====================================
Format          Model follow-up     Why
==============  ==================  ====================================
native ``.h5``  none / required     embedded ``/opensees`` zone is the
                                    model; only a native file that lacks
                                    it needs a separate ``model.h5``.
``.ladruno``    optional            self-sufficient; ``model_h5=`` only
                                    enriches (orientation + lineage).
``.mpco``       required            broker is never embedded.
==============  ==================  ====================================
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Tuple

#: File-dialog name filter shared by the results picker.
RESULTS_NAME_FILTER = (
    "Results (*.h5 *.ladruno *.mpco);;"
    "apeGmsh native (*.h5);;"
    "Ladruno (*.ladruno);;"
    "STKO MPCO (*.mpco);;"
    "All files (*)"
)

_Format = str  # "native" | "ladruno" | "mpco" | "unknown"
_Requirement = str  # "none" | "optional" | "required"


def _generator_is_ladruno(f: Any) -> bool:
    """True when the open ``h5py.File`` carries ``INFO/GENERATOR == 'Ladruno'``.

    Mirrors the identity check in
    :class:`apeGmsh.results.readers._ladruno.LadrunoReader._validate_identity`.
    Tolerates the marker living on the root group as a fallback (some
    fork builds stamp it there).
    """
    def _decode(v: Any) -> str:
        # h5py string attrs may arrive as scalar bytes/str or as a
        # shape-(1,) numpy array (the Ladruno writer stores
        # GENERATOR as ``array([b'Ladruno'])``).
        import numpy as np

        if isinstance(v, np.ndarray):
            v = v.reshape(-1)[0] if v.size else b""
        return v.decode() if isinstance(v, bytes) else str(v)

    info = f.get("INFO")
    if info is not None and "GENERATOR" in info.attrs:
        return _decode(info.attrs["GENERATOR"]) == "Ladruno"
    if "GENERATOR" in f.attrs:
        return _decode(f.attrs["GENERATOR"]) == "Ladruno"
    return False


def sniff_results_format(path: "str | Path") -> _Format:
    """Classify a results file by inspecting its contents.

    Returns one of ``"native"`` (apeGmsh native HDF5 / Composed file),
    ``"ladruno"`` (the fork's canonical recorder), ``"mpco"`` (STKO),
    or ``"unknown"``.

    Content sniffing is preferred; the filename extension is only a
    fallback when the file cannot be opened as HDF5 (e.g. it does not
    exist yet). A missing/unreadable file never raises here.
    """
    p = Path(path)
    try:
        import h5py
    except ImportError:
        h5py = None  # type: ignore[assignment]

    if h5py is not None and p.is_file():
        try:
            with h5py.File(str(p), "r") as f:
                if _generator_is_ladruno(f):
                    return "ladruno"
                # STKO MPCO carries top-level ``MODEL_STAGE[<stamp>]``
                # groups (see readers/_mpco.py _STAGE_PREFIX).
                if any(k.startswith("MODEL_STAGE[") for k in f.keys()):
                    return "mpco"
                # apeGmsh native / Composed file: FEM zone under
                # ``/model`` and/or the bridge zone under ``/opensees``.
                if "model" in f or "opensees" in f:
                    return "native"
        except (OSError, KeyError):
            pass  # fall through to extension heuristics

    suffix = p.suffix.lower()
    if suffix == ".ladruno":
        return "ladruno"
    if suffix == ".mpco":
        return "mpco"
    if suffix == ".h5":
        return "native"
    return "unknown"


def _native_has_embedded_model(path: "str | Path") -> bool:
    """True when a native file carries the ``/opensees`` bridge zone."""
    try:
        import h5py
    except ImportError:
        return False
    try:
        with h5py.File(str(Path(path)), "r") as f:
            return "opensees" in f
    except (OSError, KeyError):
        return False


def model_requirement(path: "str | Path") -> Tuple[_Format, _Requirement]:
    """Return ``(format, requirement)`` for a results file.

    ``requirement`` is:

    * ``"none"`` — no model file needed (native with embedded
      ``/opensees``).
    * ``"optional"`` — a model file enriches but is not required
      (``.ladruno``).
    * ``"required"`` — a model file must be supplied (``.mpco``, or a
      native file with no embedded bridge zone).

    For ``"unknown"`` the requirement is ``"required"`` — the caller
    should refuse rather than guess.
    """
    fmt = sniff_results_format(path)
    if fmt == "ladruno":
        return fmt, "optional"
    if fmt == "mpco":
        return fmt, "required"
    if fmt == "native":
        return fmt, "none" if _native_has_embedded_model(path) else "required"
    return fmt, "required"


def build_results(
    results_path: "str | Path",
    model_path: "Optional[str | Path]" = None,
) -> Any:
    """Construct a :class:`Results` from resolved file paths.

    Dispatches on the sniffed format. ``model_path`` may be ``None``
    when the format does not need it (native with embedded model, or a
    standalone ``.ladruno``). Qt-free — the dialog driver resolves the
    paths first, then calls this.
    """
    from apeGmsh.results.Results import Results

    fmt = sniff_results_format(results_path)
    if fmt == "native":
        # Import from the package top-level, not the submodule: viewers/
        # may only reach OpenSeesModel via the allowed surface (see
        # tests/test_viewers_pure_h5_consumer.py).
        from apeGmsh.opensees import OpenSeesModel
        # The embedded /opensees zone lives in the results file itself
        # (Composed-file pattern); a separate model_path overrides it.
        model_src = model_path if model_path is not None else results_path
        model = OpenSeesModel.from_h5(model_src)
        return Results.from_native(results_path, model=model)
    if fmt == "ladruno":
        return Results.from_ladruno(results_path, model_h5=model_path)
    if fmt == "mpco":
        if model_path is None:
            raise ValueError(
                "A .mpco file needs a model.h5 — pass model_path."
            )
        return Results.from_mpco(results_path, model_h5=model_path)
    raise ValueError(
        f"{Path(results_path)} is not a recognised results file "
        "(expected apeGmsh native .h5, .ladruno, or .mpco)."
    )


def run_open_dialog(
    parent: Any,
    on_loaded: "Callable[[Any], None]",
) -> None:
    """Drive the Open-Results flow against Qt dialogs.

    Picks a results file, asks :func:`model_requirement` whether a
    model file is needed, pops the conditional second picker, builds
    the :class:`Results`, and calls ``on_loaded(results)``. Any cancel
    aborts quietly; build/format errors surface as a message box.
    """
    from qtpy import QtWidgets

    results_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent, "Open results", "", RESULTS_NAME_FILTER,
    )
    if not results_path:
        return

    fmt, requirement = model_requirement(results_path)
    if fmt == "unknown":
        QtWidgets.QMessageBox.warning(
            parent, "Open results",
            f"{Path(results_path).name} is not a recognised results "
            "file (expected apeGmsh native .h5, .ladruno, or .mpco).",
        )
        return

    model_path: Optional[str] = None
    if requirement == "required":
        model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent, f"Select model.h5 for this {fmt} file", "",
            "Model (*.h5);;All files (*)",
        )
        if not model_path:
            QtWidgets.QMessageBox.warning(
                parent, "Open results",
                f"A {fmt} file needs a model.h5 to supply the model "
                "and FEMData — open cancelled.",
            )
            return
    elif requirement == "optional":
        choice = QtWidgets.QMessageBox.question(
            parent, "Model file",
            "Add a model.h5 for richer detail (beam orientation, full "
            "bridge records, lineage)?\n\n"
            "Choose 'No' to open the file standalone.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if choice == QtWidgets.QMessageBox.StandardButton.Yes:
            picked, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent, "Select model.h5", "",
                "Model (*.h5);;All files (*)",
            )
            # A cancel here is fine — fall back to the standalone broker.
            model_path = picked or None

    try:
        results = build_results(results_path, model_path)
    except Exception as exc:  # noqa: BLE001 — surface any load failure
        QtWidgets.QMessageBox.critical(
            parent, "Open results",
            f"Could not open {Path(results_path).name}:\n\n{exc}",
        )
        return

    on_loaded(results)


__all__ = [
    "RESULTS_NAME_FILTER",
    "sniff_results_format",
    "model_requirement",
    "build_results",
    "run_open_dialog",
]
