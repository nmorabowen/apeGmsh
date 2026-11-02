"""The ``<stem>.user.h5`` file: combinations, user results, modes, spectra (ADR 0130).

Writer, reader, the ``rf1`` fingerprint and the staleness table for ``/meta``,
``/combinations``, ``/results``, ``/modes`` and ``/spectra``. :func:`mode_table`
turns a ``modalProperties`` dict into the ``/modes`` table, and
:func:`spectrum_cases` an ``apeSees.spectrum_after_gravity`` result into the
``/spectra`` cases. The combination evaluator (slice VU2) is reached through
``Results.combinations()``, which reads spectrum terms through
:class:`UserSpectra` (slice VU2b). The layout is in ``architecture/h5-schema.md``.
"""
from __future__ import annotations

from ._descriptor import (
    Combination,
    CombinationTable,
    Refusal,
    StateAddress,
    Term,
)
from ._evaluate import Case, Combined, CombinationRefused, Field, evaluate, is_signed
from ._errors import (
    FingerprintRefused,
    UserFileError,
    UserFileRefused,
    UserWriteRefused,
    ZoneRefused,
)
from ._fields import ElementBucket, ResultEntry, UserResult
from ._fingerprint import results_fingerprint
from ._modes import ModeTable, mode_table
from ._reader import ResultIndex, UserFile
from ._spectra import CaseQuantity, SpectraIndex, SpectrumCase, spectrum_cases
from ._spectra_source import UserSpectra
from ._staleness import Staleness, ZoneState, staleness
from ._states import CombinationState, UserCombinations
from ._writer import model_hashes, write_user_file

__all__ = [
    "Case",
    "CaseQuantity",
    "Combination",
    "CombinationRefused",
    "CombinationState",
    "CombinationTable",
    "Combined",
    "ElementBucket",
    "Field",
    "FingerprintRefused",
    "ModeTable",
    "Refusal",
    "ResultEntry",
    "ResultIndex",
    "SpectraIndex",
    "SpectrumCase",
    "StateAddress",
    "Staleness",
    "Term",
    "UserFile",
    "UserFileError",
    "UserCombinations",
    "UserFileRefused",
    "UserResult",
    "UserSpectra",
    "UserWriteRefused",
    "ZoneRefused",
    "ZoneState",
    "evaluate",
    "is_signed",
    "mode_table",
    "model_hashes",
    "results_fingerprint",
    "spectrum_cases",
    "staleness",
    "write_user_file",
]
