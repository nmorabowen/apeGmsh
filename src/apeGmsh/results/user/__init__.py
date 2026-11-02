"""The ``<stem>.user.h5`` file: combinations and user results (ADR 0130).

Writer, reader, the ``rf1`` fingerprint and the staleness table for ``/meta``,
``/combinations`` and ``/results``, and the combination evaluator (slice VU2),
which ``Results.combinations()`` reaches. The layout is in
``architecture/h5-schema.md``.
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
from ._reader import ResultIndex, UserFile
from ._staleness import Staleness, ZoneState, staleness
from ._states import CombinationState, UserCombinations
from ._writer import model_hashes, write_user_file

__all__ = [
    "Case",
    "Combination",
    "CombinationRefused",
    "CombinationState",
    "CombinationTable",
    "Combined",
    "ElementBucket",
    "Field",
    "FingerprintRefused",
    "Refusal",
    "ResultEntry",
    "ResultIndex",
    "StateAddress",
    "Staleness",
    "Term",
    "UserFile",
    "UserFileError",
    "UserCombinations",
    "UserFileRefused",
    "UserResult",
    "UserWriteRefused",
    "ZoneRefused",
    "ZoneState",
    "evaluate",
    "is_signed",
    "model_hashes",
    "results_fingerprint",
    "staleness",
    "write_user_file",
]
