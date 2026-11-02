"""The ``<stem>.user.h5`` file: combinations and user results (ADR 0130).

Writer, reader, the ``rf1`` fingerprint and the staleness table for ``/meta``,
``/combinations`` and ``/results``. Nothing here evaluates a combination: that
is slice VU2. The layout is in ``architecture/h5-schema.md``.
"""
from __future__ import annotations

from ._descriptor import (
    Combination,
    CombinationTable,
    Refusal,
    StateAddress,
    Term,
)
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
from ._writer import model_hashes, write_user_file

__all__ = [
    "Combination",
    "CombinationTable",
    "ElementBucket",
    "FingerprintRefused",
    "Refusal",
    "ResultEntry",
    "ResultIndex",
    "StateAddress",
    "Staleness",
    "Term",
    "UserFile",
    "UserFileError",
    "UserFileRefused",
    "UserResult",
    "UserWriteRefused",
    "ZoneRefused",
    "ZoneState",
    "model_hashes",
    "results_fingerprint",
    "staleness",
    "write_user_file",
]
