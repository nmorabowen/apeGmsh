"""The ``<stem>.user.h5`` file: combinations, user results, modes, spectra, entities.

ADR 0130. Writer, reader, the ``rf1`` fingerprint and the staleness table for
``/meta``, ``/combinations``, ``/results``, ``/modes``, ``/spectra`` and
``/entities``. :func:`mode_table` turns a ``modalProperties`` dict into the
``/modes`` table, :func:`spectrum_cases` an ``apeSees.spectrum_after_gravity``
result into the ``/spectra`` cases, and ``SectionCutDef.to_entity`` a cut into
a :class:`CutEntity`. Nothing here evaluates a combination: that is slice VU2.
The layout is in ``architecture/h5-schema.md``.
"""
from __future__ import annotations

from ._descriptor import (
    Combination,
    CombinationTable,
    Refusal,
    StateAddress,
    Term,
)
from ._entities import CutEntity, Entities, PointSet, Storey
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
from ._staleness import Staleness, ZoneState, staleness
from ._writer import model_hashes, write_user_file

__all__ = [
    "CaseQuantity",
    "Combination",
    "CombinationTable",
    "CutEntity",
    "ElementBucket",
    "Entities",
    "FingerprintRefused",
    "ModeTable",
    "PointSet",
    "Refusal",
    "ResultEntry",
    "ResultIndex",
    "SpectraIndex",
    "SpectrumCase",
    "StateAddress",
    "Staleness",
    "Storey",
    "Term",
    "UserFile",
    "UserFileError",
    "UserFileRefused",
    "UserResult",
    "UserWriteRefused",
    "ZoneRefused",
    "ZoneState",
    "mode_table",
    "model_hashes",
    "results_fingerprint",
    "spectrum_cases",
    "staleness",
    "write_user_file",
]
