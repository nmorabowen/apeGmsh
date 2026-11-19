"""The ``<stem>.user.h5`` file: combinations, user results, modes, spectra, entities.

ADR 0130. Writer, reader, the ``rf1`` fingerprint and the staleness table for
``/meta``, ``/combinations``, ``/results``, ``/modes``, ``/spectra`` and
``/entities``. :func:`mode_table` turns a ``modalProperties`` dict into the
``/modes`` table, :func:`spectrum_cases` an ``apeSees.spectrum_after_gravity``
result into the ``/spectra`` cases, and ``SectionCutDef.to_entity`` a cut into
a :class:`CutEntity`. The combination evaluator (slice VU2) is reached through
``Results.combinations()``, which reads spectrum terms through
:class:`UserSpectra` (slice VU2b). The layout is in ``architecture/h5-schema.md``.

ADR 0132 adds two files beside the set: :class:`WorkspaceFile` and
:func:`write_workspace` read and write ``<stem>.workspace.h5`` (named sets with
their verdicts, plot definitions, saved views), and :func:`read_variants` and
:func:`write_variants` read and write ``<study>.variants.json``.
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
from ._variants import (
    Variant,
    VariantDef,
    VariantsManifest,
    VariantsRefused,
    VariantsWriteRefused,
    read_variants,
    write_variants,
)
from ._workspace import (
    PlotDef,
    RowRefusal,
    SetDef,
    SetVerdict,
    SetVerdicts,
    ViewDef,
    WorkspaceFile,
    WorkspaceRefused,
    WorkspaceRenamedAside,
    WorkspaceRow,
    WorkspaceSet,
    WorkspaceWriteRefused,
    write_workspace,
)
from ._writer import model_hashes, write_user_file

__all__ = [
    "Case",
    "CaseQuantity",
    "Combination",
    "CombinationRefused",
    "CombinationState",
    "CombinationTable",
    "Combined",
    "CutEntity",
    "ElementBucket",
    "Entities",
    "Field",
    "FingerprintRefused",
    "ModeTable",
    "PlotDef",
    "PointSet",
    "Refusal",
    "ResultEntry",
    "ResultIndex",
    "RowRefusal",
    "SetDef",
    "SetVerdict",
    "SetVerdicts",
    "SpectraIndex",
    "SpectrumCase",
    "StateAddress",
    "Staleness",
    "Storey",
    "Term",
    "UserFile",
    "UserFileError",
    "UserCombinations",
    "UserFileRefused",
    "UserResult",
    "UserSpectra",
    "UserWriteRefused",
    "Variant",
    "VariantDef",
    "VariantsManifest",
    "VariantsRefused",
    "VariantsWriteRefused",
    "ViewDef",
    "WorkspaceFile",
    "WorkspaceRefused",
    "WorkspaceRenamedAside",
    "WorkspaceRow",
    "WorkspaceSet",
    "WorkspaceWriteRefused",
    "ZoneRefused",
    "ZoneState",
    "evaluate",
    "is_signed",
    "mode_table",
    "model_hashes",
    "read_variants",
    "results_fingerprint",
    "spectrum_cases",
    "staleness",
    "write_user_file",
    "write_variants",
    "write_workspace",
]
