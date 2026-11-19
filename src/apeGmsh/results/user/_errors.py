"""What the ``<stem>.user.h5`` package raises (ADR 0130, Amendment 1)."""
from __future__ import annotations

__all__ = [
    "FingerprintRefused",
    "UserFileError",
    "UserFileRefused",
    "UserWriteRefused",
    "ZoneRefused",
]


class UserFileError(ValueError):
    """Base of every refusal in this package."""


class UserFileRefused(UserFileError):
    """The file cannot be opened at all: its ``/meta`` is missing or malformed."""


class ZoneRefused(UserFileError):
    """One zone is malformed (A3). The file still opens; the other zones read."""

    def __init__(self, zone: str, detail: str) -> None:
        super().__init__(f"{zone}: {detail}")
        self.zone = zone
        self.detail = detail


class UserWriteRefused(UserFileError):
    """A write refused before replacing anything; the previous file is whole."""


class FingerprintRefused(UserFileError):
    """No ``results_hash`` can be computed for this file."""
