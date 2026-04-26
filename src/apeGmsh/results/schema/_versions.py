"""Version constants for the results module on-disk formats.

Bump ``SCHEMA_VERSION`` minor (``"1.x"``) for additive changes,
major (``"2.0"``) for breaking changes. Bump ``PARSER_VERSION``
when recorder transcoder logic changes in a way that invalidates
cached transcoded HDF5 files.
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0"
PARSER_VERSION = "1.0"
