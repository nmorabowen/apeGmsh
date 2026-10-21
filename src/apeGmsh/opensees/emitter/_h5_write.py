"""Low-level HDF5 write helpers for :mod:`apeGmsh.opensees.emitter.h5`."""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# The attribute size limit (opensees 2.34.0)
# ---------------------------------------------------------------------------

#: The largest array payload, in bytes, written as an HDF5 attribute.
#: An attribute is one object-header message, and a message is capped at
#: 65,535 bytes: about 8,183 float64 or int64 slots, or 4,091
#: variable-length string slots (16-byte heap ids), measured on h5py 3.15
#: and HDF5 1.14.6. The margin under the cap covers the attribute's
#: name, datatype and dataspace. Every array that fitted before keeps the
#: attribute form, so its bytes and ``model_hash`` do not change.
ATTR_PAYLOAD_LIMIT: int = 64_000

#: Header bytes per slot: a numeric slot is stored inline, and a
#: variable-length string slot is stored as a global-heap id.
_NUMERIC_SLOT_BYTES: int = 8
_STRING_SLOT_BYTES: int = 16


def _payload_bytes(n_slots: int, *, has_str: bool) -> int:
    """Header bytes an ``n_slots`` attribute array takes."""
    per = _STRING_SLOT_BYTES if has_str else _NUMERIC_SLOT_BYTES
    return n_slots * per


# ---------------------------------------------------------------------------
# Low-level write helpers
# ---------------------------------------------------------------------------

def _arg_token(value: Any) -> str:
    """Render one element of a mixed flag/value tuple as a token.

    ``str(int)`` / ``repr(float)`` so ``compose._int_recover`` reads the
    number back at its original type — an ``int`` stays an ``int``, which
    is what ``OPS_GetIntInput`` demands of e.g. ``-matrixType``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _set_attr(target: Any, key: str, value: Any) -> None:
    """Write ``value`` as an HDF5 attribute on ``target``.

    HDF5 stores attributes as native scalars or arrays — never as JSON
    blobs (per the schema's "structured groups, scalar attrs, array
    datasets" rule). This helper coerces Python values to the closest
    h5py-friendly representation:

    * ``str`` → variable-length UTF-8 string
    * ``bool`` / ``int`` → int64
    * ``float`` → float64
    * ``None`` → empty-string attr (h5py rejects ``None``)
    * tuple/list of numbers → 1-D float64 / int64 array
    * tuple/list of strings → 1-D variable-length string array
    * tuple/list mixing strings and numbers → 1-D variable-length string
      array of tokens (see :func:`_arg_token`)

    A tuple/list whose payload exceeds :data:`ATTR_PAYLOAD_LIMIT` raises
    :class:`ValueError` naming ``key`` and the group: HDF5 cannot hold it
    as an attribute, and only the ``params`` pair
    (:func:`_write_param_array`) and a recorder's ``ids``
    (:func:`_set_attr_or_dataset`) have a dataset form a reader knows.
    """
    import h5py
    import numpy as np

    if isinstance(value, (tuple, list)):
        has_str = any(isinstance(v, str) for v in value)
        size = _payload_bytes(len(value), has_str=has_str)
        if size > ATTR_PAYLOAD_LIMIT:
            raise ValueError(
                f"H5Emitter: attribute {key!r} on {target.name} holds "
                f"{len(value)} values ({size} bytes), past the "
                f"{ATTR_PAYLOAD_LIMIT}-byte HDF5 attribute limit, and it "
                "has no dataset form (opensees 2.34.0 gives one only to a "
                "params pair and a recorder's ids)."
            )
    if value is None:
        target.attrs[key] = ""
        return
    if isinstance(value, bool):
        target.attrs[key] = np.int64(int(value))
        return
    if isinstance(value, int):
        target.attrs[key] = np.int64(value)
        return
    if isinstance(value, float):
        target.attrs[key] = np.float64(value)
        return
    if isinstance(value, str):
        target.attrs.create(key, value, dtype=h5py.string_dtype(encoding="utf-8"))
        return
    if isinstance(value, (tuple, list)):
        if not value:
            target.attrs.create(
                key, np.array([], dtype=np.float64),
            )
            return
        if all(isinstance(v, str) for v in value):
            target.attrs.create(
                key, list(value),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            return
        if all(isinstance(v, bool) or isinstance(v, int) for v in value):
            target.attrs[key] = np.asarray(value, dtype=np.int64)
            return
        if any(isinstance(v, str) for v in value):
            # Flag-and-value shape: ``("-matrixType", 2)`` from
            # ``system Pardiso``, ``(0.5, 0.25, "-form", "D")`` from
            # ``integrator Newmark``.  No numeric dtype can hold the flag
            # token, so store the whole tuple as tokens; the replay side
            # recovers the numerics (``compose._int_recover``).
            target.attrs.create(
                key, [_arg_token(v) for v in value],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            return
        # Mixed numeric — coerce to float64.
        target.attrs[key] = np.asarray(
            [float(v) for v in value], dtype=np.float64,
        )
        return
    # Fallback: stringify so we never crash on an unexpected attr type.
    target.attrs.create(
        key, repr(value), dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _scan_flag(
    args: tuple[int | float | str, ...], flag: str,
) -> str | None:
    """Return the argument immediately after ``flag`` (as a string), or
    ``None`` if the flag is not present.

    Used for recorder ``-file`` extraction. Treats only string args as
    flag candidates.
    """
    for i, v in enumerate(args[:-1]):
        if isinstance(v, str) and v == flag:
            nxt = args[i + 1]
            return str(nxt)
    return None


def _write_param_array(
    target: Any, key: str, params: tuple[float | str | int, ...],
) -> None:
    """Write a positional ``*args`` tuple as one or two attributes.

    OpenSees parameter lists are positional: ``Steel02 1 fy E b R0 cR1
    cR2`` is all numeric, but ``forceBeamColumn 1 i j tt it -mass m
    -iter mx tol`` interleaves numerics and flag-string tokens. To
    stay HDF5-native (no JSON blobs), we split into two parallel
    arrays:

    * ``{key}`` — float64 array, ``NaN`` in slots that hold a string.
    * ``{key}_str`` — UTF-8 vlen string array, empty string in slots
      that hold a numeric value. Written ONLY if at least one slot is
      a string (pure-numeric param lists skip ``{key}_str`` entirely).

    Slot ``i`` of the original ``*args`` is reconstructible by reading
    whichever of the two arrays has a non-sentinel value.

    A pair whose larger array exceeds :data:`ATTR_PAYLOAD_LIMIT` (opensees
    2.34.0, #2404) is written as two uncompressed datasets of the same
    names in ``target`` instead, like an element's ``args`` / ``args_str``.
    The pair switches together, so a group holds one form or the other.
    """
    import h5py
    import numpy as np

    if not params:
        target.attrs.create(key, np.array([], dtype=np.float64))
        return
    has_str = any(isinstance(v, str) for v in params)
    nums = np.empty(len(params), dtype=np.float64)
    strs: list[str] = []
    for i, v in enumerate(params):
        if isinstance(v, str):
            nums[i] = float("nan")
            strs.append(v)
        else:
            nums[i] = float(v)
            strs.append("")
    str_dtype = h5py.string_dtype(encoding="utf-8")
    if _payload_bytes(len(params), has_str=has_str) > ATTR_PAYLOAD_LIMIT:
        target.create_dataset(key, data=nums)
        if has_str:
            target.create_dataset(
                f"{key}_str", data=np.asarray(strs, dtype=object),
                dtype=str_dtype,
            )
        return
    target.attrs.create(key, nums)
    if has_str:
        target.attrs.create(f"{key}_str", strs, dtype=str_dtype)


def _set_attr_or_dataset(
    target: Any, key: str, values: tuple[int, ...] | list[int],
) -> None:
    """Write an integer sequence as an int64 attribute, or, past
    :data:`ATTR_PAYLOAD_LIMIT`, as an uncompressed int64 dataset of the
    same name (opensees 2.34.0, #2404). A declared recorder's ``ids``
    goes through here."""
    import numpy as np

    if _payload_bytes(len(values), has_str=False) > ATTR_PAYLOAD_LIMIT:
        target.create_dataset(key, data=np.asarray(values, dtype=np.int64))
        return
    _set_attr(target, key, values)
