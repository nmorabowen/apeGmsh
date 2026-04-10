from __future__ import annotations


class MissingOptionalDependency:
    """Proxy object that raises a clear ImportError when used."""

    def __init__(
        self,
        feature: str,
        package: str,
        *,
        extra: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self._feature = feature
        self._package = package
        self._extra = extra
        self._cause = cause

    def _message(self) -> str:
        msg = (
            f"{self._feature} requires the optional dependency "
            f"'{self._package}'."
        )
        if self._extra:
            msg += f" Install it with `pip install apeGmsh[{self._extra}]`."
        else:
            msg += f" Install `{self._package}` to enable this feature."
        return msg

    def _raise(self) -> None:
        raise ImportError(self._message()) from self._cause

    def __getattr__(self, name: str):
        self._raise()

    def __call__(self, *args, **kwargs):
        self._raise()

    def __repr__(self) -> str:
        return (
            f"<MissingOptionalDependency feature={self._feature!r} "
            f"package={self._package!r}>"
        )
