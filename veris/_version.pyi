"""Public result of the generated version provider; implementation is untouched."""

from typing import TypedDict

VersionInfo = TypedDict(
    "VersionInfo",
    {
        "version": str,
        "full-revisionid": str | None,
        "dirty": bool | None,
        "error": str | None,
        "date": str | None,
    },
)

def get_versions() -> VersionInfo: ...
