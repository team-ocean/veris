"""Type the vendored Versioneer entry points used by the package build."""

from setuptools import Command

def get_version() -> str: ...
def get_cmdclass(
    cmdclass: dict[str, type[Command]] | None = None,
) -> dict[str, type[Command]]: ...
