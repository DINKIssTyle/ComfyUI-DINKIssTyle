"""Validate the worker's pinned requirements without importing GPU packages."""
from importlib import metadata
from pathlib import Path


def requirement_lines(path):
    path = Path(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-r "):
            yield from requirement_lines(path.parent / line[3:].strip())
        elif line and not line.startswith("--"):
            yield line


def check_requirements(path):
    try:
        from packaging.requirements import Requirement
    except ImportError as error:
        raise ValueError("Training dependency packaging is missing. Use Prepare Runtime or setup_runtime.py.") from error
    failures = []
    for line in requirement_lines(path):
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            failures.append(f"{requirement.name}: missing ({requirement.specifier or 'required'})")
        else:
            if not requirement.specifier.contains(version, prereleases=True):
                failures.append(f"{requirement.name}: {version}, requires {requirement.specifier}")
    if failures:
        raise ValueError("Training dependencies are not ready: " + "; ".join(failures) +
                         ". Use Prepare Runtime or setup_runtime.py to prepare the isolated environment.")
