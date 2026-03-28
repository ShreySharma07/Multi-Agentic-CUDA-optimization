from pathlib import Path

def getFiles(path: str) -> list[str]:
    """Returns list of .cu files in the given directory."""
    p = Path(path)
    if not p.exists():
        return []
    return [str(f) for f in p.glob("*.cu")]