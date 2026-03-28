# tools/readfiles.py
from pathlib import Path

def readFile(path: str) -> str:
    """Reads and returns the content of a file."""
    p = Path(path)
    if not p.exists():
        return f"Error: file {path} not found"
    return p.read_text()