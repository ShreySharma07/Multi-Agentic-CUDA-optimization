from pathlib import Path


def readFile(file_name, directory):
    file_path = Path(directory) / file_name

    if not file_path.exists():
        print(f"file path {file_path} does not exists")
    with file_path.open('r') as f:
        content = f.read()
    
    return {"contents":content}