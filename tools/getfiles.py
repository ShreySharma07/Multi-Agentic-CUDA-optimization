from pathlib import Path

def getFiles(directory_path, extension):
    try:
        directory = Path(directory_path)
        if not directory.exists():
            return {"error": "Directory does not exist"}
        
        files = [f.name for f in directory.glob(f"*{extension}")]

        return {'files':files}
    except Exception as e:
        print(e)