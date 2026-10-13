"""Manager hook for installations that clone the whole repository."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "ComfyUI-DINKIssTyle/bootstrap.py"), run_name="__main__")
