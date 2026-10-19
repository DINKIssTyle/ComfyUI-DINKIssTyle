"""ComfyUI Manager and manual installation entry point."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('bootstrap.py')), run_name='__main__')
