"""Explicitly create the separate trainer venv. No install runs at plugin import."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Install DKST Qwen 2.1 training environment (Python 3.12 recommended)")
    parser.add_argument("--venv", type=Path, default=Path(__file__).resolve().parent / ".venv")
    parser.add_argument("--optional-kernels", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required; Python 3.12 is recommended.")
    args.venv = args.venv.resolve()
    subprocess.run([sys.executable, "-m", "venv", str(args.venv)], check=True)
    python = args.venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    root = Path(__file__).resolve().parent
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(root / "requirements.txt")], check=True)
    if args.optional_kernels:
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(root / "requirements-optional.txt")], check=True)
    print(f"Training Python: {python}")
    subprocess.run([str(python), str(root / "runner.py"), "--check"], check=True)


if __name__ == "__main__":
    main()
