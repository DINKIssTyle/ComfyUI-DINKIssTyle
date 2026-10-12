"""Reproduce the H3-only image LoRA subset at the same revision as Qwen."""
from pathlib import Path
import sys
from vendor_qwen_trainer import build

if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/minimax_trainer/vendor", "minimax")
