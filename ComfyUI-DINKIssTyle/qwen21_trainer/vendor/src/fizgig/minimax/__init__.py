"""MiniMax H3 — barebones native LoRA/FT training port (image-only, no audio/samples).

Third model family, alongside Klein 9B and Krea 2. The DiT is MiniMax's ~33B omni
audio+video transformer; here we train it on still images (single video frame, audio
stream skipped) — the very core of a LoRA/rotating-FT trainer, no inference or sampling.

The model class is a pure-PyTorch port of ComfyUI's reference
(comfy/ldm/minimax/model.py), weight-name-compatible with the official bf16 checkpoint
(minimax_h3_fl2va_bf16.safetensors) so a trained LoRA/FT loads cleanly.
"""
