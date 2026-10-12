"""DKST checkpoint layout adapter for the image-only Qwen 2.1 VAE.

Key layouts checked against Comfy-Org/ComfyUI comfy/ldm/wan/vae2_2.py
and Fizgig's vendored Diffusers-style Qwen 2.1 VAE. No weights are discarded.
"""
import re


def _native_key(key, num_res_blocks):
    key = re.sub(r"^conv1\.", "quant_conv.", key)
    key = re.sub(r"^conv2\.", "post_quant_conv.", key)
    key = re.sub(r"^(encoder|decoder)\.conv1\.", r"\1.conv_in.", key)
    key = re.sub(r"^(encoder|decoder)\.head\.0\.", r"\1.norm_out.", key)
    key = re.sub(r"^(encoder|decoder)\.head\.2\.", r"\1.conv_out.", key)
    key = re.sub(r"\.middle\.0\.", ".mid_block.resnets.0.", key)
    key = re.sub(r"\.middle\.1\.", ".mid_block.attentions.0.", key)
    key = re.sub(r"\.middle\.2\.", ".mid_block.resnets.1.", key)
    match = re.match(r"^(encoder|decoder)\.(downsamples|upsamples)\.(\d+)\.\2\.(\d+)\.(.+)$", key)
    if match:
        side, _, block, layer, suffix = match.groups()
        count = num_res_blocks + (side == "decoder")
        layer = int(layer)
        if layer > count:
            raise ValueError(f"Unexpected Qwen 2.1 VAE layer: {key}")
        group = "down_blocks" if side == "encoder" else "up_blocks"
        member = f"resnets.{layer}" if layer < count else ("downsampler" if side == "encoder" else "upsampler")
        key = f"{side}.{group}.{block}.{member}.{suffix}"
    for index, name in ((0, "norm1"), (2, "conv1"), (3, "norm2"), (6, "conv2")):
        key = key.replace(f".residual.{index}.", f".{name}.")
    return key.replace(".shortcut.", ".conv_shortcut.")


def normalize_vae_weights(state, expected, num_res_blocks=2):
    """Return a complete, shape-checked state dict; never use partial loading."""
    native = "encoder.conv1.weight" in state
    converted = {}
    for source, value in state.items():
        target = _native_key(source, num_res_blocks) if native else source
        if target in converted:
            raise ValueError(f"Duplicate Qwen 2.1 VAE weight: {target}")
        if target in expected and value.shape != expected[target].shape:
            # Qwen 2.1 has a single temporal kernel; remove only that axis.
            if value.ndim == 5 and expected[target].ndim == 4 and value.shape[2] == 1:
                value = value.squeeze(2)
        converted[target] = value
    missing = sorted(expected.keys() - converted.keys())
    extra = sorted(converted.keys() - expected.keys())
    wrong = [f"{key}: {tuple(value.shape)} -> {tuple(expected[key].shape)}"
             for key, value in converted.items() if key in expected and value.shape != expected[key].shape]
    if missing or extra or wrong:
        details = []
        for label, values in (("missing", missing), ("unexpected", extra), ("shape mismatch", wrong)):
            if values:
                details.append(f"{label} ({len(values)}): " + "; ".join(values[:3]))
        raise ValueError("Incompatible Qwen Image 2.1 VAE checkpoint. Select its RGBA/64-channel image VAE "
                         "in ComfyUI or Diffusers format. " + " | ".join(details))
    return converted, "ComfyUI" if native else "Diffusers"
