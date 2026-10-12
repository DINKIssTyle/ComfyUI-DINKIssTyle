"""DKST decoder for ComfyUI INT8 Qwen3-VL text encoder checkpoints.

Restore the scale and optional ConvRot basis before assigning dense HF weights.
Keep metadata/codes in their original dtype until decoding. Uses Fizgig's existing
ConvRot decoder; no optional CUDA kernels or ComfyUI imports are required.
"""
import json

import torch


def decode_text_weights(state, dtype=torch.bfloat16):
    """Consume a raw CPU state dict, returning dense weights and decoded count."""
    from fizgig.minimax.convrot import dequantize_int8_convrot

    count = 0
    for marker in [key for key in state if key.endswith(".comfy_quant")]:
        base = marker.removesuffix(".comfy_quant")
        try:
            blob = state[marker]
            if blob.dtype != torch.uint8 or blob.ndim != 1:
                raise ValueError("comfy_quant must contain uint8 JSON bytes")
            conf = json.loads(bytes(blob.cpu().tolist()).decode("utf-8"))
            if not isinstance(conf, dict) or conf.get("format") != "int8_tensorwise":
                raise ValueError("only int8_tensorwise checkpoints are supported")
            params = conf.get("params", {})
            if not isinstance(params, dict):
                raise ValueError("invalid quantization params")
            rotation = conf.get("convrot", params.get("convrot", False))
            group = conf.get("convrot_groupsize", params.get("convrot_groupsize", 256))
            if not isinstance(rotation, bool):
                raise ValueError("convrot must be a boolean")
            if rotation:
                if not isinstance(group, int) or isinstance(group, bool) or group < 1:
                    raise ValueError("invalid ConvRot group size")
                power = group
                while power > 1 and power % 4 == 0:
                    power //= 4
                if power != 1:
                    raise ValueError("ConvRot group size must be a power of four")
            weight, scale = state[base + ".weight"], state[base + ".weight_scale"]
            if weight.dtype != torch.int8 or weight.ndim != 2:
                raise ValueError("expected a two-dimensional INT8 weight")
            if not scale.is_floating_point() or scale.numel() not in {1, weight.shape[0]}:
                raise ValueError("expected a scalar or per-output-row weight scale")
            if not torch.isfinite(scale).all() or (scale < 0).any():
                raise ValueError("weight scale must be finite and nonnegative")
            if rotation and weight.shape[1] % group:
                raise ValueError("weight input dimension does not match the ConvRot group")
            state[base + ".weight"] = dequantize_int8_convrot(weight, scale,
                {"format": "int8_tensorwise", "convrot": rotation, "convrot_groupsize": group}, out_dtype=dtype)
            del state[marker], state[base + ".weight_scale"]
            count += 1
        except (ValueError, TypeError, KeyError, UnicodeError) as error:
            raise ValueError(f"Qwen3-VL quantized checkpoint: {base}: {error}. "
                             "Select a BF16 Qwen3-VL-8B encoder if this quantization format is unsupported.") from error
    for key in list(state):
        if key.endswith(".comfy_attention.config"):
            del state[key]  # Runtime attention hint; HF uses its own attention implementation.
        elif key.endswith(".weight_scale") or (key.endswith(".weight") and not state[key].is_floating_point()):
            raise ValueError(f"Qwen3-VL checkpoint has undecoded quantized weight metadata: {key}")
        elif state[key].is_floating_point():
            state[key] = state[key].to(dtype)
    return state, count
