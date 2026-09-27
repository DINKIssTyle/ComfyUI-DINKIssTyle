"""Independent, non-destructive photo adjustments for ComfyUI IMAGE tensors.

The controls approximate rendered-photo editing. They do not reproduce a RAW
camera pipeline or infer physical lens characteristics from a depth map.
"""

import base64
import io
import json
import math
import os
import tempfile
import threading
from pathlib import Path

from aiohttp import web
import torch
import torch.nn.functional as F
from PIL import Image
from server import PromptServer


# (kind, default, minimum, maximum, step). Keep this list in display order.
CONTROLS = {
    "light_exposure": ("float", 0.0, -5.0, 5.0, 0.05),
    "light_contrast": ("float", 0.0, -100.0, 100.0, 1.0),
    "light_highlights": ("float", 0.0, -100.0, 100.0, 1.0),
    "light_shadows": ("float", 0.0, -100.0, 100.0, 1.0),
    "light_whites": ("float", 0.0, -100.0, 100.0, 1.0),
    "light_blacks": ("float", 0.0, -100.0, 100.0, 1.0),
    "color_white_balance": ("choice", "Off", ("Off", "Auto", "Custom")),
    "color_temperature": ("float", 0.0, -100.0, 100.0, 1.0),
    "color_tint": ("float", 0.0, -100.0, 100.0, 1.0),
    "color_vibrance": ("float", 0.0, -100.0, 100.0, 1.0),
    "color_saturation": ("float", 0.0, -100.0, 100.0, 1.0),
    "effects_texture": ("float", 0.0, -100.0, 100.0, 1.0),
    "effects_clarity": ("float", 0.0, -100.0, 100.0, 1.0),
    "effects_dehaze": ("float", 0.0, -100.0, 100.0, 1.0),
    "effects_vignette": ("float", 0.0, -100.0, 100.0, 1.0),
    "effects_grain": ("float", 0.0, 0.0, 100.0, 1.0),
    "effects_glow": ("float", 0.0, 0.0, 100.0, 1.0),
    "detail_sharpening": ("float", 0.0, 0.0, 100.0, 1.0),
    "detail_noise_reduction": ("float", 0.0, 0.0, 100.0, 1.0),
    "detail_color_noise_reduction": ("float", 0.0, 0.0, 100.0, 1.0),
    "optics_distortion": ("float", 0.0, -100.0, 100.0, 1.0),
    "optics_vignette": ("float", 0.0, -100.0, 100.0, 1.0),
    "lens_blur_apply": ("bool", False),
    "lens_blur_focus": ("float", 0.5, 0.0, 1.0, 0.01),
    "lens_blur_amount": ("float", 50.0, 0.0, 100.0, 1.0),
    "lens_blur_bokeh": ("float", 4.0, 0.1, 30.0, 0.1),
    "lens_blur_aperture_blades": ("int", 9, 5, 18, 1),
    "lens_blur_bokeh_boost": ("float", 0.0, 0.0, 100.0, 1.0),
    "lens_blur_depth_blur_radius": ("int", 5, 0, 31, 1),
    "lens_blur_depth_sigma": ("float", 2.0, 0.1, 10.0, 0.1),
    "depth_near_is_white": ("bool", True),
    "grain_seed": ("int", 0, 0, 2147483647, 1),
}

TOOLTIPS = {
    "light_exposure": "Exposure in stops: +1 approximately doubles linear light.",
    "color_white_balance": "Off keeps the rendered white balance; Auto estimates it; Custom uses Temperature and Tint.",
    "color_temperature": "Relative rendered-image correction: negative is cooler, positive is warmer. Not Kelvin.",
    "color_tint": "Negative adds green; positive adds magenta.",
    "effects_texture": "Negative softens fine detail; positive emphasizes it.",
    "effects_clarity": "Adjusts larger-scale local contrast, especially around edges.",
    "effects_dehaze": "Negative adds haze; positive reduces haze.",
    "effects_vignette": "Creative vignette: negative darkens corners; positive brightens them.",
    "optics_distortion": "Negative and positive values compensate opposite radial lens distortions.",
    "optics_vignette": "Manual lens falloff: negative darkens corners; positive brightens them.",
    "lens_blur_focus": "Focus depth from 0 (far) to 1 (near, when white is near).",
    "lens_blur_bokeh": "Simulated f-number from 0.1 to 30. Lower values increase defocus.",
    "lens_blur_aperture_blades": "Number of straight aperture blades. Changes the shape of out-of-focus highlights.",
    "lens_blur_bokeh_boost": "Brightens out-of-focus highlights.",
    "lens_blur_depth_blur_radius": "Gaussian blur radius for the depth map. Zero disables depth smoothing.",
    "lens_blur_depth_sigma": "Gaussian sigma for depth boundaries, using the Blur Image node's kernel scale.",
}


def _defaults():
    return {key: spec[1] for key, spec in CONTROLS.items()}


def _validate_settings(values, complete=False):
    if not isinstance(values, dict) or set(values) - set(CONTROLS):
        raise ValueError("Unknown photo studio setting")
    if complete and set(values) != set(CONTROLS):
        raise ValueError("Preset must contain every photo studio setting")
    result = _defaults()
    for name, value in values.items():
        spec = CONTROLS[name]
        if spec[0] == "choice":
            if value not in spec[2]:
                raise ValueError(f"Invalid {name}")
        elif spec[0] == "bool":
            if type(value) is not bool:
                raise ValueError(f"Invalid {name}")
        elif spec[0] == "int":
            if type(value) is not int or not spec[2] <= value <= spec[3]:
                raise ValueError(f"Invalid {name}")
        elif isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not spec[2] <= value <= spec[3]:
            raise ValueError(f"Invalid {name}")
        result[name] = value
    return result


def _srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(0).pow(2.4))


def _linear_to_srgb(x):
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(0).pow(1 / 2.4) - 0.055)


def _blur(image, sigma):
    if sigma <= 0.05:
        return image
    radius = min(32, max(1, math.ceil(sigma * 2.5)))
    return _gaussian_blur(image, radius, sigma)


def _gaussian_blur(image, radius, sigma):
    if radius <= 0:
        return image
    t = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels)
    return F.conv2d(F.pad(image, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels)


def _depth_blur(image, radius, sigma):
    """Match ComfyUI Blur Image's normalized Gaussian kernel on a depth map."""
    if radius <= 0:
        return image
    coords = torch.linspace(-1, 1, radius * 2 + 1,
                            device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (coords / sigma).square())
    kernel = kernel / kernel.sum()
    mode = "reflect" if min(image.shape[-2:]) > radius else "replicate"
    image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode=mode),
                     kernel.view(1, 1, 1, -1))
    return F.conv2d(F.pad(image, (0, 0, radius, radius), mode=mode),
                    kernel.view(1, 1, -1, 1))


def _luminance(rgb):
    return (rgb * rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1, keepdim=True)


def _radial(rgb):
    _, _, height, width = rgb.shape
    y = (torch.arange(height, device=rgb.device, dtype=rgb.dtype) + 0.5) / height * 2 - 1
    x = (torch.arange(width, device=rgb.device, dtype=rgb.dtype) + 0.5) / width * 2 - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return ((xx * xx + yy * yy) * 0.5).clamp(0, 1).view(1, 1, height, width)


def _distort(rgb, depth, alpha, amount):
    _, _, height, width = rgb.shape
    y = (torch.arange(height, device=rgb.device, dtype=rgb.dtype) + 0.5) / height * 2 - 1
    x = (torch.arange(width, device=rgb.device, dtype=rgb.dtype) + 0.5) / width * 2 - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r2 = (xx * xx + yy * yy) * 0.5
    factor = 1 + (amount / 100 * 0.28) * r2
    grid = torch.stack((xx * factor, yy * factor), dim=-1).unsqueeze(0).expand(rgb.shape[0], -1, -1, -1)
    rgb = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=False)
    if depth is not None:
        depth = F.grid_sample(depth, grid, mode="bilinear", padding_mode="border", align_corners=False)
    if alpha is not None:
        alpha = F.grid_sample(alpha, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return rgb, depth, alpha


def _prepare_depth(depth_image, rgb, near_is_white, blur_radius=0, sigma=2.0):
    if depth_image is None:
        raise ValueError("Lens Blur requires a depth_image input")
    if depth_image.ndim != 4 or depth_image.shape[-1] not in (1, 3, 4):
        raise ValueError("depth_image must be a BHWC grayscale or RGB image")
    batch, _, height, width = rgb.shape
    if depth_image.shape[0] not in (1, batch):
        raise ValueError("depth_image batch size must be 1 or match image batch size")
    dh, dw = depth_image.shape[1:3]
    if abs(dw / dh - width / height) > 0.02:
        raise ValueError("depth_image aspect ratio must match image aspect ratio")
    depth = depth_image[..., :3].to(device=rgb.device, dtype=rgb.dtype).mean(-1, keepdim=True)
    if not torch.isfinite(depth).all():
        raise ValueError("depth_image contains non-finite values")
    depth = depth.permute(0, 3, 1, 2)
    depth = _depth_blur(depth, blur_radius, sigma)
    if batch > 1 and depth.shape[0] == 1:
        depth = depth.expand(batch, -1, -1, -1)
    if (dh, dw) != (height, width):
        depth = (F.interpolate(depth, size=(height, width), mode="area")
                 if dh >= height and dw >= width else _joint_upsample_depth(depth, rgb))
    depth = depth.clamp(0, 1)
    return depth if near_is_white else 1 - depth


def _joint_upsample_depth(depth, rgb):
    """Align low-resolution depth to image edges, then smooth sampling steps."""
    batch, _, height, width = rgb.shape
    dh, dw = depth.shape[-2:]
    low_rgb = F.interpolate(rgb, size=(dh, dw), mode="area")
    y = (torch.arange(height, device=rgb.device, dtype=torch.float32) + 0.5) * dh / height - 0.5
    x = (torch.arange(width, device=rgb.device, dtype=torch.float32) + 0.5) * dw / width - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    y0, x0 = yy.floor(), xx.floor()
    fy, fx = (yy - y0).clamp(0, 1), (xx - x0).clamp(0, 1)
    depth_flat = depth.flatten(2)
    guide_flat = low_rgb.flatten(2)
    target = rgb.flatten(2)
    weighted_depth = torch.zeros((batch, 1, height * width), device=rgb.device, dtype=rgb.dtype)
    total_weight = torch.zeros_like(weighted_depth)
    for yi, wy in ((y0, 1 - fy), (y0 + 1, fy)):
        for xi, wx in ((x0, 1 - fx), (x0 + 1, fx)):
            indices = (yi.clamp(0, dh - 1).long() * dw + xi.clamp(0, dw - 1).long()).reshape(1, 1, -1)
            indices = indices.expand(batch, -1, -1)
            sample_depth = depth_flat.gather(2, indices)
            sample_rgb = guide_flat.gather(2, indices.expand(-1, 3, -1))
            color_distance = (sample_rgb - target).square().mean(1, keepdim=True)
            color_weight = 0.01 + torch.exp(-color_distance / 0.02)
            weight = (wy * wx).reshape(1, 1, -1) * color_weight
            weighted_depth += sample_depth * weight
            total_weight += weight
    resized = (weighted_depth / total_weight.clamp_min(1e-6)).reshape(batch, 1, height, width)
    radius = 2
    kernel_size = 2 * radius + 1

    def local_mean(value):
        return F.avg_pool2d(value, kernel_size, stride=1, padding=radius,
                            count_include_pad=False)

    guide = _luminance(rgb.clamp(0, 1))
    mean_guide = local_mean(guide)
    mean_depth = local_mean(resized)
    variance = local_mean(guide.square()) - mean_guide.square()
    covariance = local_mean(guide * resized) - mean_guide * mean_depth
    slope = covariance / (variance.clamp_min(0) + 0.0025)
    intercept = mean_depth - slope * mean_guide
    refined = local_mean(slope) * guide + local_mean(intercept)
    return (refined * 0.8 + resized * 0.2).clamp(0, 1)


def _depth_preview(depth_image, blur_radius=0, sigma=2.0):
    """Return a compact grayscale preview of the first input depth map."""
    if depth_image is None or not isinstance(depth_image, torch.Tensor) or \
            depth_image.ndim != 4 or depth_image.shape[0] < 1 or \
            depth_image.shape[-1] not in (1, 3, 4):
        return None
    depth = depth_image[:1, ..., :3].detach().float().mean(-1, keepdim=True)
    if not torch.isfinite(depth).all():
        return None
    depth = _depth_blur(depth.permute(0, 3, 1, 2), blur_radius, sigma).permute(0, 2, 3, 1)
    height, width = depth.shape[1:3]
    if min(height, width) < 1:
        return None
    if max(height, width) > 320:
        scale = 320 / max(height, width)
        depth = F.interpolate(depth.permute(0, 3, 1, 2),
                              size=(max(1, round(height * scale)),
                                    max(1, round(width * scale))),
                              mode="area").permute(0, 2, 3, 1)
    pixels = (depth[0, ..., 0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG", compress_level=4)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _aperture_kernel(radius, blades, device, dtype):
    """Unit-energy, antialiased regular-polygon point-spread function."""
    half = max(1, math.ceil(radius + 0.5))
    coordinates = torch.arange(-half, half + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    sector = 2 * math.pi / blades
    angle = torch.atan2(yy, xx)
    offset = torch.remainder(angle + sector / 2, sector) - sector / 2
    boundary = radius * math.cos(math.pi / blades) / torch.cos(offset)
    coverage = (boundary - torch.sqrt(xx.square() + yy.square()) + 0.5).clamp(0, 1)
    kernel = coverage / coverage.sum()
    return kernel.to(device=device, dtype=dtype).view(1, 1, *kernel.shape)


def _aperture_blur(image, radius, blades):
    """Convolve with a polygon aperture, using a smaller canvas for wide blur."""
    if radius < 2:
        return _blur(image, radius / 2)
    height, width = image.shape[-2:]
    scale = max(1, math.ceil(radius / 10))
    if scale > 1:
        image = F.interpolate(image, size=(max(1, math.ceil(height / scale)),
                                           max(1, math.ceil(width / scale))), mode="area")
    kernel = _aperture_kernel(radius / scale, blades, image.device, image.dtype)
    half = kernel.shape[-1] // 2
    channels = image.shape[1]
    blurred = F.conv2d(F.pad(image, (half, half, half, half), mode="replicate"),
                       kernel.expand(channels, 1, -1, -1), groups=channels)
    if scale > 1:
        blurred = F.interpolate(blurred, size=(height, width), mode="bilinear", align_corners=False)
    return blurred


def _starburst_kernel(radius, blades, device, dtype):
    """Diffraction-style rays: odd blades produce 2N, even blades N."""
    rays = blades * 2 if blades % 2 else blades
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    distance = torch.sqrt(xx.square() + yy.square())
    sector = 2 * math.pi / rays
    angle = torch.atan2(yy, xx)
    offset = torch.remainder(angle + sector / 2, sector) - sector / 2
    transverse = distance * torch.sin(offset)
    kernel = torch.exp(-0.5 * (transverse / 0.45).square()) * \
        (1 - distance / (radius + 1)).clamp_min(0).pow(1.5)
    kernel[radius, radius] = 0
    kernel = kernel / kernel.max().clamp_min(1e-6) * (0.35 / rays)
    return kernel.to(device=device, dtype=dtype).view(1, 1, *kernel.shape)


def _add_starburst(image, source, f_number, blades):
    if f_number <= 8:
        return image
    peak = source.max(1, keepdim=True).values
    isolation = ((peak - _blur(peak, 2) - 0.2) / 0.4).clamp(0, 1)
    highlights = source * ((peak - 0.92) / 0.08).clamp(0, 1) * isolation
    if not torch.any(highlights > 1e-4):
        return image
    height, width = image.shape[-2:]
    if min(height, width) < 5:
        return image
    radius = round(12 + 20 * min(1, (f_number - 8) / 22))
    scale = min(max(1, math.ceil(radius / 10)), height, width)
    if scale > 1:
        highlights = F.max_pool2d(highlights, kernel_size=scale, stride=scale,
                                   ceil_mode=True)
    kernel = _starburst_kernel(max(2, round(radius / scale)), blades,
                               highlights.device, highlights.dtype)
    half = kernel.shape[-1] // 2
    channels = highlights.shape[1]
    rays = F.conv2d(F.pad(highlights, (half, half, half, half), mode="replicate"),
                    kernel.expand(channels, 1, -1, -1), groups=channels)
    if scale > 1:
        rays = F.interpolate(rays, size=(height, width), mode="bilinear", align_corners=False)
    strength = min(1, (f_number - 8) / 12)
    return image + rays * strength


def _lens_blur(rgb, depth, settings, depth_was_upsampled=False):
    amount = settings["lens_blur_amount"]
    if amount <= 0:
        return rgb
    focus = settings["lens_blur_focus"]
    if torch.all((depth - focus).abs() < 1e-6):
        return _add_starburst(rgb, rgb, settings["lens_blur_bokeh"],
                              settings["lens_blur_aperture_blades"])
    # Smaller f-number produces stronger apparent defocus. The polygon point-
    # spread function gives out-of-focus highlights the aperture's blade shape.
    # Depth layers are convolved in premultiplied form to protect boundaries.
    aperture = math.sqrt(4 / settings["lens_blur_bokeh"])
    max_radius = min(180.0, amount / 100 * 36 * aperture)
    blades = settings["lens_blur_aperture_blades"]
    highlight_gain = 6 + settings["lens_blur_bokeh_boost"] / 100 * 14
    linear = _srgb_to_linear(rgb.clamp(0, 1))
    focus_distance = (depth - focus).abs()
    # Protect the focus plane and extend its matte across depth discontinuities.
    # Mixed depth pixels at silhouettes must not spread subject color into the
    # blurred background, but a smooth depth gradient should still defocus.
    focus_coverage = ((0.10 - focus_distance) / 0.04).clamp(0, 1)
    if depth_was_upsampled:
        local_max = F.max_pool2d(depth, 5, stride=1, padding=2)
        local_min = -F.max_pool2d(-depth, 5, stride=1, padding=2)
        edge_strength = ((local_max - local_min - 0.08) / 0.12).clamp(0, 1)
        edge_focus = ((0.24 - focus_distance) / 0.06).clamp(0, 1) * edge_strength
        focus_coverage = torch.maximum(focus_coverage, edge_focus)
    luminance = _luminance(linear)
    isolated = ((luminance - _blur(luminance, 2) - 0.15) / 0.25).clamp(0, 1)
    highlights = linear * ((luminance - 0.72) / 0.28).clamp(0, 1) * isolated
    if not torch.any(highlights > 1e-4):
        highlights = None
    bins = min(17, max(9, math.ceil(max_radius / 12) + 1))
    accumulated_color = torch.zeros_like(linear)
    accumulated_alpha = torch.zeros_like(depth)
    focus_inserted = False
    for index in range(bins):
        center = index / (bins - 1)
        if not focus_inserted and center >= focus:
            accumulated_color = linear * focus_coverage + accumulated_color * (1 - focus_coverage)
            accumulated_alpha = focus_coverage + accumulated_alpha * (1 - focus_coverage)
            focus_inserted = True
        mask = (1 - (depth - center).abs() * (bins - 1)).clamp(0, 1)
        mask = mask * (1 - focus_coverage)
        radius = max_radius * abs(center - focus)
        if radius <= 0.3:
            layer_color, coverage = linear * mask, mask
        else:
            channels = [linear * mask, mask]
            if highlights is not None:
                channels.append(highlights * mask)
            blurred = _aperture_blur(torch.cat(channels, dim=1), radius, blades)
            layer_color, coverage = blurred[:, :3], blurred[:, 3:4].clamp(0, 1)
            if highlights is not None:
                layer_color = layer_color + blurred[:, 4:7] * highlight_gain
        # Premultiplied, far-to-near compositing prevents the sharp source
        # image from showing through translucent gaps between depth layers.
        accumulated_color = layer_color + accumulated_color * (1 - coverage)
        accumulated_alpha = coverage + accumulated_alpha * (1 - coverage)
    result = torch.where(accumulated_alpha > 1e-5,
                         accumulated_color / accumulated_alpha.clamp_min(1e-5), linear)
    result = _linear_to_srgb(result.clamp_min(0))
    return _add_starburst(result, rgb, settings["lens_blur_bokeh"], blades)


def _auto_light_controls(image):
    """Derive conservative rendered-image tone controls from robust percentiles."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] not in (3, 4):
        raise ValueError("image must be a BHWC RGB or RGBA IMAGE")
    if not torch.isfinite(image).all():
        raise ValueError("image contains non-finite values")
    rgb = image[..., :3].detach().float().clamp(0, 1).permute(0, 3, 1, 2)
    height, width = rgb.shape[-2:]
    if max(height, width) > 256:
        scale = 256 / max(height, width)
        rgb = F.interpolate(rgb, size=(max(1, round(height * scale)),
                                       max(1, round(width * scale))), mode="area")
    luminance = _luminance(rgb).flatten().to("cpu")
    p01, p05, median, p95, p99 = (float(value) for value in
                                   torch.quantile(luminance, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99])))
    values = {name: 0.0 for name in (
        "light_exposure", "light_contrast", "light_highlights",
        "light_shadows", "light_whites", "light_blacks")}
    if p99 < 0.005 or p01 > 0.995:
        return values

    def linear(x):
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    def encoded(x):
        return x * 12.92 if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055

    exposure = max(-2.5, min(2.5, math.log2(linear(0.45) / max(linear(median), 1e-5))))
    exposure = round(exposure * 20) / 20
    values["light_exposure"] = exposure
    factor = 2 ** exposure
    p01, p05, p95, p99 = (min(1.0, encoded(linear(value) * factor))
                           for value in (p01, p05, p95, p99))
    span = p95 - p05
    if span > 0.08:
        values["light_contrast"] = float(round(max(-35, min(35, (0.7 / span - 1) * 40))))
        values["light_highlights"] = float(round(max(-60, min(0, (0.88 - p99) * 400))))
        values["light_shadows"] = float(round(max(-35, min(45, (0.15 - p05) * 250))))
        values["light_whites"] = float(round(max(-35, min(35, (0.96 - p99) * 200))))
        values["light_blacks"] = float(round(max(-35, min(35, (0.02 - p01) * 250))))
    return values


def _process(image, depth_image, settings):
    rgb = image[..., :3].permute(0, 3, 1, 2).float().clamp(0, 1)
    alpha = image[..., 3:4].permute(0, 3, 1, 2).float() if image.shape[-1] == 4 else None
    depth = _prepare_depth(depth_image, rgb, settings["depth_near_is_white"],
                           settings["lens_blur_depth_blur_radius"],
                           settings["lens_blur_depth_sigma"]) if settings["lens_blur_apply"] else None
    if settings["optics_distortion"]:
        rgb, depth, alpha = _distort(rgb, depth, alpha, settings["optics_distortion"])
    if settings["optics_vignette"]:
        rgb = rgb * (1 + _radial(rgb).pow(1.5) * settings["optics_vignette"] / 100 * 0.8)

    wb = settings["color_white_balance"]
    if wb == "Auto":
        # Gray-world estimate from midtone, low-chroma pixels only.
        luma = _luminance(rgb)
        chroma = rgb.max(1, keepdim=True).values - rgb.min(1, keepdim=True).values
        sample = ((luma > 0.12) & (luma < 0.88) & (chroma < 0.18)).float()
        count = sample.sum((2, 3), keepdim=True)
        means = (rgb * sample).sum((2, 3), keepdim=True) / count.clamp_min(1)
        target = means.mean(1, keepdim=True)
        gains = (target / means.clamp_min(0.01)).clamp(0.75, 1.25)
        rgb = rgb * torch.where(count >= rgb.shape[2] * rgb.shape[3] * 0.01, gains, torch.ones_like(gains))
    elif wb == "Custom":
        temp = settings["color_temperature"] / 100
        tint = settings["color_tint"] / 100
        gains = rgb.new_tensor([1 + temp * 0.20 + tint * 0.05, 1 - tint * 0.16, 1 - temp * 0.20 + tint * 0.05]).view(1, 3, 1, 1)
        rgb = rgb * gains

    exposure = settings["light_exposure"]
    if exposure:
        rgb = _linear_to_srgb(_srgb_to_linear(rgb.clamp(0, 1)) * (2 ** exposure))
    luma = _luminance(rgb)
    delta = torch.zeros_like(luma)
    for name, center, width, gain in (
        ("light_highlights", 0.72, 0.32, 0.35),
        ("light_shadows", 0.28, 0.32, 0.35),
        ("light_whites", 0.95, 0.20, 0.28),
        ("light_blacks", 0.05, 0.20, 0.28),
    ):
        value = settings[name]
        if value:
            weight = (1 - (luma - center).abs() / width).clamp_min(0)
            delta = delta + weight * (value / 100 * gain)
    if settings["light_contrast"]:
        delta = delta + (luma - 0.5) * settings["light_contrast"] / 100 * 0.65
    if any(settings[key] for key in ("light_contrast", "light_highlights", "light_shadows", "light_whites", "light_blacks")):
        rgb = rgb + delta

    if settings["effects_dehaze"]:
        strength = settings["effects_dehaze"] / 100 * 0.35
        local_black = _blur(rgb.min(1, keepdim=True).values, 12)
        rgb = (rgb - local_black * strength) / (1 - strength * 0.5)
    if settings["detail_noise_reduction"]:
        strength = settings["detail_noise_reduction"] / 100 * 0.85
        luma = _luminance(rgb)
        rgb = rgb + (_blur(luma, 1.25) - luma) * strength
    if settings["detail_color_noise_reduction"]:
        strength = settings["detail_color_noise_reduction"] / 100 * 0.9
        luma = _luminance(rgb)
        chroma = rgb - luma
        rgb = luma + chroma * (1 - strength) + _blur(chroma, 1.5) * strength
    for name, sigma, scale in (("effects_texture", 1.2, 0.6), ("effects_clarity", 7.0, 0.7), ("detail_sharpening", 0.8, 1.0)):
        value = settings[name]
        if value:
            luma = _luminance(rgb)
            rgb = rgb + (luma - _blur(luma, sigma)) * (value / 100 * scale)

    if settings["color_saturation"] or settings["color_vibrance"]:
        luma = _luminance(rgb)
        chroma = rgb - luma
        sat = settings["color_saturation"] / 100
        vibrance = settings["color_vibrance"] / 100
        existing = (rgb.max(1, keepdim=True).values - rgb.min(1, keepdim=True).values).clamp(0, 1)
        factor = (1 + sat + vibrance * (1 - existing)).clamp_min(0)
        rgb = luma + chroma * factor

    if depth is not None:
        depth_was_upsampled = depth_image.shape[1] < image.shape[1] or \
            depth_image.shape[2] < image.shape[2]
        rgb = _lens_blur(rgb, depth, settings, depth_was_upsampled)
    if settings["effects_vignette"]:
        rgb = rgb * (1 + _radial(rgb).pow(1.5) * settings["effects_vignette"] / 100 * 0.8)
    if settings["effects_glow"]:
        highlight = (rgb - 0.65).clamp_min(0)
        rgb = rgb + _blur(highlight, 5) * settings["effects_glow"] / 100 * 0.8
    if settings["effects_grain"]:
        generator = torch.Generator(device="cpu").manual_seed(settings["grain_seed"])
        noise = torch.randn((rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]), generator=generator, dtype=torch.float32)
        rgb = rgb + noise.to(rgb.device) * settings["effects_grain"] / 100 * 0.055

    output = rgb.clamp(0, 1).permute(0, 2, 3, 1).to(image.dtype)
    if alpha is not None:
        output = torch.cat((output, alpha.permute(0, 2, 3, 1).to(image.dtype)), dim=-1)
    return output


class DINKI_Photo_Studio:
    CATEGORY = "DINKIssTyle/Color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        required = {"image": ("IMAGE",), "active": ("BOOLEAN", {"default": True}),
                    "preset": ("STRING", {"default": "Custom"})}
        for name, spec in CONTROLS.items():
            if spec[0] == "choice":
                options = {"default": spec[1]}
                if name in TOOLTIPS:
                    options["tooltip"] = TOOLTIPS[name]
                required[name] = (list(spec[2]), options)
            elif spec[0] == "bool":
                required[name] = ("BOOLEAN", {"default": spec[1]})
            else:
                options = {"default": spec[1], "min": spec[2], "max": spec[3],
                           "step": spec[4], "display": "number" if name == "grain_seed" else "slider"}
                if name in TOOLTIPS:
                    options["tooltip"] = TOOLTIPS[name]
                required[name] = ("INT" if spec[0] == "int" else "FLOAT", options)
        return {"required": required, "optional": {"depth_image": ("IMAGE",)}}

    def execute(self, image, active=True, preset="Custom", depth_image=None, **settings):
        ui = {"depth_preview": [_depth_preview(
            depth_image, settings.get("lens_blur_depth_blur_radius", 5),
            settings.get("lens_blur_depth_sigma", 2.0))],
              "auto_light_suggestion": [_auto_light_controls(image)]}
        result = self.apply(image, active, preset, depth_image, **settings)
        return {"ui": ui, "result": result}

    def apply(self, image, active=True, preset="Custom", depth_image=None, **settings):
        if not active:
            return (image,)
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] not in (3, 4):
            raise ValueError("image must be a BHWC RGB or RGBA IMAGE")
        if not torch.isfinite(image).all():
            raise ValueError("image contains non-finite values")
        settings = _validate_settings(settings)
        if all(not settings[key] for key in CONTROLS if key not in
               ("color_white_balance", "lens_blur_focus", "lens_blur_amount", "lens_blur_bokeh", "lens_blur_aperture_blades", "lens_blur_depth_blur_radius", "lens_blur_depth_sigma", "depth_near_is_white", "grain_seed")) and settings["color_white_balance"] == "Off":
            return (image,)
        return (_process(image, depth_image, settings),)


_PRESET_LOCK = threading.RLock()


def _preset_path(request):
    return Path(PromptServer.instance.user_manager.get_request_user_filepath(
        request, "dkst_photo_studio/presets.json", create_dir=True))


def _read_presets(path):
    if not path.exists():
        return {}
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Photo studio preset file is too large")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("presets"), dict):
        raise ValueError("Invalid photo studio preset file")
    legacy_controls = {"lens_blur_aperture_blades", "lens_blur_depth_blur_radius",
                       "lens_blur_depth_sigma"}
    presets = {}
    for name, values in data["presets"].items():
        if isinstance(values, dict) and set(CONTROLS) - set(values) <= legacy_controls:
            values = {**{key: CONTROLS[key][1] for key in legacy_controls if key not in values},
                      **values}
        presets[name] = _validate_settings(values, complete=True)
    return presets


def _save_preset(path, name, settings, overwrite):
    if not isinstance(name, str) or not name or len(name) > 80 or name in ("Default", "Custom") or any(ord(c) < 32 or c in "/\\" for c in name):
        raise ValueError("Invalid preset name")
    settings = _validate_settings(settings, complete=True)
    with _PRESET_LOCK:
        presets = _read_presets(path)
        if name in presets and not overwrite:
            raise FileExistsError("Preset already exists")
        if name not in presets and len(presets) >= 100:
            raise ValueError("Too many presets")
        presets[name] = settings
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".presets-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump({"version": 1, "presets": presets}, out, ensure_ascii=False)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return presets


@PromptServer.instance.routes.get("/dinki/photo-studio/presets")
async def photo_studio_presets(request):
    try:
        with _PRESET_LOCK:
            presets = _read_presets(_preset_path(request))
        return web.json_response({"presets": presets, "defaults": _defaults()})
    except (ValueError, OSError, json.JSONDecodeError) as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.post("/dinki/photo-studio/presets")
async def photo_studio_save_preset(request):
    try:
        if request.content_length is not None and request.content_length > 64 * 1024:
            raise ValueError("Preset request is too large")
        body = await request.content.read(64 * 1024 + 1)
        if len(body) > 64 * 1024:
            raise ValueError("Preset request is too large")
        payload = json.loads(body)
        presets = _save_preset(_preset_path(request), payload.get("name"),
                               payload.get("settings"), payload.get("overwrite") is True)
        return web.json_response({"presets": presets})
    except FileExistsError as error:
        return web.json_response({"error": str(error)}, status=409)
    except (ValueError, OSError, TypeError, json.JSONDecodeError, AttributeError) as error:
        return web.json_response({"error": str(error)}, status=400)
