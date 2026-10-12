"""Reproduce the Qwen-only Fizgig source subset from an unpacked upstream checkout.

Usage: python scripts/vendor_qwen_trainer.py /path/to/Fizgig
No upstream code is executed. Follow imports (including optional code paths),
retain package/config files and license notices, and record every local edit.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

COMMIT = "33454e1b0bf06dd2203551abd3bdcdfd8cdaee68"
SOURCE_URL = f"https://github.com/shootthesound/Fizgig/tree/{COMMIT}"


def build(source, destination):
    root = source / "src"
    edited = {}
    # Own layout adapter, reproduced alongside the upstream subset.
    edited["fizgig/qwen_image21/vae_weights.py"] = (Path(__file__).parent / "patches/qwen21_vae_weights.py").read_text()
    path = "fizgig/qwen_image21/vae.py"
    text = (root / path).read_text()
    text = text.replace('"""The diffusers-format VAE file', '"""The ComfyUI or Diffusers-format VAE file')
    text = text.replace("    vae.load_state_dict(load_file(path))", '''    from fizgig.qwen_image21.vae_weights import normalize_vae_weights
    state, layout = normalize_vae_weights(load_file(path), vae.state_dict(), cfg.get("num_res_blocks", 2))
    vae.load_state_dict(state, strict=True)
    print(f"[DKST] Loaded Qwen 2.1 VAE ({layout} layout, {len(state)} weights).", flush=True)''')
    edited[path] = text
    edited["fizgig/qwen_image21/text_weights.py"] = (Path(__file__).parent / "patches/qwen21_text_weights.py").read_text()
    path = "fizgig/qwen_image21/embedder.py"
    text = (root / path).read_text()
    text = text.replace('        sd = _convert_comfyui_qwen3vl_state_dict(load_split_weights(model_path, device="cpu", dtype=dtype))', '''        from fizgig.qwen_image21.text_weights import decode_text_weights
        sd, decoded = decode_text_weights(load_split_weights(model_path, device="cpu"), dtype=dtype)
        sd = _convert_comfyui_qwen3vl_state_dict(sd)
        if decoded:
            print(f"[DKST] Text encoder: decoded {decoded} ComfyUI INT8 weight(s) with scales and ConvRot metadata.", flush=True)''')
    edited[path] = text
    registry = (root / "fizgig/families/registry.py").read_text()
    for family in ("klein", "krea2", "minimax", "sdxl", "anima", "zimage", "acestep"):
        registry = "\n".join(line for line in registry.splitlines()
                             if not line.startswith(f"from fizgig.families.{family} import")) + "\n"
    registry = registry.replace("(KLEIN, MINIMAX, KREA2, QWEN_IMAGE_21, SDXL, ANIMA, ZIMAGE, ACESTEP15)",
                                "(QWEN_IMAGE_21,)")
    edited["fizgig/families/registry.py"] = registry
    # Generic caching helpers were housed in Klein's CLI files. Keep only
    # their reusable functions; omit the Klein loaders and entry points.
    for name in ("cache_latents", "cache_text"):
        path = f"fizgig/scripts/{name}.py"
        text = (root / path).read_text().split("def setup_parser()", 1)[0]
        edited[path] = "\n".join(line for line in text.splitlines()
                                   if not line.startswith("from fizgig.klein.model_utils import")) + "\n"
    # Qwen only needs this state-dict converter, not Krea's conditioner.
    path = "fizgig/krea2/embedder.py"
    original = (root / path).read_text()
    tree = ast.parse(original)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_convert_comfyui_qwen3vl_state_dict")
    edited[path] = ('"""Qwen checkpoint key conversion, extracted from Fizgig Krea2.\n'
                    'Modified by DKST; original attribution is in THIRD_PARTY_NOTICES.md.\n"""\n'
                    'from torch import Tensor\n\n' + ast.get_source_segment(original, fn) + "\n")
    # Omit optional workbench/video/auto-caption/fine-tune dependency trees.
    # Fail explicitly if an unsupported API is called, rather than leaving
    # missing imports hidden inside dormant methods.
    def unsupported_methods(path, names):
        text = edited.get(path, (root / path).read_text())
        lines = text.splitlines(keepends=True)
        nodes = [n for n in ast.walk(ast.parse(text))
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        for n in sorted(nodes, key=lambda n: n.lineno, reverse=True):
            start = n.body[0].lineno - 1
            indent = " " * n.body[0].col_offset
            lines[start:n.end_lineno] = [indent + 'raise NotImplementedError("This Qwen-only build supports image LoRA training.")\n']
        edited[path] = "".join(lines)

    unsupported_methods("fizgig/families/description.py", {"workbench_engine_class"})
    unsupported_methods("fizgig/families/loss_watch.py", {"_recaption"})
    for path in ("fizgig/families/driver.py", "fizgig/qwen_image21/driver.py"):
        tree = ast.parse((root / path).read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and any(isinstance(i, ast.ImportFrom) and i.module == "fizgig.families.ft"
                         for i in ast.walk(n))}
        unsupported_methods(path, names)
    path = "fizgig/families/train.py"
    text = (root / path).read_text()
    n = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "_FineTune")
    lines = text.splitlines(keepends=True)
    lines[n.lineno-1:n.end_lineno] = ['class _FineTune:\n',
        '    def __init__(self, *args, **kwargs):\n',
        '        raise NotImplementedError("Full fine-tuning is not included in this Qwen-only build.")\n']
    edited[path] = "".join(lines)
    path = "fizgig/dataset/image_dataset.py"
    text = (root / path).read_text()
    lines = text.splitlines(keepends=True)
    branches = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.If)
                and any(isinstance(i, ast.ImportFrom) and i.module == "fizgig.families"
                        and any(a.name == "clips" for a in i.names) for i in ast.walk(n))]
    # Only the innermost clip branch, leaving still-image processing intact.
    for n in sorted(branches, key=lambda n: n.lineno, reverse=True)[:1]:
        start, end = n.body[0].lineno - 1, n.body[-1].end_lineno
        lines[start:end] = [" " * n.body[0].col_offset + 'raise ValueError("Only still images are supported by the Qwen trainer.")\n']
    edited[path] = "".join(lines)
    pending = ["fizgig.families.train", "fizgig.families.cache", "fizgig.qwen_image21.driver"]
    found = {}

    def add_module(module):
        relative = Path(*module.split("."))
        for candidate in (relative.with_suffix(".py"), relative / "__init__.py"):
            if candidate.as_posix() in edited or (root / candidate).is_file():
                pending.append(module)
                return

    while pending:
        module = pending.pop()
        relative = Path(*module.split("."))
        path = relative.with_suffix(".py")
        if path.as_posix() not in edited and not (root / path).is_file():
            path = relative / "__init__.py"
        key = path.as_posix()
        if key in found:
            continue
        text = edited[key] if key in edited else (root / path).read_text()
        found[key] = text
        for parent in path.parents:
            if parent == Path("."):
                break
            add_module(".".join(parent.parts))
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("fizgig"):
                        add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
                if base.startswith("fizgig"):
                    add_module(base)
                    for alias in node.names:
                        add_module(base + "." + alias.name)
    destination.mkdir(parents=True, exist_ok=True)
    previous = destination / "UPSTREAM.json"
    if previous.is_file():
        for key, digest in json.loads(previous.read_text())["files"].items():
            if key not in found:
                obsolete = destination / "src" / key
                if obsolete.is_file():
                    if hashlib.sha256(obsolete.read_bytes()).hexdigest() != digest:
                        raise RuntimeError(f"Refusing to remove locally edited vendor file: {key}")
                    obsolete.unlink()
    for key, text in found.items():
        target = destination / "src" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    # Small local model configs loaded by filename are outside the import graph.
    for path in (root / "fizgig/qwen_image21").glob("*.json"):
        shutil.copy2(path, destination / "src/fizgig/qwen_image21" / path.name)
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(source / name, destination / name)
    manifest = {"upstream": SOURCE_URL, "commit": COMMIT,
                "modified_files": sorted(edited),
                "files": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(found.items())}}
    (destination / "UPSTREAM.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Vendored {len(found)} Python modules; only qwen_image21 is registered.")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/qwen21_trainer/vendor")
