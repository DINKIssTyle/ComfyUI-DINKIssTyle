"""Optional vision-LLM captions using the existing server-side DKST profiles."""
import base64
import io
import re

NONE = "-- Select connection --"
PROMPT = (
    "Write a concise English training caption for this photo. Describe the visible person's "
    "pose, viewing angle, expression, clothing, framing, lighting and background. "
    "Describe only visible details; do not identify the person or guess their name. "
    "Return only one plain-text caption, without a heading, quotation marks, bullet list "
    "or reasoning. Do not add a trigger word; the training tool adds it separately."
)


def profiles():
    from . import dinki_llm_connections
    return dinki_llm_connections


def caption_aliases():
    try:
        return [NONE, *profiles().aliases()]
    except ValueError:
        # Broken/missing LLM configuration must not prevent manual training.
        return [NONE]


def generate_caption(service, data):
    alias, model = data.get("caption_connection_alias"), data.get("caption_model", "")
    if not isinstance(alias, str) or not alias or alias == NONE:
        raise ValueError("Select an LLM alias registered in Settings > DKST > LLM for automatic captions.")
    if not isinstance(model, str) or not model.strip() or len(model) > 256 or any(ord(c) < 32 for c in model):
        raise ValueError("Enter the exact model name of an image-capable LLM for automatic captions.")
    connections = profiles()
    profile = connections.resolve(alias)
    path = service.resolve_image(data.get("image"))
    from PIL import Image, ImageOps
    try:
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
    except (OSError, ValueError, Image.DecompressionBombError):
        raise ValueError("Cannot read this gallery photo for automatic captioning.") from None
    url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    body = {"model": model.strip(), "stream": False, "temperature": 0.2, "max_tokens": 512,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]}]}
    import requests
    try:
        with requests.post(connections.connection_url(profile) + "/v1/chat/completions",
                           headers=connections.connection_headers(profile), json=body,
                           timeout=(10, 120), allow_redirects=False) as response:
            if not 200 <= response.status_code < 300:
                raise ValueError(f"Automatic caption request failed (HTTP {response.status_code}). "
                                 "Check the LLM connection, model name and image support.")
            try:
                text = response.json()["choices"][0]["message"]["content"]
                if not isinstance(text, str):
                    raise TypeError()
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
                if re.search(r"</?think>", text, re.IGNORECASE):
                    raise ValueError()
                text = " ".join(text.split()).strip('"')
                if not text or len(text) > 4000:
                    raise ValueError()
            except (ValueError, KeyError, IndexError, TypeError):
                raise ValueError("The LLM did not return a usable caption. Check the model's image support.") from None
    except requests.RequestException:
        # Exceptions and server error bodies can contain addresses or credentials.
        raise ValueError("Automatic caption request failed or timed out. Check Settings > DKST > LLM and the model.") from None
    return {"caption": text}
