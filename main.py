#!/usr/bin/env python3
"""
main.py - Evaluate Stable Diffusion LoRAs via the ComfyUI API.

For each .safetensors file found in the target directory, injects LoRA
syntax (and optional trigger words) into a ComfyUI workflow, queues
generation for each configured prompt, saves individual images, and
produces a composite grid (one row per prompt, one column per epoch)
and a similarity graph based on the first prompt.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid

import imagehash
import requests
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load and validate the JSON config file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    required = [
        "comfyui_url",
        "workflow_file",
        "positive_node_id",
        "positive_text_field",
        "negative_node_id",
        "negative_text_field",
    ]
    for key in required:
        if key not in config:
            raise ValueError(f"Config missing required key: '{key}'")

    # Support both old single-prompt format and new multi-prompt format.
    # Old: positive_prompt (str) + negative_prompt (str)
    # New: prompts (list of {label, positive}) + negative_prompt (str)
    if "prompts" not in config and "positive_prompt" not in config:
        raise ValueError(
            "Config must have either 'prompts' or 'positive_prompt'."
        )
    # negative_prompt is optional at the top level if every prompt
    # defines its own negative field
    if "negative_prompt" not in config:
        for p in config["prompts"]:
            if "negative" not in p:
                raise ValueError(
                    "Each prompt must have a 'negative' field when "
                    "top-level 'negative_prompt' is not set."
                )

    # Normalise to the multi-prompt format internally
    if "prompts" not in config:
        config["prompts"] = [
            {
                "label": "default",
                "positive": config["positive_prompt"],
                "negative": config["negative_prompt"],
            }
        ]

    return config


def load_workflow(workflow_path):
    """Load a ComfyUI API-format workflow JSON file."""
    with open(workflow_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _xdg_config_dir():
    return os.path.join(
        os.environ.get("XDG_CONFIG_HOME",
                       os.path.expanduser("~/.config")),
        "lora-eval",
    )


def _resolve_config_path(given_path):
    """Resolve config path, checking ~/.config/lora-eval/ as fallback."""
    if given_path is not None:
        return given_path
    return os.path.join(_xdg_config_dir(), "config.json")


def _resolve_workflow_path(cfg_workflow):
    """Resolve workflow path, checking ~/.config/lora-eval/ as fallback."""
    if os.path.isabs(cfg_workflow):
        return cfg_workflow
    return os.path.join(_xdg_config_dir(), cfg_workflow)


# ---------------------------------------------------------------------------
# safetensors header parsing
# ---------------------------------------------------------------------------

def read_safetensors_metadata(path):
    """
    Parse a safetensors header and return selected training metadata.

    Returns a dict with keys:
      ``trigger``  -- most-frequent @-prefixed tag, or "" if none found
      ``name``     -- ss_output_name value, or "" if absent
      ``epoch``    -- ss_epoch as an int, or None if absent/unparseable

    All values fall back gracefully if the header cannot be read or the
    relevant fields are missing.
    """
    result = {"trigger": "", "name": "", "epoch": None}

    try:
        with open(path, "rb") as fh:
            # Header length is a little-endian 64-bit uint.
            raw_len = fh.read(8)
            if len(raw_len) < 8:
                return result
            header_len = int.from_bytes(raw_len, "little")
            header_bytes = fh.read(header_len)
    except OSError:
        return result

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return result

    metadata = header.get("__metadata__", {})

    result["name"] = metadata.get("ss_output_name", "")

    raw_epoch = metadata.get("ss_epoch")
    if raw_epoch is not None:
        try:
            result["epoch"] = int(raw_epoch)
        except (ValueError, TypeError):
            pass

    ss_datasets_raw = metadata.get("ss_datasets", "")
    if not ss_datasets_raw:
        return result

    try:
        datasets = json.loads(ss_datasets_raw)
    except (json.JSONDecodeError, TypeError):
        return result

    # Accumulate tag counts across all datasets and folders.
    totals = {}
    for dataset in datasets:
        tag_freq = dataset.get("tag_frequency", {})
        for folder_tags in tag_freq.values():
            for tag, count in folder_tags.items():
                totals[tag] = totals.get(tag, 0) + count

    # Only @-prefixed tags are anima-style trigger words.
    at_tags = {t: c for t, c in totals.items() if t.startswith("@")}
    if at_tags:
        result["trigger"] = max(at_tags, key=at_tags.__getitem__)

    return result


# ---------------------------------------------------------------------------
# LoRA discovery
# ---------------------------------------------------------------------------

def _lora_sort_key(path):
    """
    Sort key that places epoch-numbered files in numeric order and puts
    any unnumbered file (the final merged epoch) last.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r'(\d+)$', stem)
    return (0, int(match.group(1))) if match else (1, 0)


def find_loras(directory):
    """
    Return a sorted list of .safetensors paths in *directory*.

    Epoch-numbered files are ordered numerically. Any file whose stem
    has no trailing number (the final merged checkpoint) is placed last.
    """
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".safetensors")
    ]
    return sorted(files, key=_lora_sort_key)


def _compute_epoch_labels(loras):
    numbers = []
    for path in loras:
        stem = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r'(\d+)$', stem)
        numbers.append(int(match.group(1)) if match else None)

    numbered = [(i, n) for i, n in enumerate(numbers) if n is not None]
    labels = []
    for num in numbers:
        if num is not None:
            labels.append(str(num))
        elif len(numbered) >= 2:
            step = numbered[-1][1] - numbered[-2][1]
            labels.append(str(numbered[-1][1] + step))
        elif len(numbered) == 1:
            labels.append(str(numbered[0][1] + 1))
        else:
            labels.append("0")
    return labels


def lora_syntax(safetensors_path, weight, trigger_words):
    """
    Build the LoRA syntax string appended to the positive prompt.

    Derives the name by resolving the path to absolute and splitting
    at the first occurrence of *models/loras/*.  If that marker is
    not present, falls back to the bare filename stem.
    Format: <lora:NAME:WEIGHT>
    """
    abs_path = os.path.abspath(safetensors_path)
    marker = "models/loras/"
    idx = abs_path.find(marker)
    if idx != -1:
        relative = abs_path[idx + len(marker):]
        stem = os.path.splitext(relative)[0]
    else:
        stem = os.path.splitext(os.path.basename(abs_path))[0]
    tag = f"<lora:{stem}:{weight}>"
    if trigger_words:  # treats None and "" identically
        return f"{trigger_words}, {tag}"
    return tag


# ---------------------------------------------------------------------------
# Workflow manipulation
# ---------------------------------------------------------------------------

def inject_prompts(workflow, config, positive_text, negative_text):
    """
    Return a deep copy of *workflow* with positive/negative text injected.
    """
    wf = copy.deepcopy(workflow)

    pos_node = str(config["positive_node_id"])
    neg_node = str(config["negative_node_id"])
    pos_field = config["positive_text_field"]
    neg_field = config["negative_text_field"]

    if pos_node not in wf:
        raise KeyError(
            f"Positive node '{pos_node}' not found in workflow."
        )
    if neg_node not in wf:
        raise KeyError(
            f"Negative node '{neg_node}' not found in workflow."
        )

    wf[pos_node]["inputs"][pos_field] = positive_text
    wf[neg_node]["inputs"][neg_field] = negative_text

    return wf


# ---------------------------------------------------------------------------
# ComfyUI API helpers
# ---------------------------------------------------------------------------

def queue_prompt(api_url, workflow, client_id):
    """POST a workflow to the /prompt endpoint and return the prompt_id."""
    payload = json.dumps(
        {"prompt": workflow, "client_id": client_id}
    ).encode("utf-8")
    url = f"{api_url.rstrip('/')}/prompt"
    resp = requests.post(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_prompt(api_url, prompt_id, poll_interval=1.0, timeout=600):
    """
    Poll /history/{prompt_id} until the job is complete.

    Returns the history entry dict for the completed job.
    Raises TimeoutError if *timeout* seconds elapse first.
    """
    url = f"{api_url.rstrip('/')}/history/{prompt_id}"
    elapsed = 0.0
    while elapsed < timeout:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(
        f"Prompt {prompt_id} did not complete within {timeout}s."
    )


def fetch_image(api_url, filename, subfolder, folder_type):
    """Download a generated image from /view and return a PIL Image."""
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type,
    })
    url = f"{api_url.rstrip('/')}/view?{params}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    from io import BytesIO
    return Image.open(BytesIO(resp.content)).copy()


def collect_images(api_url, history_entry):
    """
    Extract all output images from a completed history entry.

    Returns a list of PIL Image objects in the order ComfyUI reports them.
    """
    images = []
    outputs = history_entry.get("outputs", {})
    for _node_id, node_output in outputs.items():
        for img_meta in node_output.get("images", []):
            img = fetch_image(
                api_url,
                img_meta["filename"],
                img_meta.get("subfolder", ""),
                img_meta.get("type", "output"),
            )
            images.append(img)
    return images


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def image_path_for_lora(out_dir, lora_path, prompt_index=0):
    """
    Return the output .png path for a LoRA/prompt combination.

    The first prompt (index 0) uses the plain stem so LoRA Manager can
    find it by swapping the .safetensors extension. Additional prompts
    get a suffix, e.g. stem_p1.png, stem_p2.png.
    """
    stem = os.path.splitext(os.path.basename(lora_path))[0]
    if prompt_index == 0:
        return os.path.join(out_dir, f"{stem}.png")
    return os.path.join(out_dir, f"{stem}_p{prompt_index}.png")


def save_individual(image, out_dir, lora_path, prompt_index=0):
    """Save a PIL image for the given LoRA and prompt index."""
    dest = image_path_for_lora(out_dir, lora_path, prompt_index)
    image.save(dest)
    return dest


def sha256_of_file(path, chunk=1 << 20):
    """Return the hex SHA-256 digest of *path*, reading in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_metadata(lora_path, image_path, trigger_words="", overwrite=False):
    """
    Create or update the LoRA Manager metadata JSON for *lora_path*.

    On first run the full template is written. On subsequent runs only
    ``preview_url`` is updated so user edits are not clobbered,
    unless *overwrite* is True.
    """
    stem = os.path.splitext(os.path.basename(lora_path))[0]
    meta_path = os.path.join(
        os.path.dirname(lora_path), f"{stem}.metadata.json"
    )

    if os.path.exists(meta_path) and not overwrite:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["preview_url"] = image_path
        if trigger_words:
            meta.setdefault("civitai", {})["trainedWords"] = [trigger_words]
        print(f"  Updated metadata: {os.path.basename(meta_path)}")
    else:
        stat = os.stat(lora_path)
        print(
            f"  Hashing {os.path.basename(lora_path)}"
            " (this may take a moment)…"
        )
        meta = {
            "file_name": stem,
            "model_name": stem,
            "file_path": os.path.abspath(lora_path),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "sha256": sha256_of_file(lora_path),
            "base_model": "Unknown",
            "preview_url": image_path,
            "preview_nsfw_level": 0,
            "notes": "",
            "from_civitai": True,
            "civitai": (
                {"trainedWords": [trigger_words]}
                if trigger_words else {}
            ),
            "tags": [],
            "modelDescription": "",
            "civitai_deleted": False,
            "favorite": False,
            "exclude": False,
            "db_checked": False,
            "skip_metadata_refresh": False,
            "metadata_source": None,
            "last_checked_at": 0,
            "hash_status": "completed",
            "usage_tips": "{}",
        }
        print(f"  Created metadata: {os.path.basename(meta_path)}")

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta_path


# ---------------------------------------------------------------------------
# Shared font helper
# ---------------------------------------------------------------------------

def _get_font(size):
    """Try to load a TTF font; fall back to the PIL default."""
    candidates = [
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/home/thnikk/.local/share/fonts/iosevka-custom-regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except IOError:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Composite image
# ---------------------------------------------------------------------------

def _truncate_label(label, font, max_width):
    """Truncate *label* with an ellipsis if it exceeds *max_width* px."""
    def text_width(t):
        try:
            bb = font.getbbox(t)
            return bb[2] - bb[0]
        except AttributeError:
            return font.getlength(t)

    if text_width(label) <= max_width * 0.92:
        return label
    while len(label) > 1:
        label = label[:-1]
        if text_width(label + "…") <= max_width * 0.92:
            return label + "…"
    return "…"


def build_composite(rows, out_path, title=""):
    """
    Build and save a multi-row composite image.

    *rows* is a list of rows, where each row is a list of
    (PIL.Image, label_str) tuples representing one epoch.
    All rows must have the same number of columns.

    Layout:
    - Column headers (epoch labels) drawn once across the top.
    - Row label drawn as a left-side sidebar for each prompt row.
    - Images fill the grid cells.
    """
    if not rows or not rows[0]:
        return

    n_cols = len(rows[0])
    n_rows = len(rows)

    base_w, base_h = rows[0][0][0].size

    # Font and label bar scale with image height
    font_size = int(base_h * 0.06)
    col_header_h = int(font_size * 2.0)
    font = _get_font(font_size)

    # Row label sidebar — dynamic width to fit content
    row_label_font_size = max(16, int(base_h * 0.04))
    row_label_font = _get_font(row_label_font_size)

    def _text_width(f, t):
        try:
            bb = f.getbbox(t)
            return bb[2] - bb[0]
        except AttributeError:
            return f.getlength(t)

    needed = int(base_w * 0.28)
    pad = 24
    if title:
        needed = max(needed, _text_width(font, title) + pad)
    for row in rows:
        label = getattr(row, "label", "")
        needed = max(needed, _text_width(row_label_font, label) + pad)
    row_label_w = needed

    total_w = row_label_w + base_w * n_cols
    total_h = col_header_h + base_h * n_rows

    composite = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(composite)

    # Title in the top-left corner
    draw.rectangle(
        [0, 0, row_label_w, col_header_h],
        fill=(40, 40, 40),
    )
    try:
        bb = font.getbbox(title)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        tw, th = draw.textsize(title, font=font)
    draw.text(
        ((row_label_w - tw) // 2, (col_header_h - th) // 2),
        title,
        fill=(200, 200, 200),
        font=font,
    )

    # Column headers — epoch labels from the first row
    for col_idx, (_, label) in enumerate(rows[0]):
        x0 = row_label_w + col_idx * base_w
        draw.rectangle(
            [x0, 0, x0 + base_w, col_header_h],
            fill=(50, 50, 50),
        )
        label = _truncate_label(label, font, base_w)
        try:
            bb = font.getbbox(label)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)
        draw.text(
            (x0 + (base_w - tw) // 2, (col_header_h - th) // 2),
            label,
            fill=(220, 220, 220),
            font=font,
        )

    # Rows
    for row_idx, row in enumerate(rows):
        y0 = col_header_h + row_idx * base_h

        # Row label sidebar
        draw.rectangle(
            [0, y0, row_label_w, y0 + base_h],
            fill=(40, 40, 40),
        )
        # Use the prompt label stored on the row (passed as row_label)
        row_label = getattr(row, "label", "")
        try:
            bb = row_label_font.getbbox(row_label)
            rw, rh = bb[2] - bb[0], bb[3] - bb[1]
        except AttributeError:
            rw, rh = draw.textsize(row_label, font=row_label_font)
        draw.text(
            (
                (row_label_w - rw) // 2,
                y0 + (base_h - rh) // 2,
            ),
            row_label,
            fill=(180, 180, 180),
            font=row_label_font,
        )

        # Images
        for col_idx, (img, _) in enumerate(row):
            if img.size != (base_w, base_h):
                img = img.resize((base_w, base_h), Image.LANCZOS)
            composite.paste(
                img, (row_label_w + col_idx * base_w, y0)
            )

    composite.save(out_path)
    return out_path


class LabelledRow(list):
    """A list subclass that carries a prompt label for build_composite."""

    def __init__(self, label, items):
        super().__init__(items)
        self.label = label


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

METHODS = ("phash", "histogram", "pixel")


def _deltas_phash(imgs):
    """
    Epoch-to-epoch Hamming distance between pHash digests, normalised
    to 0-1 (max distance for a 64-bit pHash is 64).
    """
    hashes = [imagehash.phash(img) for img in imgs]
    return [
        (hashes[i] - hashes[i + 1]) / 64.0
        for i in range(len(hashes) - 1)
    ]


def _deltas_histogram(imgs):
    """
    Epoch-to-epoch histogram difference, normalised to 0-1.

    Converts each image to greyscale, builds a 256-bin histogram,
    normalises it to a probability distribution, then computes the
    L1 distance between consecutive pairs. L1 on normalised histograms
    is bounded [0, 2], so we divide by 2.
    """
    def hist(img):
        grey = img.convert("L")
        h = grey.histogram()
        total = sum(h)
        return [v / total for v in h]

    hists = [hist(img) for img in imgs]
    return [
        sum(abs(a - b) for a, b in zip(hists[i], hists[i + 1])) / 2.0
        for i in range(len(hists) - 1)
    ]


def _deltas_pixel(imgs):
    """
    Epoch-to-epoch mean absolute pixel difference, normalised to 0-1.

    Resizes all images to match the first, converts to greyscale, then
    computes the average per-pixel absolute difference normalised by 255.
    """
    base_size = imgs[0].size

    def to_pixels(img):
        if img.size != base_size:
            img = img.resize(base_size, Image.LANCZOS)
        grey = img.convert("L")
        return list(grey.get_flattened_data())

    pixel_lists = [to_pixels(img) for img in imgs]
    return [
        sum(
            abs(a - b)
            for a, b in zip(pixel_lists[i], pixel_lists[i + 1])
        ) / (255.0 * len(pixel_lists[i]))
        for i in range(len(pixel_lists) - 1)
    ]


_METRIC_FNS = {
    "phash": _deltas_phash,
    "histogram": _deltas_histogram,
    "pixel": _deltas_pixel,
}

_SERIES_STYLES = {
    # (line colour, dot colour, legend label)
    "phash":     ((100, 180, 255), (255, 220,  80), "pHash"),
    "histogram": ((120, 220, 120), (255, 140,  60), "Histogram"),
    "pixel":     ((220, 120, 180), (180, 255, 180), "Pixel diff"),
}


# ---------------------------------------------------------------------------
# Similarity graph
# ---------------------------------------------------------------------------

def _epoch_label(full_label):
    """
    Extract a short epoch label from a full composite label string.

    e.g. "mymodel-000008 (1.0)" -> "8", "mymodel (1.0)" -> "final".
    """
    stem = re.sub(r'\s*\(.*\)$', '', full_label).strip()
    match = re.search(r'(\d+)$', stem)
    if match:
        return str(int(match.group(1)))
    return "final"


def build_similarity_graph(images_and_labels, out_path, methods=("phash",)):
    """
    Build and save a line graph of epoch-to-epoch image similarity.

    Uses only the first prompt's images since that is the style prompt.
    *methods* is a tuple of names from METHODS; all series share the
    same normalised 0-1 y-axis.
    """
    if len(images_and_labels) < 2:
        print("  Skipping similarity graph: need at least 2 images.")
        return

    imgs, labels = zip(*images_and_labels)

    short_labels = [_epoch_label(lbl) for lbl in labels]
    x_labels = short_labels[1:]
    n_deltas = len(imgs) - 1

    series = {}
    for method in methods:
        print(f"  Computing {method} similarity…")
        series[method] = _METRIC_FNS[method](list(imgs))

    # --- Layout ---
    width = max(800, 120 * n_deltas)
    height = 500
    pad_left = 70
    pad_right = 120
    pad_top = 40
    pad_bottom = 60

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    img_out = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img_out)

    font_small = _get_font(max(14, height // 30))
    font_label = _get_font(max(12, height // 36))

    y_max = 1.0

    def to_xy(idx, val):
        """Convert a (delta_index, normalised_value) pair to pixels."""
        if n_deltas > 1:
            x = pad_left + int(idx * plot_w / (n_deltas - 1))
        else:
            x = pad_left + plot_w // 2
        y = pad_top + plot_h - int(val / y_max * plot_h)
        return x, y

    for grid_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gy = pad_top + plot_h - int(grid_val / y_max * plot_h)
        draw.line(
            [(pad_left, gy), (pad_left + plot_w, gy)],
            fill=(70, 70, 70), width=1,
        )
        draw.text(
            (pad_left - 6, gy),
            f"{grid_val:.2f}",
            fill=(160, 160, 160),
            font=font_label,
            anchor="rm",
        )

    dot_r = max(4, height // 80)
    for method, deltas in series.items():
        line_col, dot_col, _ = _SERIES_STYLES[method]
        points = [to_xy(i, d) for i, d in enumerate(deltas)]

        for i in range(len(points) - 1):
            draw.line(
                [points[i], points[i + 1]], fill=line_col, width=2,
            )

        for i, (px, py) in enumerate(points):
            draw.ellipse(
                [(px - dot_r, py - dot_r), (px + dot_r, py + dot_r)],
                fill=dot_col,
            )
            if len(series) == 1:
                draw.text(
                    (px, py - dot_r - 6),
                    f"{deltas[i]:.3f}",
                    fill=(220, 220, 220),
                    font=font_label,
                    anchor="mb",
                )

    for i in range(n_deltas):
        px, _ = to_xy(i, 0)
        draw.text(
            (px, pad_top + plot_h + 12),
            x_labels[i],
            fill=(180, 180, 180),
            font=font_label,
            anchor="mt",
        )

    legend_x = pad_left + plot_w + 10
    legend_y = pad_top
    for method in methods:
        line_col, _, legend_lbl = _SERIES_STYLES[method]
        draw.rectangle(
            [legend_x, legend_y, legend_x + 16, legend_y + 14],
            fill=line_col,
        )
        draw.text(
            (legend_x + 22, legend_y),
            legend_lbl,
            fill=(200, 200, 200),
            font=font_label,
        )
        legend_y += 24

    draw.text(
        (pad_left + plot_w // 2, 12),
        "Epoch-to-epoch similarity (higher = more change)",
        fill=(210, 210, 210),
        font=font_small,
        anchor="mt",
    )

    img_out.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(config, lora_dir, dry_run=False, overwrite_images=False,
             overwrite_composite=False, overwrite_graph=False,
             overwrite_json=False, methods=("phash",),
             auto_trigger=False):
    """Run the full evaluation pipeline for all LoRAs in *lora_dir*."""
    api_url = config["comfyui_url"]
    prompts = config["prompts"]
    global_negative = config.get("negative_prompt", "")
    lora_weight = config.get("lora_weight", 1.0)
    # Config trigger words are only used when auto_trigger is off.
    base_trigger_words = config.get("trigger_words", "")

    workflow_path = config["workflow_file"]
    workflow = load_workflow(workflow_path)

    lora_dir = os.path.abspath(lora_dir)

    loras = find_loras(lora_dir)
    if not loras:
        print(f"No .safetensors files found in: {lora_dir}")
        return

    print(f"Found {len(loras)} LoRA(s) in {lora_dir}")
    print(f"Running {len(prompts)} prompt(s) per LoRA")

    # composite_rows[prompt_index] = LabelledRow of (img, epoch_label)
    composite_rows = [
        LabelledRow(p["label"], []) for p in prompts
    ]

    epoch_labels = _compute_epoch_labels(loras)

    # first_prompt_entries used for the similarity graph
    first_prompt_entries = []
    # composite title: use ss_output_name from first LoRA if available
    composite_title = os.path.basename(os.path.normpath(lora_dir))

    for lora_idx, lora_path in enumerate(loras):
        # Always parse the header; fields not needed fall back silently.
        lora_meta = read_safetensors_metadata(lora_path)

        if lora_meta["name"] and lora_idx == 0:
            composite_title = lora_meta["name"]

        # Prefer header epoch over filename-parsed label.
        if lora_meta["epoch"] is not None:
            epoch_label = str(lora_meta["epoch"])
        else:
            epoch_label = epoch_labels[lora_idx]

        if auto_trigger:
            trigger_words = lora_meta["trigger"]
            if trigger_words:
                print(
                    f"  Auto-trigger: '{trigger_words}'"
                    f" (from header of"
                    f" {os.path.basename(lora_path)})"
                )
            else:
                print(
                    "  Auto-trigger: no @-prefixed tag found in header,"
                    " continuing without trigger word."
                )
        else:
            trigger_words = base_trigger_words

        syntax = lora_syntax(lora_path, lora_weight, trigger_words)

        print(f"\nProcessing: {os.path.basename(lora_path)}")

        if dry_run:
            print("  [dry-run] Skipping API call.")
            continue

        for p_idx, prompt_cfg in enumerate(prompts):
            positive = f"{prompt_cfg['positive']}, {syntax}"
            # Per-prompt negative takes priority over the global one
            negative = prompt_cfg.get("negative", global_negative)
            print(f"  Prompt [{prompt_cfg['label']}]: {positive}")

            dest = image_path_for_lora(lora_dir, lora_path, p_idx)

            if os.path.exists(dest) and not overwrite_images:
                print(
                    f"  Skipping: {os.path.basename(dest)} already exists."
                )
                img = Image.open(dest).copy()
            else:
                wf = inject_prompts(workflow, config, positive, negative)
                client_id = str(uuid.uuid4())
                prompt_id = queue_prompt(api_url, wf, client_id)
                print(f"  Queued prompt {prompt_id}")

                history = wait_for_prompt(api_url, prompt_id)
                images = collect_images(api_url, history)

                if not images:
                    print("  Warning: no images returned.")
                    continue

                img = images[0]
                saved = save_individual(img, lora_dir, lora_path, p_idx)
                print(f"  Saved: {saved}")

            # Write metadata pointing at the first prompt's image only
            if p_idx == 0:
                write_metadata(
                    lora_path, dest,
                    trigger_words=trigger_words,
                    overwrite=overwrite_json,
                )

            composite_rows[p_idx].append((img, epoch_label))

        # Track first prompt image for similarity graph
        if composite_rows[0]:
            first_prompt_entries.append(composite_rows[0][-1])

    if any(composite_rows) and not dry_run:
        composite_path = os.path.join(lora_dir, "_composite.png")
        if overwrite_composite or not os.path.exists(composite_path):
            build_composite(
                composite_rows, composite_path,
                title=composite_title,
            )
            print(f"\nComposite saved: {composite_path}")
        else:
            print(
                "\nSkipping composite"
                " (already exists, use -o composite to regen)"
            )

        graph_path = os.path.join(lora_dir, "_similarity.png")
        if overwrite_graph or not os.path.exists(graph_path):
            build_similarity_graph(
                first_prompt_entries, graph_path, methods=methods
            )
            print(f"Similarity graph saved: {graph_path}")
        else:
            print("Skipping graph (already exists, use -o graph to regen)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LoRAs via the ComfyUI API.",
    )
    parser.add_argument(
        "lora_dir",
        help="Directory containing .safetensors files.",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to the JSON config file"
             " (default: ~/.config/lora-eval/config.json).",
    )
    parser.add_argument(
        "-w", "--workflow",
        default=None,
        help="Path to the ComfyUI workflow JSON file"
             " (default: path from config or"
             " ~/.config/lora-eval/<filename>).",
    )
    parser.add_argument(
        "-p", "--preset",
        default=None,
        metavar="NAME",
        help="Use a named prompt preset from 'prompt_presets' in the"
             " config, overriding the default 'prompts' list.",
    )
    parser.add_argument(
        "-a", "--auto-trigger",
        action="store_true",
        help="Append the most-frequent @-prefixed tag from each LoRA's"
             " safetensors header to the prompt. Silently skips LoRAs"
             " with no @ tag. Overrides --trigger-words and the config"
             " value. The header is always read for name and epoch.",
    )
    parser.add_argument(
        "-t", "--trigger-words",
        default=None,
        help="Override trigger words from config.",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Validate config and list LoRAs without calling the API.",
    )
    parser.add_argument(
        "-o", "--overwrite",
        default="",
        metavar="ITEMS",
        help="Comma-separated items to regenerate: images,composite,graph,"
             "json, or all. (default: none)",
    )
    parser.add_argument(
        "-m", "--method",
        nargs="+",
        choices=METHODS + ("all",),
        default=["all"],
        metavar="METHOD",
        help=(
            "Similarity method(s): phash, histogram, pixel, all."
            " (default: all)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.lora_dir):
        print(
            f"Error: '{args.lora_dir}' is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = _resolve_config_path(args.config)

    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.preset is not None:
        presets = config.get("prompt_presets", {})
        if args.preset not in presets:
            available = ", ".join(presets) if presets else "(none defined)"
            print(
                f"Error: preset '{args.preset}' not found in config."
                f" Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        config["prompts"] = presets[args.preset]

    if args.workflow is not None:
        config["workflow_file"] = args.workflow
    # --auto-trigger takes priority; skip the config/CLI trigger words.
    if args.auto_trigger:
        config["trigger_words"] = ""
    elif args.trigger_words is not None:
        config["trigger_words"] = args.trigger_words

    config["workflow_file"] = _resolve_workflow_path(
        config["workflow_file"]
    )

    if args.overwrite == "all":
        ow = {"images": True, "composite": True, "graph": True, "json": True}
    elif args.overwrite:
        ow = {k: False for k in ("images", "composite", "graph", "json")}
        for item in args.overwrite.split(","):
            if item not in ow:
                print(f"Error: unknown overwrite item '{item}'",
                      file=sys.stderr)
                sys.exit(1)
            ow[item] = True
    else:
        ow = {k: False for k in ("images", "composite", "graph", "json")}

    methods = list(METHODS) if "all" in args.method else args.method

    evaluate(
        config, args.lora_dir,
        dry_run=args.dry_run,
        overwrite_images=ow["images"],
        overwrite_composite=ow["composite"],
        overwrite_graph=ow["graph"],
        overwrite_json=ow["json"],
        methods=tuple(methods),
        auto_trigger=args.auto_trigger,
    )


if __name__ == "__main__":
    main()
