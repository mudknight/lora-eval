#!/usr/bin/env python3
"""
main.py - Evaluate Stable Diffusion LoRAs via the ComfyUI API.

For each .safetensors file found in the target directory, injects LoRA
syntax (and optional trigger words) into a ComfyUI workflow, queues
generation, saves individual images, and produces a composite grid
showing all epochs side-by-side with labels.
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
        "positive_prompt",
        "negative_prompt",
        "workflow_file",
        "positive_node_id",
        "positive_text_field",
        "negative_node_id",
        "negative_text_field",
    ]
    for key in required:
        if key not in config:
            raise ValueError(f"Config missing required key: '{key}'")

    return config


def load_workflow(workflow_path):
    """Load a ComfyUI API-format workflow JSON file."""
    with open(workflow_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# LoRA discovery
# ---------------------------------------------------------------------------

def _lora_sort_key(path):
    """
    Sort key that places epoch-numbered files in numeric order and puts
    any unnumbered file (the final merged epoch) last.

    e.g. model-000001, model-000002, ..., model  (no number -> last)
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r'(\d+)$', stem)
    # Numbered files sort by their integer value; unnumbered files sort
    # after all numbered ones by using float('inf')
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


def lora_syntax(safetensors_path, weight, trigger_words):
    """
    Build the LoRA syntax string appended to the positive prompt.

    Uses only the filename stem (no directory, no extension), as
    ComfyUI expects the name relative to the models/loras folder.
    Format: <lora:NAME:WEIGHT>
    """
    stem = os.path.splitext(os.path.basename(safetensors_path))[0]
    tag = f"<lora:{stem}:{weight}>"
    if trigger_words:
        return f"{trigger_words}, {tag}"
    return tag


# ---------------------------------------------------------------------------
# Workflow manipulation
# ---------------------------------------------------------------------------

def inject_prompts(workflow, config, positive_text, negative_text):
    """
    Return a deep copy of *workflow* with positive/negative text injected.

    Node IDs and field names come from the config so this works with any
    workflow that exposes its prompt nodes.
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

def image_path_for_lora(out_dir, lora_path):
    """Return the expected .png path for a given LoRA file."""
    stem = os.path.splitext(os.path.basename(lora_path))[0]
    return os.path.join(out_dir, f"{stem}.png")


def save_individual(image, out_dir, lora_path):
    """
    Save a single PIL image alongside the .safetensors file.

    Filename matches the LoRA stem so tools like LoRA Manager can locate
    the preview by swapping the extension on the .safetensors filename.
    """
    dest = image_path_for_lora(out_dir, lora_path)
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


def write_metadata(lora_path, image_path, overwrite=False):
    """
    Create or update the LoRA Manager metadata JSON for *lora_path*.

    On first run the full template is written. On subsequent runs only
    ``preview_url`` is updated so user edits are not clobbered,
    unless *overwrite* is True, in which case the file is recreated.
    """
    stem = os.path.splitext(os.path.basename(lora_path))[0]
    meta_path = os.path.join(
        os.path.dirname(lora_path), f"{stem}.metadata.json"
    )

    if os.path.exists(meta_path) and not overwrite:
        # File already exists — update preview_url only
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["preview_url"] = image_path
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
            "civitai": {},
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
    """
    Try to load a TTF font; fall back to the PIL default if unavailable.
    """
    # Common font locations across Linux / macOS / Windows
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
    """
    Truncate *label* with an ellipsis if it exceeds *max_width* pixels.

    Keeps the font size fixed so the bar height always matches the text.
    """
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


def build_composite(images_and_labels, out_path):
    """
    Build and save a horizontal composite of *images_and_labels*.

    Each entry is a (PIL.Image, str) tuple. A text label is drawn above
    each image. Font size and label bar height scale with the image height
    so labels are legible at any output resolution. All images are assumed
    to be the same size; if not they are resized to match the first.
    """
    if not images_and_labels:
        return

    base_w, base_h = images_and_labels[0][0].size

    # Font size: 6% of image height, label bar is font + equal padding.
    # Both scale together so the text always fills the bar.
    font_size = int(base_h * 0.06)
    label_height = int(font_size * 2.0)
    font = _get_font(font_size)

    total_w = base_w * len(images_and_labels)
    total_h = base_h + label_height

    composite = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(composite)

    for idx, (img, label) in enumerate(images_and_labels):
        # Resize if necessary
        if img.size != (base_w, base_h):
            img = img.resize((base_w, base_h), Image.LANCZOS)

        x_offset = idx * base_w

        # Draw label background strip
        draw.rectangle(
            [x_offset, 0, x_offset + base_w, label_height],
            fill=(50, 50, 50),
        )

        label = _truncate_label(label, font, base_w)

        try:
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(label, font=font)

        # Centre text horizontally and vertically in the label bar
        text_x = x_offset + (base_w - text_w) // 2
        text_y = (label_height - text_h) // 2
        draw.text((text_x, text_y), label, fill=(220, 220, 220), font=font)

        # Paste the image below the label strip
        composite.paste(img, (x_offset, label_height))

    composite.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Similarity graph
# ---------------------------------------------------------------------------

def _epoch_label(full_label):
    """
    Extract a short epoch label from a full label string.

    Pulls the trailing numeric portion from the stem and strips leading
    zeros, e.g. "mymodel-000008 (1.0)" -> "8". Files with no trailing
    number (the final merged epoch) are labelled "final".
    """
    stem = re.sub(r'\s*\(.*\)$', '', full_label).strip()
    match = re.search(r'(\d+)$', stem)
    if match:
        return str(int(match.group(1)))
    return "final"


def build_similarity_graph(images_and_labels, out_path):
    """
    Build and save a line graph of epoch-to-epoch perceptual hash distance.

    Hashes each image with pHash and plots the Hamming distance between
    each consecutive pair. A distance of 0 means identical; 64 is the
    maximum for a 64-bit pHash.
    """
    if len(images_and_labels) < 2:
        print("  Skipping similarity graph: need at least 2 images.")
        return

    imgs, labels = zip(*images_and_labels)

    # Compute pHash for every image
    hashes = [imagehash.phash(img) for img in imgs]

    # Delta distance between each consecutive pair
    deltas = [
        hashes[i] - hashes[i + 1]
        for i in range(len(hashes) - 1)
    ]

    # Short labels for every epoch; the x-axis label for delta i shows
    # the left epoch of each pair, e.g. "8" for the 8->9 transition.
    # If the last epoch is unlabeled ("final"), we override the rightmost
    # label with the inferred next number (max numbered epoch + 1) so
    # the final epoch shows up on the x-axis aligned with its data point.
    short_labels = [_epoch_label(lbl) for lbl in labels]
    x_labels = short_labels[:-1]
    if x_labels and short_labels and short_labels[-1] == "final":
        numbered = [int(s) for s in short_labels if s.isdigit()]
        if numbered:
            x_labels[-1] = str(max(numbered) + 1)

    # --- Layout constants ---
    width = max(800, 120 * len(deltas))
    height = 500
    pad_left = 80
    pad_right = 40
    pad_top = 40
    pad_bottom = 60

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    img_out = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img_out)

    font_small = _get_font(max(14, height // 30))
    font_label = _get_font(max(12, height // 36))

    # Y axis: fixed 0-64 range (full pHash range) for consistency
    y_max = 64

    def to_xy(idx, val):
        """Convert a (delta_index, distance) pair to pixel coords."""
        if len(deltas) > 1:
            x = pad_left + int(idx * plot_w / (len(deltas) - 1))
        else:
            x = pad_left + plot_w // 2
        y = pad_top + plot_h - int(val / y_max * plot_h)
        return x, y

    # Grid lines at 0, 16, 32, 48, 64
    for grid_val in range(0, y_max + 1, 16):
        gx0 = pad_left
        gx1 = pad_left + plot_w
        gy = pad_top + plot_h - int(grid_val / y_max * plot_h)
        draw.line([(gx0, gy), (gx1, gy)], fill=(70, 70, 70), width=1)
        draw.text(
            (pad_left - 8, gy),
            str(grid_val),
            fill=(160, 160, 160),
            font=font_label,
            anchor="rm",
        )

    # Plot line and points
    points = [to_xy(i, d) for i, d in enumerate(deltas)]

    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=(100, 180, 255), width=3)

    dot_r = max(4, height // 80)
    for i, (px, py) in enumerate(points):
        draw.ellipse(
            [(px - dot_r, py - dot_r), (px + dot_r, py + dot_r)],
            fill=(255, 220, 80),
        )
        # Distance value above each point
        draw.text(
            (px, py - dot_r - 6),
            str(deltas[i]),
            fill=(220, 220, 220),
            font=font_label,
            anchor="mb",
        )

    # X-axis labels: single-line epoch numbers centred under each point
    for i, (px, _) in enumerate(points):
        draw.text(
            (px, pad_top + plot_h + 12),
            x_labels[i],
            fill=(180, 180, 180),
            font=font_label,
            anchor="mt",
        )

    # Chart title
    draw.text(
        (width // 2, 12),
        "Epoch-to-epoch pHash distance (lower = more similar)",
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
             overwrite_json=False):
    """Run the full evaluation pipeline for all LoRAs in *lora_dir*."""
    api_url = config["comfyui_url"]
    base_positive = config["positive_prompt"]
    base_negative = config["negative_prompt"]
    lora_weight = config.get("lora_weight", 1.0)
    trigger_words = config.get("trigger_words", "")
    images_per_lora = config.get("images_per_lora", 1)

    workflow_path = config["workflow_file"]
    workflow = load_workflow(workflow_path)

    loras = find_loras(lora_dir)
    if not loras:
        print(f"No .safetensors files found in: {lora_dir}")
        return

    print(f"Found {len(loras)} LoRA(s) in {lora_dir}")

    # Each entry: (PIL.Image, label_str) for the composite and graph
    composite_entries = []

    for lora_path in loras:
        syntax = lora_syntax(lora_path, lora_weight, trigger_words)
        positive = f"{base_positive}, {syntax}"
        negative = base_negative

        print(f"\nProcessing: {os.path.basename(lora_path)}")
        print(f"  Positive: {positive}")

        if dry_run:
            print("  [dry-run] Skipping API call.")
            continue

        dest = image_path_for_lora(lora_dir, lora_path)
        stem = os.path.splitext(os.path.basename(lora_path))[0]
        label = f"{stem} ({lora_weight})"

        # If the image already exists and overwrite_images is off, reuse
        # it for the composite without hitting the API at all.
        if os.path.exists(dest) and not overwrite_images:
            print(f"  Skipping: {os.path.basename(dest)} already exists.")
            write_metadata(lora_path, dest, overwrite=overwrite_json)
            composite_entries.append((Image.open(dest).copy(), label))
            continue

        wf = inject_prompts(workflow, config, positive, negative)
        client_id = str(uuid.uuid4())

        images = []
        for img_index in range(images_per_lora):
            prompt_id = queue_prompt(api_url, wf, client_id)
            print(f"  Queued prompt {prompt_id} (image {img_index + 1})")

            history = wait_for_prompt(api_url, prompt_id)
            batch = collect_images(api_url, history)

            if not batch:
                print("  Warning: no images returned for this prompt.")
                continue

            images.extend(batch)

        if not images:
            continue

        saved = save_individual(images[0], lora_dir, lora_path)
        print(f"  Saved: {saved}")
        write_metadata(lora_path, saved, overwrite=overwrite_json)

        composite_entries.append((images[0], label))

    if composite_entries and not dry_run:
        composite_path = os.path.join(lora_dir, "_composite.png")
        if overwrite_composite or not os.path.exists(composite_path):
            build_composite(composite_entries, composite_path)
            print(f"\nComposite saved: {composite_path}")
        else:
            print(f"\nSkipping composite (already exists, use -oc to regen)")

        graph_path = os.path.join(lora_dir, "_similarity.png")
        if overwrite_graph or not os.path.exists(graph_path):
            build_similarity_graph(composite_entries, graph_path)
            print(f"Similarity graph saved: {graph_path}")
        else:
            print(f"Skipping graph (already exists, use -og to regen)")


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
        "--config",
        default="config.json",
        help="Path to the JSON config file (default: config.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and list LoRAs without calling the API.",
    )

    # Granular overwrite flags
    parser.add_argument(
        "-oi", "--overwrite-images",
        action="store_true",
        help="Regenerate images even if they already exist.",
    )
    parser.add_argument(
        "-oc", "--overwrite-composite",
        action="store_true",
        help="Rebuild the composite even if it already exists.",
    )
    parser.add_argument(
        "-og", "--overwrite-graph",
        action="store_true",
        help="Rebuild the similarity graph even if it already exists.",
    )
    parser.add_argument(
        "-oj", "--overwrite-json",
        action="store_true",
        help="Recreate metadata JSON files even if they already exist.",
    )
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="Shorthand for -oi -oc -og -oj (overwrite everything).",
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

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    # -o is shorthand for all four overwrite flags
    overwrite_all = args.overwrite

    evaluate(
        config, args.lora_dir,
        dry_run=args.dry_run,
        overwrite_images=args.overwrite_images or overwrite_all,
        overwrite_composite=args.overwrite_composite or overwrite_all,
        overwrite_graph=args.overwrite_graph or overwrite_all,
        overwrite_json=args.overwrite_json or overwrite_all,
    )


if __name__ == "__main__":
    main()
