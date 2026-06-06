#!/usr/bin/env python3
"""
evaluate_loras.py - Evaluate Stable Diffusion LoRAs via the ComfyUI API.

For each .safetensors file found in the target directory, injects LoRA
syntax (and optional trigger words) into a ComfyUI workflow, queues
generation, saves individual images, and produces a composite grid
showing all epochs side-by-side with labels.
"""

import argparse
import copy
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

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

def find_loras(directory):
    """
    Return a sorted list of .safetensors paths in *directory*.
    Sorting is lexicographic, which puts epoch-numbered files in order
    when the naming convention is consistent (e.g. model-000001.safetensors).
    """
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".safetensors")
    ]
    return sorted(files)


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

def save_individual(image, out_dir, lora_path, index):
    """
    Save a single PIL image alongside the .safetensors file.

    Filename: <lora_stem>_<index>.png
    """
    stem = os.path.splitext(os.path.basename(lora_path))[0]
    filename = f"{stem}.png"
    dest = os.path.join(out_dir, filename)
    image.save(dest)
    return dest


# ---------------------------------------------------------------------------
# Composite image
# ---------------------------------------------------------------------------

def _get_font(size):
    """
    Try to load a TTF font; fall back to the PIL default if unavailable.
    """
    # Common font locations across Linux / macOS / Windows
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
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


def _fit_font(label, max_width, target_size, min_size=32):
    """
    Return a (font, size) whose rendered *label* fits within *max_width*.

    Starts at *target_size* and steps down by 2 until it fits, stopping
    at *min_size* so text is never invisibly small.
    """
    size = target_size
    while size >= min_size:
        font = _get_font(size)
        try:
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
        except AttributeError:
            # Rough estimate for very old Pillow
            text_w = int(size * len(label) * 0.6)
        if text_w <= max_width * 0.92:
            return font, size
        size -= 2
    return _get_font(min_size), min_size


def build_composite(images_and_labels, out_path):
    """
    Build and save a horizontal composite of *images_and_labels*.

    Each entry is a (PIL.Image, str) tuple. A text label is drawn above
    each image. Font size and label bar height scale with the image width
    so labels are legible at any output resolution. All images are assumed
    to be the same size; if not they are resized to match the first.
    """
    if not images_and_labels:
        return

    base_w, base_h = images_and_labels[0][0].size

    # Font target: ~5% of image width, minimum 24px for legibility
    target_font_size = max(24, base_w // 20)
    # Label bar: enough vertical room for the font plus comfortable padding
    label_height = int(target_font_size * 2.2)

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

        # Largest font that fits within the column width
        font, _ = _fit_font(label, base_w, target_font_size)

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
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(config, lora_dir, dry_run=False):
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

    # Each entry: (PIL.Image, label_str) for the composite
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

        wf = inject_prompts(workflow, config, positive, negative)
        client_id = str(uuid.uuid4())

        # Track whether we've added this LoRA to the composite yet
        added_to_composite = False

        for img_index in range(images_per_lora):
            prompt_id = queue_prompt(api_url, wf, client_id)
            print(f"  Queued prompt {prompt_id} (image {img_index + 1})")

            history = wait_for_prompt(api_url, prompt_id)
            images = collect_images(api_url, history)

            if not images:
                print("  Warning: no images returned for this prompt.")
                continue

            for img in images:
                saved = save_individual(
                    img, lora_dir, lora_path, img_index
                )
                print(f"  Saved: {saved}")

            # Add only the first image of the first generation to the
            # composite, regardless of batch size or node count
            if not added_to_composite:
                stem = os.path.splitext(
                    os.path.basename(lora_path)
                )[0]
                label = f"{stem} ({lora_weight})"
                composite_entries.append((images[0], label))
                added_to_composite = True

    if composite_entries and not dry_run:
        composite_path = os.path.join(lora_dir, "_composite.png")
        build_composite(composite_entries, composite_path)
        print(f"\nComposite saved: {composite_path}")


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
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.lora_dir):
        print(f"Error: '{args.lora_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    evaluate(config, args.lora_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
