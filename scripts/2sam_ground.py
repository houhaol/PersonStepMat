import os
import json
import cv2
import numpy as np
import argparse
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

def load_feet_prompts(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def run_sam_on_points(image_path, prompt_path, predictor, output_mask_dir, overlay_dir=None):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    prompts = load_feet_prompts(prompt_path)

    h, w, _ = image_rgb.shape
    left_mask = np.zeros((h, w), dtype=bool)
    right_mask = np.zeros((h, w), dtype=bool)

    for idx, prompt in enumerate(prompts):
        point = np.array([prompt["point"]])
        label = np.array([1])  # foreground

        masks, scores, logits = predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=False
        )

        mask = masks[0]  # shape: (H, W), bool

        # Combine masks per foot
        if prompt["foot"] == "left":
            left_mask = np.logical_or(left_mask, mask)
        elif prompt["foot"] == "right":
            right_mask = np.logical_or(right_mask, mask)

        # Optional overlay for individual points
        if overlay_dir:
            overlay = image_bgr.copy()
            x, y = prompt["point"]
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(overlay, prompt["foot"], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([0, 255, 255])).astype(np.uint8)
            overlay_path = os.path.join(overlay_dir, os.path.basename(prompt_path).replace("_feet_prompt.json", f"_{prompt['foot']}_overlay.png"))
            cv2.imwrite(overlay_path, overlay)

    # Final fused mask
    # fused_mask = np.logical_and(left_mask, right_mask).astype(np.uint8)
    fused_mask = np.logical_or(left_mask, right_mask).astype(np.uint8)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    fused_mask_clean = cv2.morphologyEx(fused_mask * 255, cv2.MORPH_OPEN, kernel)

    # Save final mask
    mask_name = os.path.basename(prompt_path).replace("_feet_prompt.json", "_fused_mask.png")
    mask_path = os.path.join(output_mask_dir, mask_name)
    cv2.imwrite(mask_path, fused_mask_clean)
    print(f"✅ Saved fused cleaned mask: {mask_path}")
    
    # Save masked region on original RGB image
    masked_rgb = image_bgr.copy()
    masked_rgb[fused_mask_clean == 0] = 0  # Zero out background

    rgb_masked_path = os.path.join(output_mask_dir, mask_name.replace("_fused_mask.png", "_masked_rgb.png"))
    cv2.imwrite(rgb_masked_path, masked_rgb)
    print(f"🖼️ Saved masked RGB image: {rgb_masked_path}")

    # Optional: save overlay of combined mask
    if overlay_dir:
        overlay = image_bgr.copy()
        overlay[fused_mask_clean > 0] = (0.4 * overlay[fused_mask_clean > 0] + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
        overlay_path = os.path.join(overlay_dir, mask_name.replace(".png", "_overlay.png"))
        cv2.imwrite(overlay_path, overlay)
        print(f"✅ Saved overlay of combined mask: {overlay_path}")


def main(args):
    os.makedirs(args.output_mask_dir, exist_ok=True)
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)

    # Load SAM model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    predictor = SamPredictor(sam)

    # Process each pair of (image, prompt)
    for fname in sorted(os.listdir(args.prompt_dir)):
        if not fname.endswith("_feet_prompt.json"):
            continue

        prompt_path = os.path.join(args.prompt_dir, fname)
        image_name = fname.replace("_feet_prompt.json", ".png")
        image_path = os.path.join(args.image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue

        run_sam_on_points(
            image_path=image_path,
            prompt_path=prompt_path,
            predictor=predictor,
            output_mask_dir=args.output_mask_dir,
            overlay_dir=args.overlay_dir
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM segmentation based on feet prompts.")
    parser.add_argument("--image_dir", required=True, help="Directory of original images (e.g., frames_*.png)")
    parser.add_argument("--prompt_dir", required=True, help="Directory of _feet_prompt.json files")
    parser.add_argument("--output_mask_dir", required=True, help="Directory to save binary mask PNGs")
    parser.add_argument("--overlay_dir", help="Optional: directory to save visualization overlays")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM model checkpoint")
    parser.add_argument("--model_type", default="vit_b", help="SAM model type: vit_b / vit_l / vit_h")
    parser.add_argument("--device", default="cuda", help="Device to run SAM (cuda or cpu)")
    args = parser.parse_args()

    main(args)
