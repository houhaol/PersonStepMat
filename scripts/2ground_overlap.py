import os
import json
import numpy as np
import cv2
import argparse
from pycocotools import mask as maskUtils

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def get_touched_ground_mask(json_path, keypoints, radius=20, return_patch=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    height = data.get("img_height", 720)
    width = data.get("img_width", 1280)
    touched_mask = np.zeros((height, width), dtype=np.uint8)
    patches = []

    for pt in keypoints:
        px, py = pt
        x1, x2 = max(0, px - radius), min(width, px + radius + 1)
        y1, y2 = max(0, py - radius), min(height, py + radius + 1)
        foot_patch = np.zeros((height, width), dtype=np.uint8)
        foot_patch[y1:y2, x1:x2] = 1

        best_iou = 0
        best_ann = None
        best_mask = None
        best_bbox = None

        for ann in data["annotations"]:
            if ann["class_name"] == "walkway":
                binary_mask = maskUtils.decode(ann["segmentation"])
                patch = binary_mask[y1:y2, x1:x2]
                if np.any(patch):
                    iou = compute_iou(foot_patch, binary_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_ann = ann
                        best_mask = binary_mask
                        best_bbox = [x1, y1, x2, y2]
        if best_iou > 0:
            touched_mask = np.logical_or(touched_mask, best_mask).astype(np.uint8)
            if return_patch:
                patches.append({
                    "point": (px, py),
                    "patch": best_mask[y1:y2, x1:x2].tolist(),
                    "bbox": best_bbox,
                    "iou": best_iou,
                    "matched_segment": best_ann
                })
            break

    if return_patch:
        return touched_mask, patches
    return touched_mask

def process_frame(ground_json_path, offset_json, keypoint_json_path, mask_output_path=None):
    with open(offset_json, "r") as f:
        crop_offsets = json.load(f)
    with open(keypoint_json_path, "r") as f:
        data = json.load(f)

    people = data.get("people", [])
    foot_prompts = []
    fname = os.path.basename(keypoint_json_path)
    img_fname = fname.replace("_keypoints.json", ".png")
    offset = crop_offsets.get(img_fname, {"x1": 0, "y1": 0})
    dx, dy = offset["x1"], offset["y1"]

    foot_points = []
    for person in people:
        keypoints = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)
        keypoints[:, 0] += dx
        keypoints[:, 1] += dy

        for side, indices in [("left", [21, 19, 20]), ("right", [24, 22, 23])]:
            for idx in indices:
                x, y, conf = keypoints[idx]
                if conf > 0.1:
                    px, py = int(x), int(y)
                    foot_points.append((px, py))
                    foot_prompts.append({
                        "foot": side,
                        "point": [px, py]
                    })
                    break

    if mask_output_path:
        touched_mask, patches = get_touched_ground_mask(ground_json_path, foot_points, return_patch=True)
        for patch in patches:
            x1, y1, x2, y2 = patch["bbox"]
            cv2.rectangle(touched_mask, (x1, y1), (x2 - 1, y2 - 1), color=128, thickness=2)
        cv2.imwrite(mask_output_path, touched_mask * 255)

    return foot_prompts

def batch_process(kpt_dir, ground_dir, offset_json, output_prompt_dir=None, output_mask_dir=None):
    if output_prompt_dir:
        os.makedirs(output_prompt_dir, exist_ok=True)
    if output_mask_dir:
        os.makedirs(output_mask_dir, exist_ok=True)

    for fname in sorted(os.listdir(ground_dir)):
        if not fname.startswith("results_frame_") or not fname.endswith(".json"):
            continue
        frame_id = fname.replace("results_frame_", "").replace(".json", "")
        ground_json_path = os.path.join(ground_dir, fname)
        keypoint_json_path = os.path.join(kpt_dir, f"frame_{frame_id}_keypoints.json")
        if not os.path.exists(keypoint_json_path):
            print(f"⚠️ Missing keypoint file for frame {frame_id}")
            continue
        mask_output_path = os.path.join(output_mask_dir, f"frame_{frame_id}_ground_mask.png") if output_mask_dir else None
        prompt_output_path = os.path.join(output_prompt_dir, f"frame_{frame_id}_feet_prompt.json") if output_prompt_dir else None

        foot_data = process_frame(
            ground_json_path=ground_json_path,
            offset_json=offset_json,
            keypoint_json_path=keypoint_json_path,
            mask_output_path=mask_output_path
        )

        if prompt_output_path:
            with open(prompt_output_path, "w") as f:
                json.dump(foot_data, f, indent=2)
            print(f"✅ Frame {frame_id}: prompts → {prompt_output_path}")
        if mask_output_path:
            print(f"🖼  Frame {frame_id}: touched mask → {mask_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch detect foot-ground overlap using keypoints and segmentation mask.")
    parser.add_argument("--batch_keypoint_dir", required=True, help="Directory containing keypoint JSONs for batch processing")
    parser.add_argument("--batch_ground_dir", required=True, help="Directory containing ground JSONs for batch processing")
    parser.add_argument("--offset_json", required=True, help="Path to the crop offsets JSON file")
    parser.add_argument("--output_prompt_dir", help="Directory to save batch prompts (optional)")
    parser.add_argument("--output_mask_dir", help="Directory to save batch ground masks (optional)")
    args = parser.parse_args()

    batch_process(
        kpt_dir=args.batch_keypoint_dir,
        ground_dir=args.batch_ground_dir,
        offset_json=args.offset_json,
        output_prompt_dir=args.output_prompt_dir,
        output_mask_dir=args.output_mask_dir
    )
