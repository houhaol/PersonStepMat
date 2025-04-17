import os
import cv2
import numpy as np
import argparse
import json

def get_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, w, h

def pad_bbox(x, y, w, h, scale, image_shape):
    cx, cy = x + w / 2, y + h / 2
    new_w, new_h = w * scale, h * scale
    x1 = int(max(0, cx - new_w / 2))
    y1 = int(max(0, cy - new_h / 2))
    x2 = int(min(image_shape[1], cx + new_w / 2))
    y2 = int(min(image_shape[0], cy + new_h / 2))
    return x1, y1, x2, y2

def process_all_frames(frame_dir, mask_dir, output_dir, padding_scale):
    os.makedirs(output_dir, exist_ok=True)
    offsets = {}

    frame_files = sorted(os.listdir(frame_dir))
    for fname in frame_files:
        if not fname.endswith(".png"):
            continue

        frame_path = os.path.join(frame_dir, fname)
        mask_path = os.path.join(mask_dir, fname.replace("frame", "mask").replace(".png", ".npy"))
        out_path = os.path.join(output_dir, fname)

        if not os.path.exists(mask_path):
            print(f"⚠️ Missing mask: {mask_path}")
            continue

        frame = cv2.imread(frame_path)
        mask = np.load(mask_path)

        bbox = get_bbox_from_mask(mask)
        if bbox is None:
            print(f"❌ No person in mask: {fname}")
            continue

        x, y, w, h = bbox
        x1, y1, x2, y2 = pad_bbox(x, y, w, h, padding_scale, frame.shape)
        cropped = frame[y1:y2, x1:x2]
        cv2.imwrite(out_path, cropped)

        offsets[fname] = {'x1': x1, 'y1': y1}
        print(f"✅ Saved: {out_path} | Offset: ({x1}, {y1})")

    # Save crop offsets for later use
    offset_path = os.path.join(output_dir, "crop_offsets.json")
    with open(offset_path, "w") as f:
        json.dump(offsets, f)
    print(f"\n📄 Saved offset metadata to: {offset_path}")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop frames using masks and save padded bounding boxes.")
    parser.add_argument("--frame_dir", required=True, help="Path to PNG frame directory.")
    parser.add_argument("--mask_dir", required=True, help="Path to .npy mask directory.")
    parser.add_argument("--output_dir", required=True, help="Path to save cropped frames.")
    parser.add_argument("--padding_scale", type=float, default=1.2, help="Padding scale around bbox (default=1.2)")

    args = parser.parse_args()

    process_all_frames(
        frame_dir=args.frame_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        padding_scale=args.padding_scale
    )
