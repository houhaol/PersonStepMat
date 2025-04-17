import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

# Predefined color palette
MATERIAL_COLORS = {
    "TerracottaTile": (80, 150, 220),
    "Wood": (42, 42, 165),
    "Dirt": (42, 42, 42),
    "Metal": (0, 215, 255),
    "Concrete": (180, 180, 180),
    "VoidAcrylicPolymer": (128, 128, 0),
    "Grass": (60, 179, 113),
    "CeramicTiles": (255, 165, 0),
    "Rubberplayground": (255, 128, 0),
    "Asphalt": (128, 64, 128),
    "RedCycling": (60, 20, 220),
    "ConcretePavers": (100, 100, 200),
    "GlazedBricks": (255, 255, 0)
}

def load_predictions(json_file):
    with open(json_file, 'r') as f:
        return {
            entry["image"].replace("_masked_rgb.png", ".png"): entry
            for entry in json.load(f)
        }

def overlay_mask_and_label(frame, mask, label, color, alpha=0.5):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.uint8)

    # Create a copy of frame for overlay
    overlayed = frame.copy()

    # Apply color only where mask is 1
    color_mask = np.zeros_like(frame)
    color_mask[mask == 1] = color

    # Blend only on masked region
    mask_indices = mask == 1
    overlayed[mask_indices] = cv2.addWeighted(
        frame[mask_indices], 1 - alpha, color_mask[mask_indices], alpha, 0
    )

    # Add material label in RED
    cv2.putText(overlayed, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2, cv2.LINE_AA)

    return overlayed


def save_video_from_frames(output_dir, video_path, fps=10):
    frame_paths = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".png")
    ])
    if not frame_paths:
        print("⚠️ No frames to compile into video.")
        return

    # Read first frame to get video size
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print(f"🎥 Writing video to {video_path} ...")
    for frame_path in tqdm(frame_paths):
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print("✅ Video saved.")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    predictions = load_predictions(args.json_file)

    frames = sorted([f for f in os.listdir(args.frame_dir) if f.endswith(".png")])

    for frame_name in tqdm(frames):
        frame_path = os.path.join(args.frame_dir, frame_name)
        mask_name = frame_name.replace(".png", "_fused_mask.png")
        mask_path = os.path.join(args.mask_dir, mask_name)

        if frame_name not in predictions:
            continue

        label = predictions[frame_name].get("label_voted", "Unknown")
        color = MATERIAL_COLORS.get(label, (255, 255, 255))

        frame = cv2.imread(frame_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if frame is None or mask is None:
            print(f"⚠️ Missing frame or mask for {frame_name}")
            continue

        annotated = overlay_mask_and_label(frame, mask, label, color, alpha=args.alpha)

        out_path = os.path.join(args.output_dir, frame_name)
        cv2.imwrite(out_path, annotated)

    if args.merge_video:
        save_video_from_frames(args.output_dir, args.video_path, fps=args.fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay material masks and labels on RGB frames and optionally export video")
    parser.add_argument("--frame_dir", required=True, help="Directory of RGB frames")
    parser.add_argument("--mask_dir", required=True, help="Directory of binary masks")
    parser.add_argument("--json_file", required=True, help="Path to prediction JSON")
    parser.add_argument("--output_dir", default="visualized", help="Output directory for annotated images")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay transparency")
    parser.add_argument("--merge_video", action="store_true", help="Merge visualized frames into video")
    parser.add_argument("--video_path", default="material_overlay.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for output video")
    args = parser.parse_args()
    main(args)
