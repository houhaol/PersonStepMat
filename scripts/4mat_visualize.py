import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
import h5py
import csv
import pandas as pd

from mat_utils import MATERIAL_COLORS
from openpose_utils import BODY_25_PAIRS_RENDER, POSE_COLORS



# ----------------- FUNCTIONS -----------------
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

    overlayed = frame.copy()
    color_mask = np.zeros_like(frame)
    color_mask[mask == 1] = color
    mask_indices = mask == 1

    overlayed[mask_indices] = cv2.addWeighted(
        frame[mask_indices], 1 - alpha, color_mask[mask_indices], alpha, 0
    )

    cv2.putText(overlayed, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return overlayed

def draw_keypoints(image, keypoints, point_radius=3, line_thickness=2):
    # keypoints: (25, 2) array, no confidence value
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(image, (int(x), int(y)), point_radius, (0, 255, 255), -1)

    # for idx, (a, b) in enumerate(BODY_25_PAIRS_RENDER):
    #     ptA = (int(keypoints[a][0]), int(keypoints[a][1]))
    #     ptB = (int(keypoints[b][0]), int(keypoints[b][1]))
    #     color = POSE_COLORS[idx % len(POSE_COLORS)]
    #     cv2.line(image, ptA, ptB, color, thickness=line_thickness)
    return image

def save_video_from_frames(output_dir, video_path, fps=10):
    frame_paths = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".png")
    ])
    if not frame_paths:
        print("⚠️ No frames to compile into video.")
        return

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


def render_estimation(frame, keypoints, ground_mask, material_label, frame_id, args):
    # frame: numpy array, keypoints: np.ndarray, ground_mask: np.ndarray, material_label: str
    img = frame  # Already loaded as numpy array
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw keypoints
        img = draw_keypoints(img, keypoints)

        # Overlay ground mask and label
        if ground_mask is not None and material_label is not None:
            color = MATERIAL_COLORS.get(material_label, (128, 128, 128))
            mask = (ground_mask > 0).astype(np.uint8)
            overlay = img.copy()
            overlay[mask == 1] = color
            alpha = args.alpha if hasattr(args, 'alpha') else 0.5
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            # Put material label text
            # Put material label text in the middle with increased font size
            font_scale = 5.0
            thickness = 4
            text = material_label
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            img_h, img_w = img.shape[:2]
            x = (img_w - text_width) // 2
            y = (img_h - 200)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        # Save visualization
        vis_path = os.path.join(args.output_vis_dir, f"{frame_id}_vis.png")
        cv2.imwrite(vis_path, img)
        print('Visualization saved to', vis_path)
    else:
        print('Image data not found for visualization:', frame_id)

# ----------------- MAIN -----------------

def load_material_labels_csv(csv_file):
    df = pd.read_csv(csv_file)
    # Assumes columns: frame,label
    return dict(zip(df['frame'], df['label_raw']))

def main(args):
    with open(args.crop_offsets_json) as f:
        crop_offsets = json.load(f)

    keypoints_h5 = h5py.File(args.keypoint_json, 'r')
    ground_mask_h5 = h5py.File(args.ground_mask, 'r')
    frame_h5 = h5py.File(args.frame_name, 'r')

    # Load material labels from CSV
    material_labels = load_material_labels_csv(args.material_label_csv)

    if args.output_vis_dir:
        os.makedirs(args.output_vis_dir, exist_ok=True)

        for i, frame_id in tqdm(enumerate(keypoints_h5['pose_keypoints'])):
            keypoints = keypoints_h5['pose_keypoints'][frame_id][:]
            ground_mask = ground_mask_h5['binary_masks'][frame_id][:]
            frame = frame_h5['frames'][frame_id][:]

            keypoints = np.array(keypoints)
            offset = crop_offsets.get(frame_id, {"x1": 0, "y1": 0})
            dx, dy = offset["x1"], offset["y1"]
            for j in range(len(keypoints)):
                keypoints[j, 0] += dx
                keypoints[j, 1] += dy
            ground_mask = np.array(ground_mask)
            frame = np.array(frame)

            # Get material label for this frame
            material_label = material_labels.get(frame_id, None)
            # Visualization for debug
            if args.output_vis_dir and i % args.interval == 0:
                render_estimation(
                    frame=frame,
                    keypoints=keypoints,
                    ground_mask=ground_mask,
                    material_label=material_label,
                    frame_id=frame_id,
                    args=args
                )


# ----------------- ARGUMENT PARSER -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay material masks and labels on RGB frames and optionally export video")

    parser.add_argument("--crop_offsets_json", required=True, help="Crop offset JSON for remapping keypoints")
    parser.add_argument("--keypoint_json", required=True, help="HDF5 file with keypoints")
    parser.add_argument("--ground_mask", required=True, help="HDF5 file with ground masks")
    parser.add_argument("--frame_name", required=True, help="HDF5 file with frames")
    parser.add_argument("--material_label_csv", required=True, help="CSV file with material labels for each frame")
    parser.add_argument("--output_vis_dir", default="visualized", help="Output directory for annotated images")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay transparency")
    parser.add_argument("--interval", type=int, default=1, help="Visualization interval")

    args = parser.parse_args()
    main(args)
