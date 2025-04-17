import os
import json
import argparse
import numpy as np
import cv2

# OpenPose BODY_25 color palette and connections
BODY_25_PAIRS_RENDER = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9,10), (10,11), (8,12), (12,13), (13,14),
    (0,1), (0,15), (15,17), (0,16), (16,18),
    (14,21), (11,22), (14,19), (19,20), (11,24), (22,23)
]

# BGR colors similar to OpenPose's rendering (sampled for each limb)
POSE_COLORS = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170),
    (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
    (128, 128, 255), (128, 255, 128), (255, 128, 128), (255, 255, 128)
]

def draw_keypoints(image, keypoints, point_radius=3, line_thickness=2):
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.05:
            cv2.circle(image, (int(x), int(y)), point_radius, (0, 255, 255), -1)  # yellow joints

    for idx, (partA, partB) in enumerate(BODY_25_PAIRS_RENDER):
        if keypoints[partA][2] > 0.05 and keypoints[partB][2] > 0.05:
            ptA = (int(keypoints[partA][0]), int(keypoints[partA][1]))
            ptB = (int(keypoints[partB][0]), int(keypoints[partB][1]))
            color = POSE_COLORS[idx % len(POSE_COLORS)]
            cv2.line(image, ptA, ptB, color, thickness=line_thickness)
    return image


def map_keypoints(cropped_json_dir, offset_json_path, output_dir, full_frame_dir, overlay_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Load offset map
    with open(offset_json_path, "r") as f:
        crop_offsets = json.load(f)

    # Process each keypoint JSON
    for fname in sorted(os.listdir(cropped_json_dir)):
        if not fname.endswith("_keypoints.json"):
            continue

        input_path = os.path.join(cropped_json_dir, fname)
        with open(input_path, "r") as f:
            data = json.load(f)

        people = data.get("people", [])
        if not people:
            continue

        # Determine original frame name to get offset + full image
        img_fname = fname.replace("_keypoints.json", ".png")
        offset = crop_offsets.get(img_fname, {"x1": 0, "y1": 0})
        dx, dy = offset["x1"], offset["y1"]

        full_image_path = os.path.join(full_frame_dir, img_fname)
        image = cv2.imread(full_image_path)

        # Shift keypoints and overlay on image
        for person in people:
            keypoints = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)
            keypoints[:, 0] += dx
            keypoints[:, 1] += dy
            person["pose_keypoints_2d"] = keypoints.flatten().tolist()

            # Draw on image
            image = draw_keypoints(image, keypoints)

        # Save updated JSON
        output_path = os.path.join(output_dir, fname)
        with open(output_path, "w") as f:
            json.dump(data, f)

        # Save overlay image
        overlay_path = os.path.join(overlay_dir, img_fname)
        cv2.imwrite(overlay_path, image)

        print(f"✅ Remapped & visualized: {fname} → {overlay_path}")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map OpenPose cropped keypoints back to full-frame coordinates and overlay on original images.")
    parser.add_argument("--cropped_json_dir", required=True, help="Directory of OpenPose JSON outputs from cropped frames.")
    parser.add_argument("--offset_json", required=True, help="Path to crop_offsets.json.")
    parser.add_argument("--output_dir", required=True, help="Directory to save updated full-frame JSONs.")
    parser.add_argument("--full_frame_dir", required=True, help="Directory of original full-frame images (e.g., frame_0000.png).")
    parser.add_argument("--overlay_dir", required=True, help="Directory to save overlay visualizations.")

    args = parser.parse_args()

    map_keypoints(
        cropped_json_dir=args.cropped_json_dir,
        offset_json_path=args.offset_json,
        output_dir=args.output_dir,
        full_frame_dir=args.full_frame_dir,
        overlay_dir=args.overlay_dir
    )
