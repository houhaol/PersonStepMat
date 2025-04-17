import os
import json
import argparse
import numpy as np
import cv2

# OpenPose BODY_25 connections
BODY_25_PAIRS_RENDER = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (0, 1), (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 21), (11, 22), (14, 19), (19, 20), (11, 24), (22, 23)
]

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
            cv2.circle(image, (int(x), int(y)), point_radius, (0, 255, 255), -1)

    for idx, (a, b) in enumerate(BODY_25_PAIRS_RENDER):
        if keypoints[a][2] > 0.05 and keypoints[b][2] > 0.05:
            ptA = (int(keypoints[a][0]), int(keypoints[a][1]))
            ptB = (int(keypoints[b][0]), int(keypoints[b][1]))
            color = POSE_COLORS[idx % len(POSE_COLORS)]
            cv2.line(image, ptA, ptB, color, thickness=line_thickness)
    return image

def map_keypoints(cropped_json_dir, offset_json_path, output_dir, full_frame_dir, overlay_dir, prompt_dir, shift_y=10):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)

    with open(offset_json_path, "r") as f:
        crop_offsets = json.load(f)

    for fname in sorted(os.listdir(cropped_json_dir)):
        if not fname.endswith("_keypoints.json"):
            continue

        input_path = os.path.join(cropped_json_dir, fname)
        with open(input_path, "r") as f:
            data = json.load(f)

        people = data.get("people", [])
        if not people:
            continue

        img_fname = fname.replace("_keypoints.json", ".png")
        offset = crop_offsets.get(img_fname, {"x1": 0, "y1": 0})
        dx, dy = offset["x1"], offset["y1"]

        full_image_path = os.path.join(full_frame_dir, img_fname)
        image = cv2.imread(full_image_path)

        foot_prompts = []

        for person in people:
            keypoints = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)
            keypoints[:, 0] += dx
            keypoints[:, 1] += dy
            person["pose_keypoints_2d"] = keypoints.flatten().tolist()

            image = draw_keypoints(image, keypoints)

            # Foot prompts - left
            for idx in [21, 19, 20]:
                x, y, conf = keypoints[idx]
                if conf > 0.1:
                    foot_prompts.append({"foot": "left", "point": [int(x), int(y + shift_y)]})
                    break

            # Foot prompts - right
            for idx in [24, 22, 23]:
                x, y, conf = keypoints[idx]
                if conf > 0.1:
                    foot_prompts.append({"foot": "right", "point": [int(x), int(y + shift_y)]})
                    break

        # Draw foot prompts
        for prompt in foot_prompts:
            px, py = prompt["point"]
            cv2.circle(image, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(image, prompt["foot"], (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out_json_path = os.path.join(output_dir, fname)
        with open(out_json_path, "w") as f:
            json.dump(data, f)

        overlay_path = os.path.join(overlay_dir, img_fname)
        cv2.imwrite(overlay_path, image)

        if foot_prompts:
            prompt_path = os.path.join(prompt_dir, fname.replace("_keypoints.json", "_feet_prompt.json"))
            with open(prompt_path, "w") as f:
                json.dump(foot_prompts, f, indent=2)
            print(f"🦶 Saved prompts → {prompt_path}")

        print(f"✅ Remapped & visualized: {fname} → {overlay_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap OpenPose keypoints to full image and extract foot prompts.")
    parser.add_argument("--cropped_json_dir", required=True, help="Dir of cropped keypoints")
    parser.add_argument("--offset_json", required=True, help="JSON containing crop offsets")
    parser.add_argument("--output_dir", required=True, help="Dir to save updated keypoints JSONs")
    parser.add_argument("--full_frame_dir", required=True, help="Dir with full original frames")
    parser.add_argument("--overlay_dir", required=True, help="Dir to save visualization overlays")
    parser.add_argument("--prompt_dir", required=True, help="Dir to save per-foot prompt JSONs")
    parser.add_argument("--shift_y", type=int, default=10, help="Vertical offset in pixels to nudge prompts down")

    args = parser.parse_args()

    map_keypoints(
        cropped_json_dir=args.cropped_json_dir,
        offset_json_path=args.offset_json,
        output_dir=args.output_dir,
        full_frame_dir=args.full_frame_dir,
        overlay_dir=args.overlay_dir,
        prompt_dir=args.prompt_dir,
        shift_y=args.shift_y
    )