import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm


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

def overlay_ground_mask(image, mask_path, color=(255, 255, 0), alpha=0.4):
    if not os.path.exists(mask_path):
        return image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image.shape[:2]:
        return image
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def map_keypoints(cropped_json_dir, offset_json_path, full_frame_dir, ground_mask_dir, overlay_dir, shift_y=10, save_video=False, video_name="overlay_output.mp4"):
    if not save_video:
        os.makedirs(overlay_dir, exist_ok=True)

    with open(offset_json_path, "r") as f:
        crop_offsets = json.load(f)

    frame_list = []
    file_list = sorted([f for f in os.listdir(cropped_json_dir) if f.endswith("_keypoints.json")])
    for fname in tqdm(file_list, desc="Processing frames"):
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
        if image is None:
            print(f"⚠️ Skipped: cannot read {full_image_path}")
            continue

        # Overlay ground mask
        mask_path = os.path.join(ground_mask_dir, img_fname.replace(".png", "_fused_mask.png"))
        image = overlay_ground_mask(image, mask_path, color=(80, 150, 220))

        for person in people:
            keypoints = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)
            keypoints[:, 0] += dx
            keypoints[:, 1] += dy
            image = draw_keypoints(image, keypoints, line_thickness=8)

        # Put "Concrete" text in the middle of the image with large font size
        text = "Tiles"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3.0
        thickness = 6
        color = (0, 0, 255)
        cv2.putText(image, text, (100, 100), font, font_scale, color, thickness, cv2.LINE_AA)
        if save_video:
            frame_list.append(image)
        else:
            overlay_path = os.path.join(overlay_dir, img_fname)
            cv2.imwrite(overlay_path, image)


    # Save video if enabled
    if save_video and frame_list:
        height, width, _ = frame_list[0].shape
        out_path = os.path.join(overlay_dir, video_name)
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        for frame in frame_list:
            out.write(frame)
        out.release()
        print(f"🎥 Saved video to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay OpenPose keypoints and ground mask on full image or save as video.")
    parser.add_argument("--cropped_json_dir", required=True, help="Dir of cropped keypoints JSON files")
    parser.add_argument("--offset_json", required=True, help="JSON containing crop offsets")
    parser.add_argument("--full_frame_dir", required=True, help="Dir with original full-frame images")
    parser.add_argument("--ground_mask_dir", required=True, help="Dir with binary ground masks")
    parser.add_argument("--overlay_dir", required=True, help="Dir to save output (images or video)")
    parser.add_argument("--shift_y", type=int, default=10, help="Vertical shift for foot prompts (unused now)")
    parser.add_argument("--save_video", action="store_true", help="If set, saves output as a video instead of images")
    parser.add_argument("--video_name", type=str, default="overlay_output.mp4", help="Filename of output video (if saving video)")

    args = parser.parse_args()

    map_keypoints(
        cropped_json_dir=args.cropped_json_dir,
        offset_json_path=args.offset_json,
        full_frame_dir=args.full_frame_dir,
        ground_mask_dir=args.ground_mask_dir,
        overlay_dir=args.overlay_dir,
        shift_y=args.shift_y,
        save_video=args.save_video,
        video_name=args.video_name
    )
