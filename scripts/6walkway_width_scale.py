import os
import json
import numpy as np
from PIL import Image
import argparse
import csv
import cv2

import tqdm
import h5py

from openpose_utils import get_shoulder_avg, get_lowest_foot, get_wrist_avg


def get_foot_to_edge_distance(bottom, ground_line):
    """
    Calculate the minimum distance from the lowest foot to the left or right edge of the ground line (walkway).
    Args:
        bottom: (x, y) coordinate of the lowest foot.
        ground_line: ((x_left, y), (x_right, y)) tuple for ground line endpoints
    Returns:
        min_dist_px: minimum distance in pixels to the left or right edge of the ground line
        left_dist_px: distance to left edge of ground line
        right_dist_px: distance to right edge of ground line
    """
    if ground_line is None:
        return None, None, None
    x = bottom[0]
    x_left = ground_line[0][0]
    x_right = ground_line[1][0]
    left_dist = abs(x - x_left)
    right_dist = abs(x - x_right)
    min_dist = min(left_dist, right_dist)
    return min_dist, left_dist, right_dist

def get_ground_width(mask, foot_point):
    if mask.ndim == 3:
        mask = mask[...,0]
    y = int(round(foot_point[1]))
    y = np.clip(y, 0, mask.shape[0]-1)
    xs = np.where(mask[y]==255)[0]  # Only 255 is ground
    if len(xs)==0:
        return 0, None
    return xs[-1]-xs[0], ((xs[0],y),(xs[-1],y))


def render_estimation(frame, keypoints, top, bottom, ground_line, ground_width_m, frame_id, args):
    # Load the original image (assume same name as frame_name, update path as needed)
    img = frame  # Already loaded as numpy array
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw all keypoints
        for i in range(0, len(keypoints)):
            x, y = keypoints[i,0], keypoints[i,1]
            cv2.circle(img, (int(x), int(y)), 8, (0,255,0), -1)

        # Draw vertical line (shoulder avg to lowest foot)
        # cv2.line(img, (int(top[0]), int(top[1])), (int(bottom[0]), int(bottom[1])), (0,0,255), 10)

        # Draw dashed vertical line
        for i in range(0, int(abs(bottom[1] - top[1])), 10):
            start_y = int(top[1] + i)
            end_y = int(min(top[1] + i + 5, bottom[1]))
            cv2.line(img, (int(top[0]), start_y), (int(top[0]), end_y), (0,0,255), 10)

        # Draw ground horizontal line
        if ground_line:
            cv2.line(img, ground_line[0], ground_line[1], (255,0,0), 10)
            # Draw dashed horizontal line
            for i in range(0, int(abs(ground_line[1][0] - ground_line[0][0])), 10):
                start_x = int(ground_line[0][0] + i)
                end_x = int(min(ground_line[0][0] + i + 5, ground_line[1][0]))
                cv2.line(img, (start_x, int(ground_line[0][1])), (end_x, int(ground_line[0][1])), (255,0,0), 1)

        # Draw estimated width on the image
        if ground_width_m != "NA":
            text = f"Width: {ground_width_m:.2f} m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4
            thickness = 3
            # cyan color
            color = (255, 255, 0)
            # Position text at the bottom middle of the image
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = img.shape[0] - 100
            cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

        # Save visualization
        vis_path = os.path.join(args.output_vis_dir, f"{frame_id}_vis.png")
        cv2.imwrite(vis_path, img)
        print('Visualization saved to', vis_path)
    else:
        print('Image data not found for visualization:', frame_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypoint_json', required=True, help="Directory containing keypoint JSON files")
    parser.add_argument('--crop_offsets_json', required=True, help="Path to crop offsets JSON file")
    parser.add_argument('--ground_mask', required=True, help="Directory containing ground mask images")
    parser.add_argument('--frame_name', required=True, help="Directory containing frame images")
    parser.add_argument('--real_height', type=float, required=True, help="Real height of the person in meters")
    parser.add_argument('--output_vis_dir', help="Directory to save visualization results (optional)")
    parser.add_argument('--output_csv', help="Path to save the CSV file with estimation results (optional)")
    args = parser.parse_args()

    with open(args.crop_offsets_json) as f:
        crop_offsets = json.load(f)

    keypoints_h5 = h5py.File(args.keypoint_json, 'r')
    ground_mask_h5 = h5py.File(args.ground_mask, 'r')
    frame_h5 = h5py.File(args.frame_name, 'r')

    if args.output_vis_dir:
        os.makedirs(args.output_vis_dir, exist_ok=True)

    results = []
    n = len(frame_h5['frames'])
    module = n//10
    # for i, frame_id in tqdm.tqdm(enumerate(keypoints_h5['pose_keypoints'])):
    for i, frame_id in tqdm.tqdm(enumerate(frame_h5['frames'])):
        try:
            keypoints = keypoints_h5['pose_keypoints'][frame_id][:]
            # Access data from HDF5 files
            ground_mask = ground_mask_h5['binary_masks'][frame_id][:]
            frame = frame_h5['frames'][frame_id][:]
        except:
            print(f"Data missing for frame {frame_id}, skipping.")
            continue
        # Convert HDF5 datasets to numpy arrays
        keypoints = np.array(keypoints)
        offset = crop_offsets.get(frame_id, {"x1": 0, "y1": 0})
        dx, dy = offset["x1"], offset["y1"]
        for j in range(len(keypoints)):
            keypoints[j, 0] += dx
            keypoints[j, 1] += dy
        ground_mask = np.array(ground_mask)
        frame = np.array(frame)
        # import pdb; pdb.set_trace()
        # Top: avg shoulder
        top = get_shoulder_avg(keypoints)
        
        # top = get_wrist_avg(keypoints)
        
        # Bottom: lowest foot
        bottom = get_lowest_foot(keypoints)

        # Check if no feet keypoints were extracted
        if bottom is None:
            print(f"{frame_id}: No feet keypoints extracted. Walkway width estimation: NA")
            results.append({"frame_name": frame_id, "width_m": "NA"})
            continue

        # Pixel height
        pixel_height = abs(bottom[1] - top[1])
        # Ground width in px
        ground_width_px, ground_line = get_ground_width(ground_mask, bottom)

        # Check if ground line equals the width of the image
        if ground_width_px == frame.shape[1]:
            print(f"{frame_id}: Ground line equals image width. Walkway width estimation: NA")
            results.append({"frame_name": frame_id, "width_m": "NA", "foot_to_edge_m": "NA"})
            continue

        # Scale
        scale = args.real_height / pixel_height if pixel_height > 0 else 0
        ground_width_m = ground_width_px * scale

        # Foot to edge distance (in px and meters, relative to ground line)
        foot_to_edge_px, _, _ = get_foot_to_edge_distance(bottom, ground_line)
        foot_to_edge_m = foot_to_edge_px * scale if foot_to_edge_px is not None else "NA"

        # Append result (only minimal, scaled)
        results.append({
            "frame_name": frame_id,
            "width_m": ground_width_m,
            "foot_to_edge_m": foot_to_edge_m
        })

        # Visualization for debug
        if args.output_vis_dir and (i % module == 0):
            render_estimation(
                frame=frame,
                keypoints=keypoints,
                top=top,
                bottom=bottom,
                ground_line=ground_line,
                ground_width_m=ground_width_m,
                frame_id=frame_id,
                args=args
            )

    # Save results to CSV
    if args.output_csv:
        with open(args.output_csv, mode='w', newline='') as csvfile:
            fieldnames = ["frame_name", "width_m", "foot_to_edge_m"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Estimation results saved to {args.output_csv}")

if __name__ == '__main__':
    main()
