import os
import json
import numpy as np
from PIL import Image
import argparse
import csv
import h5py

def get_shoulder_avg(keypoints):
    # Shoulders: 2 (RShoulder), 5 (LShoulder)
    s1 = keypoints[2*3:2*3+2]
    s2 = keypoints[5*3:5*3+2]
    return [(s1[0]+s2[0])/2, (s1[1]+s2[1])/2]

def get_lowest_foot(keypoints):
    # Left: 21, 19, 20; Right: 24, 22, 23
    left = [keypoints[i*3:i*3+2] for i in [21,19,20]]
    right = [keypoints[i*3:i*3+2] for i in [24,22,23]]
    left_y = [pt[1] for pt in left]
    right_y = [pt[1] for pt in right]
    left_idx = np.argmax(left_y)
    right_idx = np.argmax(right_y)
    left_foot = left[left_idx]
    right_foot = right[right_idx]
    # Pick the lower (max y)
    return left_foot if left_foot[1] > right_foot[1] else right_foot

def get_ground_width(mask_path, foot_point):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[...,0]
    y = int(round(foot_point[1]))
    y = np.clip(y, 0, mask.shape[0]-1)
    xs = np.where(mask[y]==255)[0]  # Only 255 is ground
    if len(xs)==0:
        return 0, None
    return xs[-1]-xs[0], ((xs[0],y),(xs[-1],y))

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

    for frame_id in keypoints_h5['pose_keypoints']:
        keypoints = keypoints_h5['pose_keypoints'][frame_id][:]
        # Access data from HDF5 files
        ground_mask = ground_mask_h5['binary_masks'][frame_id][:]
        frame = frame_h5['frames'][frame_id][:]
        import pdb; pdb.set_trace()

        # Convert HDF5 datasets to numpy arrays
        keypoints = np.array(keypoints)
        ground_mask = np.array(ground_mask)
        frame = np.array(frame)

        # Top: avg shoulder
        top = get_shoulder_avg(keypoints)
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
            results.append({"frame_name": frame_id, "width_m": "NA"})
            continue

        # Scale
        scale = args.real_height / pixel_height if pixel_height > 0 else 0
        ground_width_m = ground_width_px * scale
        print(f"{frame_id}: Ground width (meters):", ground_width_m)

        # Append result
        results.append({"frame_name": frame_id, "width_m": ground_width_m})

        # Visualization for debug
        if args.output_vis_dir:
            import cv2

            # Load the original image (assume same name as frame_name, update path as needed)
            img = frame  # Already loaded as numpy array
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Draw all keypoints
                for i in range(0, len(keypoints), 3):
                    x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
                    if conf > 0.1:
                        cv2.circle(img, (int(x), int(y)), 3, (0,255,0), -1)

                # Draw vertical line (shoulder avg to lowest foot)
                cv2.line(img, (int(top[0]), int(top[1])), (int(bottom[0]), int(bottom[1])), (0,0,255), 2)

                # Draw dashed vertical line
                for i in range(0, int(abs(bottom[1] - top[1])), 10):
                    start_y = int(top[1] + i)
                    end_y = int(min(top[1] + i + 5, bottom[1]))
                    cv2.line(img, (int(top[0]), start_y), (int(top[0]), end_y), (0,0,255), 1)

                # Draw ground horizontal line
                if ground_line:
                    cv2.line(img, ground_line[0], ground_line[1], (255,0,0), 2)

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

    # Save results to CSV
    if args.output_csv:
        with open(args.output_csv, mode='w', newline='') as csvfile:
            fieldnames = ["frame_name", "width_m"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Estimation results saved to {args.output_csv}")

if __name__ == '__main__':
    main()
