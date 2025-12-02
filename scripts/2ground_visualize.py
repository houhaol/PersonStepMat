import h5py
import cv2
import numpy as np
import argparse
import os
import json
from pycocotools import mask as maskUtils

def visualize_filter_masks(ground_h5_path, frame_h5_path, output_dir, interval=5):
    os.makedirs(output_dir, exist_ok=True)

    # Open the HDF5 files
    with h5py.File(ground_h5_path, 'r') as output_h5, h5py.File(frame_h5_path, 'r') as frame_h5:
        binary_masks = output_h5['binary_masks']
        frames = frame_h5['frames']

        for i, frame_name in enumerate(binary_masks):
            if i % interval != 0:
                continue

            # Load the binary mask and the corresponding frame
            mask = binary_masks[frame_name][:]
            frame = frames[frame_name][:]

            # Ensure the mask is binary and uint8
            mask = (mask > 0).astype(np.uint8) * 255

            # Convert mask to 3 channels for overlay
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Overlay the mask on the frame (e.g., mask in red)
            overlay = frame.copy()
            red_mask = np.zeros_like(frame)
            red_mask[:, :, 2] = mask  # Red channel
            # Blend the original frame and the red mask
            overlay = cv2.addWeighted(overlay, 1.0, red_mask, 0.3, 0)

            # Save the visualized frame
            output_path = os.path.join(output_dir, f"{frame_name}.png")
            # convert bgr to rgb
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path, overlay)
            print(f"✅ Saved visualization for {frame_name} to {output_path}")

def visualize_masks(ground_h5_path, frame_h5_path, output_dir, interval=5):
    os.makedirs(output_dir, exist_ok=True)

    # Open the HDF5 files
    with h5py.File(ground_h5_path, 'r') as output_h5, h5py.File(frame_h5_path, 'r') as frame_h5:
        ground_h5 = output_h5['results']
        frames = frame_h5['frames']

        for i, frame_name in enumerate(frames):
            if i % interval != 0:
                continue

            grounds = ground_h5[frame_name]
            annotations = json.loads(grounds['annotations'][()])
            frame = frames[frame_name][:]
            overlay = frame.copy()

            # Generate distinct colors for each segmentation
            num_anns = len(annotations)
            colors = [
                tuple(np.random.randint(0, 256, 3).tolist())
                for _ in range(num_anns)
            ]

            for idx, ann in enumerate(annotations):
                seg = ann['segmentation']
                rle = {
                    'counts': seg['counts'],
                    'size': seg['size']
                }
                mask = maskUtils.decode(rle)
                mask = (mask > 0).astype(np.uint8) * 255
                color = colors[idx]
                color_mask = np.zeros_like(frame, dtype=np.uint8)
                for c in range(3):
                    color_mask[:, :, c] = (mask // 255) * color[c]

                # Overlay the mask on the image
                overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)

            output_path = os.path.join(output_dir, f"{frame_name}.png")
            # Convert BGR to RGB for saving
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path, overlay)
            print(f"✅ Saved visualization for {frame_name} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize binary masks over frames.")
    parser.add_argument("--ground_h5", required=True, help="Path to the HDF5 file containing binary masks.")
    parser.add_argument("--frame_h5", required=True, help="Path to the HDF5 file containing frames.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the visualized frames.")
    parser.add_argument("--interval", type=int, default=900, help="Interval to save visualized frames.")
    args = parser.parse_args()

    if 'filter' in args.ground_h5:
        print("Visualizing filter masks...")
        visualize_filter_masks(
            ground_h5_path=args.ground_h5,
            frame_h5_path=args.frame_h5,
            output_dir=args.output_dir,
            interval=args.interval
        )
    else:
        visualize_masks(
            ground_h5_path=args.ground_h5,
            frame_h5_path=args.frame_h5,
            output_dir=args.output_dir,
            interval=args.interval
        )