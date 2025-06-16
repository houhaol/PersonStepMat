import h5py
import cv2
import numpy as np
import argparse
import os

def visualize_masks(ground_h5_path, frame_h5_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Open the HDF5 files
    with h5py.File(ground_h5_path, 'r') as output_h5, h5py.File(frame_h5_path, 'r') as frame_h5:
        binary_masks = output_h5['binary_masks']
        frames = frame_h5['frames']

        for frame_name in binary_masks:
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
            overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)

            # Save the visualized frame
            output_path = os.path.join(output_dir, f"{frame_name}.png")
            cv2.imwrite(output_path, overlay)
            print(f"✅ Saved visualization for {frame_name} to {output_path}")
            import pdb; pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize binary masks over frames.")
    parser.add_argument("--ground_h5", required=True, help="Path to the HDF5 file containing binary masks.")
    parser.add_argument("--frame_h5", required=True, help="Path to the HDF5 file containing frames.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the visualized frames.")
    args = parser.parse_args()

    visualize_masks(
        ground_h5_path=args.ground_h5,
        frame_h5_path=args.frame_h5,
        output_dir=args.output_dir
    )