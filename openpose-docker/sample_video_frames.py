import cv2
import os
import argparse

def sample_video_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞️ FPS: {fps:.2f}, Total frames: {total_frames}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as PNG
        frame_name = f"frame_{frame_idx:04d}.png"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame)

        print(f"✅ Saved: {frame_name}")
        frame_idx += 1

    cap.release()
    print(f"\n📁 All frames saved to: {output_dir}")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample all video frames at original FPS.")
    parser.add_argument("--video_path", required=True, help="Path to input video file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save sampled frames.")
    args = parser.parse_args()

    sample_video_frames(args.video_path, args.output_dir)
