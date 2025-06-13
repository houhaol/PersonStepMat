import os
import pandas as pd
import cv2
from glob import glob

import matplotlib.pyplot as plt

def read_walkway_width(csv_path):
    df = pd.read_csv(csv_path)
    return df

def plot_width(df, save_path=None):
    df = df[df['width_m'] != 0]
    plt.figure(figsize=(10, 5))
    plt.plot(df['frame_name'], df['width_m'], marker='o')
    plt.xlabel('Frame Number')
    plt.ylabel('Walkway Width')
    plt.title('Walkway Width per Frame')
    plt.grid(True)
    # Set x-ticks at 50 interval
    plt.xticks(
        ticks=range(0, len(df), 100),
        labels=[str(i) for i in range(0, len(df), 100)]
    )
    if save_path:
        plt.savefig(save_path)
    plt.show()

def frames_to_video(frames_dir, output_path, fps=30):
    frame_files = sorted(glob(os.path.join(frames_dir, '*.png')))
    if not frame_files:
        print("No frames found in directory:", frames_dir)
        return
    # Read first frame to get size
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for file in frame_files:
        img = cv2.imread(file)
        video.write(img)
    video.release()
    print(f"Video saved to {output_path}")

def calculate_error(df):
    df = df[df['width_m'] != 0].copy()
    df['frame_num'] = df['frame_name'].str.extract(r'(\d+)').astype(int)

    def assign_gt_width(frame_num):
        if 0 <= frame_num <= 180:
            return 2.5
        elif 400 <= frame_num <= 550:
            return 4.0
        elif 650 <= frame_num <= 700:
            return 2.0
        else:
            return None

    df['gt_width'] = df['frame_num'].apply(assign_gt_width)
    df = df.dropna(subset=['gt_width'])

    df['abs_error'] = (df['width_m'] - df['gt_width']).abs()
    df['sq_error'] = (df['width_m'] - df['gt_width']) ** 2

    error = 0
    for (start, end, gt) in [(0, 180, 2.5), (400, 550, 4.0), (650, 700, 2.0)]:
        seg = df[(df['frame_num'] >= start) & (df['frame_num'] <= end)]
        if not seg.empty:
            mae = seg['abs_error'].mean()
            rmse = (seg['sq_error'].mean()) ** 0.5
            print(f"Frames {start}-{end}, GT={gt}: MAE={mae:.3f}, RMSE={rmse:.3f}")
            error += mae
    print(f"Total MAE across all segments: {error/3:.3f}")
if __name__ == "__main__":
    csv_path = '../dataset/pilot0/walkway_estimation/walkway_width.csv'
    df = read_walkway_width(csv_path)
    calculate_error(df)
    plot_width(df, save_path='../dataset/pilot0/walkway_width_plot.png')

    # Example usage for merging frames into a video
    # frames_dir = '../dataset/pilot10/walkway_estimation/'
    # output_video = '../dataset/pilot10/walkway_frames_video.mp4'
    # frames_to_video(frames_dir, output_video, fps=30)