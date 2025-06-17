import os
import pandas as pd
import cv2
from glob import glob

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter



def read_walkway_width(csv_path):
    df = pd.read_csv(csv_path)
    df['abs_time'] = df['frame_name'].str.extract(r'(\d+)$')
    df['abs_time'] = df['abs_time'].astype(int)
    # import pdb; pdb.set_trace()
    df['rel_time'] = df['abs_time'] - df['abs_time'].min()
    # nanoseconds to seconds
    df['rel_time'] = df['rel_time'] / 1e9
    return df

def plot_width(df, save_path=None, segments_detect=False):
    """
    Plot walkway width over time, optionally highlighting consistent segments.

    Parameters:
        df (pd.DataFrame): The DataFrame containing walkway width data.
        save_path (str): Path to save the plot.
        consistent_segments (list): A list of DataFrames, each representing a consistent segment.
    """
    df = df[df['width_m'] != 0]
    plt.figure(figsize=(10,6))
    plt.plot(df['rel_time'], df['width_m'], marker='o', markersize=0.5, linestyle='-', color='g', label='Walkway Width', alpha=0.2)
    plt.plot(df['rel_time'], df['width_m_gaussian'], linestyle='-', color='r', label='Gaussian Smoothed Width', alpha=0.7)
    # plt.plot(df['rel_time'], df['width_m_savgol'], linestyle='-', color='g', label='Savitzky-Golay Smoothed Width')

    # Plot consistent segments if provided
    if segments_detect:
        # Parameters: Time window 10s consider 1.2m/s for human, around 10-12m to enable a segment detection. Variance_threshold=0.1 is around +- 0.3 meters
        segments = segment_detect(df, 'width_m_gaussian',time_window=10, step_size=5, variance_threshold=0.1) 
        merged_segments = merge_segments(segments, 'width_m_gaussian', time_gap=3, dist_gap=0.5)
        # when plotting the segments, name the segment orderly
        for i, segment in enumerate(merged_segments):
            avg_value = segment['width_m_gaussian'].mean()
            # plot the number on the top of the segment
            plt.text(segment['rel_time'].mean(), avg_value + 0.1, f'{i+1}', fontsize=10, ha='center', va='bottom', color='blue')
            # plot the segment as a horizontal line
            plt.hlines(y=avg_value, xmin=segment['rel_time'].min(), xmax=segment['rel_time'].max(),
                       colors='b', linestyles='-', label=f'{i+1} Consistent Segment (avg={avg_value:.2f})', linewidth=2)

    plt.xlabel('Relative Time')
    # more dense x-axis ticks
    plt.xticks(rotation=45)
    plt.xticks(ticks=plt.MultipleLocator(50.0).tick_values(df['rel_time'].min()+1, df['rel_time'].max()))
    plt.yticks(ticks=plt.MultipleLocator(1.0).tick_values(df['width_m'].min(), df['width_m'].max()))
    plt.ylabel('Walkway Width (meters)')
    plt.title('Walkway Width Over Time')
    plt.grid(True)
    # plot legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(0.8, 1), fontsize='small')

    if save_path:
        plt.savefig(save_path)
    # plt.show()

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

def smooth_data(df):
    gaussian = gaussian_filter1d(df['width_m'], sigma=10)
    savgol = savgol_filter(df['width_m'], window_length=51, polyorder=3)
    df['width_m_gaussian'] = gaussian
    df['width_m_savgol'] = savgol
    return df

def segment_detect(df, width_category,time_window=10, step_size=3, variance_threshold=0.15):
    """
    Detect all consistent segments where the width is relatively similar within a time window, allowing overlapping windows.

    Parameters:
        df (pd.DataFrame): The DataFrame containing walkway width data.
        time_window (float): The duration of the time window in seconds.
        step_size (float): The step size in seconds for moving the window.
        variance_threshold (float): The maximum allowable variance for a segment to be considered consistent.

    Returns:
        list: A list of DataFrames, each representing a consistent segment.
    """
    consistent_segments = []

    # Sort the DataFrame by relative time to ensure proper segmentation
    df = df.sort_values(by='rel_time')

    start_time = df['rel_time'].min()
    end_time = df['rel_time'].max()

    # Iterate through the DataFrame using overlapping time windows
    current_start = start_time
    while current_start + time_window <= end_time:
        segment = df[(df['rel_time'] >= current_start) & (df['rel_time'] < current_start + time_window)]
        variance = segment[width_category].var()
        mean = segment[width_category].mean()
        # Check if the variance is below the threshold
        if variance <= variance_threshold and mean > 0.5:
            consistent_segments.append(segment)
        

        current_start += step_size  # Move the window by the step size

    return consistent_segments


def merge_segments(segments, width_category,time_gap=5, dist_gap=0.3):
    """
    Merge consistent segments if they are close in time.

    Parameters:
        segments (list): A list of DataFrames, each representing a consistent segment.
        time_gap (float): The maximum allowable time gap (in seconds) between segments to merge them.

    Returns:
        list: A list of merged DataFrames.
    """
    if not segments:
        return []

    # Sort segments by their start time
    segments = sorted(segments, key=lambda seg: seg['rel_time'].min())

    merged_segments = [segments[0]]

    for segment in segments[1:]:
        last_segment = merged_segments[-1]

        # Check if the current segment is close enough to the last merged segment, the average width of the last segment and the current segment should be similar
        if (segment['rel_time'].min() - last_segment['rel_time'].max() <= time_gap and
            abs(segment[width_category].mean() - last_segment[width_category].mean()) <= dist_gap):
            # Merge the segments
            merged_segments[-1] = pd.concat([last_segment, segment])
        else:
            # Add as a new segment
            merged_segments.append(segment)

    return merged_segments

if __name__ == "__main__":
    csv_path = '../dataset/pilot10/walkway_width.csv'
    df = read_walkway_width(csv_path)
    df = smooth_data(df)
    # calculate_error(df)
    plot_width(df, save_path='../dataset/pilot10/walkway_width_plot.png', segments_detect=True)

    # Example usage for merging frames into a video
    # frames_dir = '../dataset/pilot10/walkway_estimation/'
    # output_video = '../dataset/pilot10/walkway_frames_video.mp4'
    # frames_to_video(frames_dir, output_video, fps=30)