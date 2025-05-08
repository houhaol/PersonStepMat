import pandas as pd
import numpy as np
import argparse
from scipy.spatial import cKDTree

def load_tum(tum_file):
    df = pd.read_csv(tum_file, delim_whitespace=True, header=None)
    df.columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    return df

def load_labels_and_timestamps(csv_file, npy_file, time_offset_sec):
    labels_df = pd.read_csv(csv_file)
    timestamps = np.load(npy_file)  # relative seconds

    if len(timestamps) != len(labels_df):
        raise ValueError("Mismatch between number of labels and number of timestamps!")

    # Apply offset
    absolute_ts = timestamps + time_offset_sec
    labels_df['timestamp'] = absolute_ts
    return labels_df

def merge_nearest_within_range(tum_df, labels_df, max_time_diff=0.1):
    # Convert TUM timestamps to relative time (0-based)
    t0 = tum_df['timestamp'].iloc[0]
    tum_df['timestamp_rel'] = tum_df['timestamp'] - t0

    # Build KD-tree on label timestamps (already absolute video-relative)
    label_ts = labels_df['timestamp'].values.reshape(-1, 1)
    tree = cKDTree(label_ts)

    # Query nearest label for each relative trajectory timestamp
    tum_ts_rel = tum_df['timestamp_rel'].values.reshape(-1, 1)
    distances, indices = tree.query(tum_ts_rel, k=1)

    # Only keep matches within max time difference
    matched = labels_df.iloc[indices].reset_index(drop=True)
    matched.loc[distances > max_time_diff, ['label_raw', 'label_voted', 'confidence']] = np.nan

    # Drop temporary relative timestamp, merge, return
    merged = pd.concat([tum_df.drop(columns='timestamp_rel').reset_index(drop=True),
                        matched[['label_raw', 'label_voted', 'confidence']]], axis=1)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', required=True, help="Trajectory in TUM format")
    parser.add_argument('--csv', required=True, help="CSV file with image labels")
    parser.add_argument('--npy', required=True, help="Timestamp .npy file for the labels")
    parser.add_argument('--offset', type=float, required=True, help="Start time offset in seconds (e.g. 240.0)")
    parser.add_argument('--output', required=True, help="Path to output merged CSV")
    parser.add_argument('--max_diff', type=float, default=0.1, help="Max time diff (in seconds) for matching")
    args = parser.parse_args()

    tum_df = load_tum(args.traj)
    labels_df = load_labels_and_timestamps(args.csv, args.npy, args.offset)
    merged_df = merge_nearest_within_range(tum_df, labels_df, max_time_diff=args.max_diff)

    merged_df.to_csv(args.output, index=False)
    print(f"✅ Merged file saved to: {args.output}")

if __name__ == "__main__":
    main()
