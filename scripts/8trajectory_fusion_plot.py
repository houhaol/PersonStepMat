import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Add this import at the top
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from mat_utils import MATERIAL_COLORS, smooth_data, filter_data, convert_label_to_id
import json

# WGS84 ellipsoid constants
a = 6378137.0
f = 1 / 298.257223563
e2 = f * (2 - f)

def ecef_to_gps(x, y, z):
    """Convert ECEF (x, y, z) to WGS84 (lat, lon, alt)."""
    # WGS84 constants
    b = a * (1 - f)
    ep = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep**2 * b * np.sin(th)**3, p - e2 * a * np.cos(th)**3)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)
    return lat, lon, alt

def load_traj(traj_path):
    # Load trajectory TXT (TUM format)
    # df_traj = pd.read_csv(traj_path, delim_whitespace=True, header=None)
    # df_traj.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    
    # Load trajectory csv
    df_traj = pd.read_csv(traj_path)
    # Rename timestamp_ns to timestamp
    df_traj = df_traj.rename(columns={'timestamp_ns': 'timestamp'})
    # If traj timestamps are in seconds, convert to nanoseconds for matching
    if df_traj['timestamp'].max() < 1e12:
        df_traj['timestamp'] = (df_traj['timestamp'] * 1e9).astype(np.int64)

    df_traj = df_traj.sort_values('timestamp')
    return df_traj

def load_feature(width_path):
    # Load walkway width CSV
    df_width = pd.read_csv(width_path, dtype=object)
    try:
        df_width['timestamp'] = df_width['frame_name'].str.extract(r'frame_(\d+)').astype(np.int64)
    except:
        df_width['timestamp'] = df_width['frame'].str.extract(r'frame_(\d+)').astype(np.int64)

    # may align the timestamps a bit. timestamps in df_width is in GMT+8, while traj is in GMT+0
    df_width['timestamp'] = df_width['timestamp'] + 8 * 3600 * 1e9  # Convert to nanoseconds and adjust for timezone
    df_width['timestamp'] = df_width['timestamp'].astype(np.int64)
    return df_width

def fuse_feature(traj_path, feature_path, feature, categorical=False):
    df_traj = load_traj(traj_path)
    # import pdb; pdb.set_trace()
    df_feature = load_feature(feature_path)
    if categorical:
        df_feature = convert_label_to_id(df_feature)
        df_feature = smooth_data(df_feature, window_size=10)
    # else:
        # df_feature = width_smooth_data(df_feature)
    # Nearest neighbor join
    df_fused = pd.merge_asof(
        df_traj,
        df_feature,
        on='timestamp',
        direction='nearest'
    )
    if categorical:
        # Ensure categorical feature is treated as string
        df_fused[feature] = df_fused[feature].astype(str)
    else:
        df_fused[feature] = df_fused[feature].astype(float)  # Ensure feature is float type
        # filter out rows where feature is greater than 0
        df_fused = df_fused[df_fused[feature] > 0]
    # Then use the max timestamp from this filtered data as threshold
    max_timestamp = df_feature['timestamp'].max()
    df_final = df_fused[df_fused['timestamp'] <= max_timestamp]
    return df_final

    # Save result
    # df_fused.to_csv(output_path, index=False)
    # print(f"Fused file saved as {output_path}")

def plot_traj_with_numerical_features(df_plot, feature='width_m', category_col=None):
    """
    Plots trajectory points with size and color based on a numerical column (e.g., width_m).
    Swaps x and y axes so y becomes x and x becomes y. (0,0) is at the bottom left.
    Negative y values will be shown from down to up (invert y-axis).
    Every 100s, show the time data above the datapoints.
    """
    # df_plot = df_plot.iloc[::40].reset_index(drop=True)  # Optionally reduce density
    df_plot['rel_time'] = df_plot['timestamp'] - df_plot['timestamp'].min()
    df_plot['rel_time'] = df_plot['rel_time'] / 1e9  # nanoseconds to seconds

    width = df_plot[feature]
    width_norm = (width - width.min()) / (width.max() - width.min())
    size = width * 20  # Scale size for visibility, adjust as needed

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_plot['y'], df_plot['x'],
                        c=width,
                        cmap='plasma',
                        alpha=0.2,
                        edgecolor='none',
                        s=size,
                        marker='o',
                        )
    cbar = plt.colorbar(scatter)
    if hasattr(cbar, 'solids') and cbar.solids is not None:
        cbar.solids.set_alpha(1)
    cbar.set_label(f'{feature}')
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title(f'Trajectory with {feature}')
    # plt.title(f'Trajectory with walkway width')
    plt.grid(True)

    # Annotate every 100s
    for i, row in df_plot.iterrows():
        if row['rel_time'] % 100 < 1:  # within 1s of a 100s mark
            plt.text(row['y'], row['x'], f"{int(row['rel_time'])}s", color='black', fontsize=11, ha='center', va='bottom')

    plt.gca().invert_yaxis()
    plt.savefig('trajectory_with_numerical_features.png')
    # plt.show()
    print("Saved figure trajectory_with_numerical_features.png")

def plot_traj_with_categorical_features(df_plot, feature='label_raw'):
    """
    Plots trajectory points with size and color based on a categorical column (e.g., label_raw).
    Swaps x and y axes so y becomes x and x becomes y. (0,0) is at the bottom left.
    Negative y values will be shown from down to up (invert y-axis).
    Every 100s, show the time data above the datapoints.
    """
    # df_plot = df_plot.iloc[::40].reset_index(drop=True)  # Optionally reduce density
    df_plot['rel_time'] = df_plot['timestamp'] - df_plot['timestamp'].min()
    df_plot['rel_time'] = df_plot['rel_time'] / 1e9  # nanoseconds to seconds

    category = df_plot[feature].astype(str)
    # Use MATERIAL_COLORS for color mapping, normalize to [0,1]
    def rgb_to_mpl(rgb):
        return tuple([v / 255.0 for v in rgb])

    category_color_map = {k: rgb_to_mpl(MATERIAL_COLORS[k]) for k in category.unique() if k in MATERIAL_COLORS}
    # For unknown categories, assign a default color (e.g., gray)
    default_color = (0.5, 0.5, 0.5)
    color_values = category.map(lambda x: category_color_map.get(x, default_color))

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_plot['y'], df_plot['x'],
                        c=list(color_values),
                        alpha=0.2,
                        edgecolor='none',
                        marker='o',
                        )
    plt.xlabel('Y')
    plt.ylabel('X')
    # Create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=category_color_map.get(cat, default_color), markersize=10, label=cat)
               for cat in category.unique()]
    # Get top 5 most frequent categories
    top_categories = category.value_counts().head(5).index.tolist()
    top_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=category_color_map.get(cat, default_color), markersize=10, label=cat)
                   for cat in top_categories if cat in category_color_map]
    
    plt.legend(handles=top_handles, title="Material Labels", loc='center right')
    plt.title(f'Trajectory with material')
    plt.grid(True)

    # Annotate every 100s
    for i, row in df_plot.iterrows():
        if row['rel_time'] % 100 < 1:  # within 1s of a 100s mark
            plt.text(row['y'], row['x'], f"{int(row['rel_time'])}s", color='black', fontsize=11, ha='center', va='bottom')

    plt.gca().invert_yaxis()
    plt.savefig('trajectory_with_categorical_features.png')
    print("Saved figure trajectory_with_categorical_features.png")

def width_smooth_data(df):
    gaussian = gaussian_filter1d(df['width_m'], sigma=10)
    savgol = savgol_filter(df['width_m'], window_length=51, polyorder=3)
    df['width_m_gaussian'] = gaussian
    df['width_m_savgol'] = savgol
    return df

def append_lat_lon(df, json_path):
    with open(json_path, 'r') as f:
        alignment = json.load(f)
        scale = alignment['scale']
        rotation_matrix = np.array(alignment['rotation_matrix'])
        translation_vector = np.array(alignment['translation_vector'])
    # Apply alignment to trajectory points
    points = df[['x', 'y', 'z']].values
    xyz_ecef = scale * rotation_matrix @ points.T + translation_vector.reshape(3, 1)
    lat, lon, alt = ecef_to_gps(*xyz_ecef)
    df['lat'] = lat
    df['lon'] = lon
    df['alt'] = alt
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse walkway width to trajectory using nearest timestamp.")
    parser.add_argument('--traj', type=str, required=True, help='Path to traj.txt')
    parser.add_argument('--width', type=str, help='Path to walkway_width.csv')
    parser.add_argument('--mat', type=str, help='Path to material.csv')
    parser.add_argument('--gait', type=str, help='Path to gait.csv')
    parser.add_argument('--output', type=str, default='traj_with_width.csv', help='Output CSV file')
    parser.add_argument('--json', type=str, help='Path to JSON file')
    args = parser.parse_args()

    # width_fused_df = fuse_feature(args.traj, args.width, 'width_m_gaussian')

    # plot_traj_with_numerical_features(width_fused_df, feature='width_m_gaussian', category_col=None)

    mat_fused_df = fuse_feature(args.traj, args.mat, 'label_smooth', categorical=True)
    if args.json:
        width_fused_df = append_lat_lon(mat_fused_df, args.json)
        # remove file extension from output
        width_fused_df.to_csv(args.output, index=False)
    plot_traj_with_categorical_features(mat_fused_df, feature='label_voted')
    mat_fused_df.to_csv(args.output, index=False)
    # import pdb; pdb.set_trace()


    

    
