#GoPro_timestamp.py
#Created by Chris Rillahan
#Last Updated: 1/30/2015
#Written with Python 2.7.2, OpenCV 2.4.8

#This script uses ffprobe to interogate a MP4 file and extract the creation time.
#This information is then used to initiate a counter/clock which is used to put
#the timestamp on each frame of the video.  The videos are samed as *.avi files
#using DIVX compression due to the availability in OpenCV.  The audio is stripped
#out in this script.

import cv2, os, sys, subprocess, shlex, re, time
import datetime as dt
from subprocess import call
import numpy as np
import argparse
import h5py

# Add argument parsing
parser = argparse.ArgumentParser(description="Process GoPro video and save frames and timestamps.")
parser.add_argument("--filename", type=str, help="Path to the GoPro video file.")
parser.add_argument("--duration", type=float, default=24, help="Duration in minutes to process the video.")
parser.add_argument("--output", type=str, help="Path to the output HDF5 file.")
args = parser.parse_args()

filename = args.filename
max_duration = args.duration * 60 * 1000  # Convert minutes to milliseconds

#This function initiates a call to ffprobe which returns a summary report about
#the file of interest.  The returned information is then parsed to extract only
#the creation time of the file.

def creation_time(filename):
    import os, sys, subprocess, shlex, re
    from subprocess import call

    cmnd = ['ffprobe', '-show_format', '-pretty', '-loglevel', 'quiet', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(filename)
    out, err =  p.communicate()
    print("==========output==========")
    print(out)
    if err:
        print("========= error ========")
        print(err)
    t = out.splitlines()
    time = str(t[14][18:-1])
    time = time[2:-2]
    return time

# Opens the video import and sets parameters
video = cv2.VideoCapture(filename)

# Checks to see if the video was properly imported
status = video.isOpened()

if status:
    FPS = video.get(cv2.CAP_PROP_FPS)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_lapse = (1 / FPS) * 1000

    # Initializes time origin of the video
    t = creation_time(filename)
    initial = dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
    timestamp = initial

    # Initializes the frame counter
    current_frame = 0
    start = time.time()
    print("Processing....")
    print(" ")

    # Update the output file path to use the argument
    output_file = args.output if args.output else filename[:-4] + '_data.h5'

    # Modify the HDF5 file to save frames with names as 'frame_{timestamp}'
    h5_file = h5py.File(output_file, 'w')
    frames_group = h5_file.create_group('frames')

    # Reads through each frame, calculates the timestamp, and saves the frame
    while current_frame < total_frames:
        success, image = video.read()
        # change the color space from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not success:
            break

        elapsed_time = video.get(cv2.CAP_PROP_POS_MSEC)
        if elapsed_time > max_duration:
            break

        current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = initial + dt.timedelta(microseconds=elapsed_time * 1000)
        timestamp_str = int(timestamp.timestamp() * 1e9)  # Convert to nanoseconds

        # Reduce the frame rate by skipping frames
        if (current_frame + 1) % 2 != 0:  # Skip every other frame for 30fps to 15fps reduction
            continue

        # Show progress when saving the h5py data
        print(f"Saving frame {current_frame}/{total_frames}...")

        # Save the frame with the name 'frame_{timestamp}'
        frames_group.create_dataset(f'frame_{timestamp_str}', data=image, dtype='uint8')

    video.release()
    cv2.destroyAllWindows()

    # Close the HDF5 file
    h5_file.close()

    # Calculate how long the timestamping took
    duration = (time.time() - float(start)) / 60

    print("Video has been timestamped")
    print("This video took:" + str(duration) + " minutes")

else:
    print("Error: Could not load video")