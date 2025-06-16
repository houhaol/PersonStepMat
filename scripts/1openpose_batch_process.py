# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import h5py

import pyopenpose as op


# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--input_h5_path", required=True, help="Path to input HDF5 file containing cropped frames.")
parser.add_argument("--output_h5_path", required=True, help="Path to output HDF5 file to save pose keypoints.")
args = parser.parse_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../models/"


params["number_people_max"] = 1  # Only detect one person

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Load cropped data from HDF5 file
input_h5_path = args.input_h5_path
output_h5_path = args.output_h5_path

with h5py.File(input_h5_path, 'r') as input_h5:
    cropped_frames = input_h5['cropped_frames']

    # Create a new HDF5 file to save pose keypoints
    with h5py.File(output_h5_path, 'w') as output_h5:
        pose_keypoints_group = output_h5.create_group("pose_keypoints")

        # Iterate through each frame in the HDF5 file
        for frame_name in cropped_frames:
            frame_data = cropped_frames[frame_name][:]  # Load cropped frame data

            # Process the frame with OpenPose
            datum = op.Datum()
            datum.cvInputData = frame_data
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
#            cv2.imwrite("test.png", datum.cvOutputData) 
#            import pdb; pdb.set_trace()
            # Extract pose keypoints (x, y positions only)
            if datum.poseKeypoints is not None:
                keypoints = datum.poseKeypoints[0, :, :2]  # Assuming single person detection
                pose_keypoints_group.create_dataset(frame_name, data=keypoints, dtype="float32")

#print(f"✅ Pose keypoints saved to: {output_h5_path}")
