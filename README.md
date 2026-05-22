# Walkway identification and attribute analysis

The main procedure is: 
1. Track participants using EfficientTAM.
2. Pose estimation to extract participant's feet as spatial cues given the tracking results. 
3. Walkway segmentation from Large Vision models (Grounding DINO + SAM)
4. Filter the walkway candidates from step 3 to extract the actual walkway stepped by participant
5. Walkway material prediction
6. Walkway width measurement

## 0TrackingParticipants 

### 0.1 Sample Video frames from GoPro Video
```
cd scripts
python 0gopro_video_to_h5py.py --filename /path/to/gopro_video.MP4 --start --duration time --output /path/to/output_frames.h5
```

### 0.2 Use efficientTAM to track participants. 
For tracking purpose, use the first image to define the region on participants. Use warmup_only option for defining the participants in the first frame. Revise the points and run the inference without warmup_only option. 
```
cd EfficientTAM/notebooks
python 1gopro_tracking.py --h5_file /path/to/gopro_frames.h5 --output_file /path/to/pp_masks.h5 --warmup_only
```
Check the segmented results and specify the region to run the tracking inference. 
```
python 1gopro_tracking.py --h5_file /path/to/gopro_frames.h5 --output_file /path/to/pp_masks.h5 --points 900,400 900,550 --labels 1 1

```
Remove tmp files under /tmp/resized* to save the space.

## 1Openpose for feet detection
Check out openpose to install locally or in docker

```
# Crop the participant region given the tracking results
python 1crop_from_masks.py
--frame_h5 /path/to/gopro_frames.h5   
--mask_h5 /path/to/pp_masks.h5   
--output_dir /path/to/cropped   
--padding_scale 1.25 
--save_to_h5 
--file_suffix 01
```

In Docker run, dir should under ~/workspace/openpose-docker, if cuda not work, stop and restart

### 1.1 Large scale implementation
Use python api:
```
cd /openpose/examples/tutorial_api_python
python3 befit_batch_process.py --input_h5 /path/to/cropped/cropped_data_01.h5 --output_h5 /path/to/cropped/openpose_01.h5
```
Output is h5py file, saving all keypoints. 

## 2GroundingSAM2
### 2.1 Ground segments first, then feet to filter
Altertiavely, we can apply grounded_sam_2 to get all candidate ground segments in video frames. Ground segments will be saved as h5 file. Then, for feet, we set a patch around feet and see if these two patches are overlapped with candidate ground segments. 

Check out Grounded-SAM-2 to install locally or in docker.
```
python 2ground_segmentation_batch_individual.py --output_h5 /home/houhao/Grounded-SAM-2/data/BF003/ground_01.h5 --frame_h5 /home/houhao/Grounded-SAM-2/data/BF003/frames_01.h5

# Filtering the ground candidates by using feet cues
python 2ground_overlap.py --keypoints_h5 ../dataset/BF010/cropped/openpose_01.h5 --ground_h5 ../dataset/BF010/ground_01.h5 --output_h5 ../dataset/BF010/filter_ground_01.h5 --offset_json ../dataset/BF010/cropped/crop_offsets_01.json
```

## 3MaterialClassification

For large scale implementation, we use h5 file to access the image data
```
python 3mat_classifier_Linear_Dino.py --support ../materials/SGWalkwayMaterials/ --frame_name ../dataset/BF002/frames_01.h5 --ground_mask ../dataset/BF002/filter_ground_01.h5 --csv --val_ratio 0 --output ../dataset/BF002/mat_pred_01
```
Material dataset can be accessed via [G_Drive_Link](https://drive.google.com/drive/folders/1Oex0lCpCWkymzPtYtujVvvE_Cd2_IZnS?usp=sharing)
The results are output with json or csv 

## 6Walkway width estimation
```
python 6walkway_width_scale.py --keypoint_json ../dataset/BF002/cropped/openpose_01.h5 --crop_offsets_json ../dataset/BF002/cropped/crop_offsets_01.json --ground_mask ../dataset/BF002/filter_ground_01.h5 --frame_name ../dataset/BF002/frames_01.h5 --real_height 1.4 --output_csv ../dataset/BF002/walkway_width_01.csv
```

## 7walkway width visualization and map to trajectory
First run width_visualize to create gaussian estimation for width, a csv file will be saved
```
python 7walkway_width_visualize.py
```
Then 
```
python 8trajectory_fusion_plot.py --traj /home/houhao/workspace/VINS-Fusion/BF002/slam_gps_01.csv --width ../dataset/BF002/walkway_width_smoothed_01.csv
```

## 8Plot feature to trajectory
```
python 8trajectory_fusion_plot.py --traj /home/houhao/workspace/VINS-Fusion/pilot10/traj/traj.txt --mat /home/houhao/workspace/PersonStepMat/dataset/pilot10/mat_pred/temporal_predictions.csv
# For gait fusion, check out gait loader and sliding window, it is the subpart of the class
```

## Verification
For verification purpose, render some sample results for each step. 
```
python 0verify_h5.py
```