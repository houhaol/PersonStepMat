# 0TrackingParticipants 
Downsampling the video frames and merge them in windows. files under BEFIT\scripts
Gopro timestamp can be genereated from py_gpmf_parser, see utilis scripts. Timestamps are in relative timeframes. 

## 0.1 Large scale implementatiom, efficientTAM
For large scale implementation, we use efficientTAM instead. \
We need to save the video data accordingly in npy file, so we have
```
# under efficientTAM directory notebooks
python 0gopro_video_to_h5py.py --filename /home/houhao/workspace/VINS-Fusion/pilot10/GX010048.MP4 --start --duration 660 --output ../dataset/pilot10/frames.h5
```
For tracking purpose, use the first image to define the region on participants. Use warmup_only option for defining the participants in the first frame. Revise the points and run the inference without warmup_only option. 
Then revise the code to run the tracking inference. 
```
python 1gopro_tracking.py --h5_file /home/houhao/workspace/PersonStepMat/dataset/BF002/frames_03.h5 --output_file /home/houhao/workspace/PersonStepMat/dataset/BF002/pp_masks_03.h5 --warmup_only

python 1gopro_tracking.py --h5_file /home/houhao/workspace/PersonStepMat/dataset/BF002/frames_03.h5 --output_file /home/houhao/workspace/PersonStepMat/dataset/BF002/pp_masks_03.h5 --points 900,400 900,550 --labels 1 1
# Don't forget to remove tmp files under /tmp/resized* to save the space

# for sequentially processing, run the run_tracking.bash file

# For verification purpose, render some masks results enabling the inspect_pp_tracking function. 
python 0verify_h5.py
```
# 1Openpose for feet detection
```
conda activate sam

python 1crop_from_masks.py   --frame_h5 ../dataset/BF001/frames_01.h5   --mask_h5 ../dataset/BF001/pp_masks_01.h5   --output_dir ../dataset/BF001/cropped   --padding_scale 1.25 --save_to_h5 --file_suffix 01
```

In Docker run, dir should under ~/workspace/openpose-docker, if cuda not work, stop and restart
```
sudo docker exec -it b9f0bb72fadc /bin/bash
```
## 1.1 Large scale implementation
For large scale implementation, have to use python api. 
```
cd /openpose/examples/tutorial_api_python
python3 befit_batch_process.py --input_h5 /home/houhao/BF001/cropped/cropped_data_01.h5 --output_h5 /home/houhao/BF001/cropped/openpose_01.h5
```
Output is h5py file, saving all keypoints. 

# 2GroundingSAM2
## 2.1 Ground segments first, then feet to filter
Altertiavely, we can apply grounded_sam_2 to get all candidate ground segments in video frames. Ground segments will be saved as h5 file. Then, for feet, we set a patch around feet and see if these two patches are overlapped with candidate ground segments. 
```
sudo docker exec -it c75ab3b1739e /bin/bash
# files under Grounded-SAM-2/data/pilot0/frames/. nano local_video_frames.py
python 2ground_segmentation_batch_individual.py --output_h5 /home/houhao/Grounded-SAM-2/data/BF003/ground_01.h5 --frame_h5 /home/houhao/Grounded-SAM-2/data/BF003/frames_01.h5

python  2ground_overlap.py --keypoints_h5 ../dataset/pilot10/openpose.h5 --ground_h5 ../dataset/pilot10/ground.h5 --output_h5 ../dataset/pilot10/filter_ground.h5 --offset_json ../dataset/pilot10/cropped/crop_offsets.json

# For verification results, render some segmentations results. 
python 0verify_h5.py

# render raw ground results
#python 2ground_visualize.py --ground_h5 ../dataset/pilot10/ground.h5 --frame_h5 ../dataset/pilot10/frames.h5 --output_dir ../dataset/pilot10/ground_raw --interval 900

# render filter ground results
#python 2ground_visualize.py --ground_h5 ../dataset/pilot10/filter_ground.h5 --frame_h5 ../dataset/pilot10/frames.h5 --output_dir ../dataset/pilot10/ground_verify --interval 900
```

# 3MaterialClassification
Use the few shot learning to classify the ground material. \
```
python 3mat_classifier.py \
  --support ../support_data/mat_support \
  --infer ../dataset/ground_mask \
  --output ../dataset/mat_predictions \
  --val_ratio 0.0
```
For large scale implementation, we use h5 file to access the image data
```
python 3mat_classifier_Linear_Dino.py --support ../materials/SGWalkwayMaterials/ --frame_name ../dataset/BF002/frames_01.h5 --ground_mask ../dataset/BF002/filter_ground_01.h5 --csv --val_ratio 0 --output ../dataset/BF002/mat_pred_01
```
The results are output with json or csv \

Fuse ground segments with keypoints json
```
python 3ground_keypoints_overlay.py     --cropped_json_dir ../dataset/pilot1_park_cut/openpose_json/     --offset_json ../dataset/pilot1_park_cut/cropped/crop_offsets.json     --full_frame_dir ../dataset/pilot1_park_cut/frames     --ground_mask_dir ../dataset/pilot1_park_cut/ground_mask     --overlay_dir ../dataset/pilot1_park_cut/output_overlay --save_video
```

Use 03mat_dataset_prepare.py to sample the frames and masks. Manually assign label and save the results. 

# 4MaterialVisualization video export
Run visulization to merge frames and overlay masks and predicted materials. 
```
python visualize_material_overlay.py \
  --frame_dir ./frames \
  --mask_dir ./masks \
  --json_file ./predictions.json \
  --output_dir ./output_overlay \
  --draw_keypoints \
  --keypoint_json_dir ./cropped_keypoints \
  --keypoint_offset_json ./offsets.json \
  --merge_video \
  --video_path overlay.mp4 \
  --fps 30
```
or for statistic analysis
```
python 4mat_plot_stat.py
```
## 4.1 Mateiral clustering and classification
```
python 4mat_cluster.py --frames ../dataset/pilot10/frames.h5 --ground_encode_features ../dataset/pilot10/ground_encode_features.h5 --masks ../dataset/pilot10/masks.h5 --output ../dataset/pilot10/
```

For visualization
```
python 4mat_visualize.py --crop_offsets_json ../dataset/pilot10/cropped/crop_offsets.json --keypoint_json ../dataset/pilot10/openpose.h5 --ground_mask ../dataset/pilot10/filter_ground.h5 --frame_name ../dataset/pilot10/frames.h5 --material_label_csv /home/houhao/workspace/PersonStepMat/dataset/pilot10/mat_pred/temporal_predictions.csv --output_vis_dir ../dataset/pilot10/mat_annotations --interval 1000
```

# 5Mateiral fusion with trajectory
npy file is obtained from gopro. see step0. offset is the start of the video clip if have.  
```
python 5mat_fuse.py \
  --traj trajectory.tum \
  --csv image_labels.csv \
  --npy image_timestamps.npy \
  --offset 240.0 \
  --output merged.csv
```

# 6Walkway width estimation
```
python 6walkway_width_scale.py --keypoint_json ../dataset/BF002/cropped/openpose_01.h5 --crop_offsets_json ../dataset/BF002/cropped/crop_offsets_01.json --ground_mask ../dataset/BF002/filter_ground_01.h5 --frame_name ../dataset/BF002/frames_01.h5 --real_height 1.4 --output_csv ../dataset/BF002/walkway_width_01.csv
```


# 7walkway width visualization and map to trajectory
```
# First run width_visualize to create gaussian estimation for width, a csv file will be saved
python 7walkway_width_visualize.py

# Then 
python 8trajectory_fusion_plot.py --traj /home/houhao/workspace/VINS-Fusion/BF002/slam_gps_01.csv --width ../dataset/BF002/walkway_width_smoothed_01.csv
```

# 8Plot feature to trajectory
```
python 8trajectory_fusion_plot.py --traj /home/houhao/workspace/VINS-Fusion/pilot10/traj/traj.txt --mat /home/houhao/workspace/PersonStepMat/dataset/pilot10/mat_pred/temporal_predictions.csv
```

count labels from csv
=TAKE(SORTBY(UNIQUE(K2:K10089),COUNTIF(K2:K10089,UNIQUE(K2:K10089)),-1),5)