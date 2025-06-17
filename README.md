# 0TrackingParticipants 
Downsampling the video frames and merge them in windows. files under BEFIT\scripts
Gopro timestamp can be genereated from py_gpmf_parser, see utilis scripts. Timestamps are in relative timeframes. 

## 0.1 Large scale implementatiom, efficientTAM
For large scale implementation, we use efficientTAM instead. \
We need to save the video data accordingly in npy file, so we have
```
python 0gopro_video_to_h5py.py --filename /home/houhao/workspace/VINS-Fusion/pilot10/GX010048.MP4 --duration 0.25 -
-output ../dataset/pilot10/frames.h5
```
For tracking purpose, use the first image to define the region on participants. Use jupyer for better visualization. Then revise the code to run the tracking inference. 
```
python example_video_gopro.py --h5_file /home/houhao/workspace/VINS-Fusion/pilot10/GX010048_data_test.h5 --output_file /home/houhao/workspace/VINS-Fusion/pilot10/masks_data.h5

For verification purpose, render some masks results. 
python 0render_mask_results.py
```
# 1Openpose for feet detection
```
conda activate sam

python 1crop_from_masks.py   --frame_h5 ../dataset/pilot10/GX010048_data_test.h5   --mask_h5 ../dataset/pilot10/masks.h5   --output_dir ../dataset/pilot10/cropped   --padding_scale 1.25 --save_to_h5
```

In Docker run, dir should under ~/workspace/openpose-docker, if cuda not work, stop and restart
```
sudo docker exec -it b9f0bb72fadc /bin/bash
```
## 1.1 Large scale implementation
For large scale implementation, have to use python api
```
python3 befit_batch_process.py --input_h5 /home/houhao/pilot10/cropped/cropped_data.h5 --output_h5 /home/houhao/pilot10/openpose.h5
```
Output is h5py file, saving all keypoints. 

# 2GroundingSAM2
## 2.1 Ground segments first, then feet to filter
Altertiavely, we can apply grounded_sam_2 to get all candidate ground segments in video frames. Ground segments will be saved as h5 file. Then, for feet, we set a patch around feet and see if these two patches are overlapped with candidate ground segments. 
```
sudo docker exec -it c75ab3b1739e /bin/bash
# files under Grounded-SAM-2/data/pilot0/frames/. nano local_video_frames.py
python 2ground_segmentation_batch.py --output_h5 /home/houhao/Grounded-SAM-2/data/pilot10/ground.h5 --frame_h5 /home/houhao/Grounded-SAM-2/data/pilot10/GX010048_data_test.h5

python  2ground_overlap.py --keypoints_h5 ../dataset/pilot10/openpose.h5 --ground_h5 ../dataset/pilot10/ground.h5 --output_h5 ../dataset/pilot10/filter_ground.h5 --offset_json ../dataset/pilot10/cropped/crop_offsets.json

# For verification results, render some segmentations results. 
# render raw ground results
python 2ground_visualize.py --ground_h5 ../dataset/pilot10/ground.h5 --frame_h5 ../dataset/pilot10/frames.h5 --output_dir ../dataset/pilot10/ground_raw --interval 900

# render filter ground results
python 2ground_visualize.py --ground_h5 ../dataset/pilot10/filter_ground.h5 --frame_h5 ../dataset/pilot10/frames.h5 --output_dir ../dataset/pilot10/ground_verify --interval 900
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
The results are output with json or csv \

Fuse ground segments with keypoints json
```
python 3ground_keypoints_overlay.py     --cropped_json_dir ../dataset/pilot1_park_cut/openpose_json/     --offset_json ../dataset/pilot1_park_cut/cropped/crop_offs
ets.json     --full_frame_dir ../dataset/pilot1_park_cut/frames     --ground_mask_dir ../dataset/pilot1_park_cut/ground_mask     --overlay_dir ../dataset/pilot1_park_cut/output_overlay --save_video
```

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
python 6walkway_width_scale.py --keypoint_json ../dataset/pilot10/openpose.h5 --crop_offsets_json ../dataset/pilot10/cropped/crop_offsets.json --ground_mask ../dataset/pilot10/filter_ground.h5 --frame_name ../dataset/pilot10/frames.h5 --real_height 1.5 --output_csv ../dataset/pilot10/walkway_width.csv
```


# 7walkway width visualization
```
python 7walkway_width_visualize.py
```