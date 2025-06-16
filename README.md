# 0TrackingParticipants 
Downsampling the video frames and merge them in windows. files under BEFIT\scripts
Gopro timestamp can be genereated from py_gpmf_parser, see utilis scripts. Timestamps are in relative timeframes. 
```
python sample_and_merge_video.py --video E:BEFIT\Pilot5_Mar272025\TPSVideo\Pilot5_Mat_Transit.mp4 --frame_dir E:BEFIT\Pilot5_Mar272025\TPSVideo\downsampling --output E:BEFIT\Pilot5_Mar272025\TPSVideo\merged.mp4
or
python sample.py --video /path/to/video --timestamps path/to/timestamp --output output
```

```conda activate track
# cd Track-Anything to run python `python app.py --device cuda:0 --mask_save False`
python app.py --device cuda:0 --mask_save False
```
Get the mask and tracking results for participants in video frames. \
Masks results are under ./Track-Anything/result/mask/508/mask_00000.npy \ 

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
```
# 1Openpose for feet detection
```
conda activate sam

python 1crop_from_masks.py   --frame_h5 ../dataset/pilot10/GX010048_data_test.h5   --mask_h5 ../dataset/pilot10/masks.h5   --output_dir ../dataset/pilot10/cropped   --padding_scale 1.25 --save_to_h5
```

In Docker run, dir should under ~/workspace/openpose-docker, if cuda not work, stop and restart
```
sudo docker exec -it b9f0bb72fadc /bin/bash
./build/examples/openpose/openpose.bin --image_dir /home/houhao/PersonStepMat/dataset/cropped --write_json /home/houhao/PersonStepMat/dataset/openpose_json/ --display 0 --num_gpu 1 --render_pose 0 --number_people_max 1
```
## 1.1 Large scale implementation
For large scale implementation, have to use python api
```
python3 befit_batch_process.py --input_h5 /home/houhao/pilot10/cropped/cropped_data.h5 --output_h5 /home/houhao/pilot10/openpose.h5
```
Output is h5py file, saving all keypoints. 

## 1.2 feet prompts (optional)
After that, map openpose detected keypoints on full frame in the local terminal and also use the feet as prompts for ground detection. 
```
python 1map_keypoints.py \
  --cropped_json_dir ../dataset/openpose_json \
  --offset_json ../dataset/cropped/crop_offsets.json \
  --output_dir ../dataset/output_json_full_frame \
  --full_frame_dir ../dataset/sampled_frames \
  --overlay_dir ../dataset/overlays \
  --prompt_dir ../dataset/prompt \
  --shift_y 40
```

# 2GroundingSAM2
Use the feet as the prompts to get the ground segments. left, right, logical_and, morph-clean
```
python 2sam_ground.py \
  --image_dir ../dataset/sampled_frames \
  --prompt_dir ../dataset/prompt/ \
  --output_mask_dir ../dataset/ground_mask \
  --overlay_dir ../dataset/ground_overlay \
  --checkpoint ~/workspace/EyeTrackingSam/model/sam_vit_b_01ec64.pth
  --model_type vit_b
```

## 2.1 Ground segments first, then feet to filter
Altertiavely, we can apply grounded_sam_2 to get all candidate ground segments in video frames. Ground segments will be saved as json file. Then, for feet, we set a patch around feet and see if these two patches are overlapped with candidate ground segments. 
```
sudo docker exec -it c75ab3b1739e /bin/bash
# files under Grounded-SAM-2/data/pilot0/frames/. nano local_video_frames.py
python 2ground_segmentation_batch.py --output_h5 /home/houhao/Grounded-SAM-2/data/pilot10/ground.h5 --frame_h5 /home/houhao/Grounded-SAM-2/data/pilot10/GX010048_data_test.h5

python 2ground_overlap.py --batch_keypoint_dir ../dataset/pilot5_cilp_mat/openpose_json/ --offset_json ../dataset/pilot5_cilp_mat/cropped/crop_offsets.json --batch_ground_dir ~/workspace/Grounded-SAM-2/results/pilot9/grounded_sam2_frames/ --output_prompt_dir ../dataset/pilot5_cilp_mat/ground_overlap_prompts --output_mask_dir ../dataset/pilot5_cilp_mat/ground_overlap_masks

python  2ground_overlap.py --keypoints_h5 ../dataset/pilot10/openpose.h5 --ground_h5 ../dataset/pilot10/ground.h5 --output_h5 ../dataset/pilot10/filter_ground.h5 --offset_json ../dataset/pilot10/cropped/crop_offsets.json
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
python 6walkway_width_scale.py --keypoint_json ../dataset/pilot5_cilp_mat/openpose_json/ --crop_offsets_json ../dataset/pilot5_cilp_mat/cropped/crop_offsets.json --ground_mask ../dataset/pilot5_cilp_mat/ground_overlap_masks/ --frame_name ../dataset/pilot5_cilp_mat/frames/ --real_height 1.3 --output_csv ../dataset/pilot5_cilp_mat/walkway_estimation/walkway_width.csv --output_vis_dir ../dataset/pilot5_cilp_mat/walkway_estimation/

python 6walkway_width_scale.py --keypoint_json ../dataset/pilot10/openpose.h5 --crop_offsets_json ../dataset/pilot10/cropped/crop_offsets.json --ground_mask ../dataset/pilot10/filter_ground.h5 --frame_name ../dataset/pilot10/frames.h5 --real_height 1.5 --output_csv ../dataset/pilot10/walkway_width.csv 


python 6walkway_width_depth.py --keypoint_json ../dataset/pilot5_cilp_mat/openpose_json/ --crop_offsets_json ../dataset/pilot5_cilp_mat/cropped/crop_offsets.json --ground_mask ../dataset/pilot5_cilp_mat/ground_overlap_masks/ --frame_name ../dataset/pilot5_cilp_mat/frames/ --depth_dir ../dataset/pilot5_cilp_mat/depths/ --intrinsic_matrix ../dataset/pilot5_cilp_mat/intrinsic.json --cloud_dir ../dataset/pilot5_cilp_mat/cloud
```

python 6walkway_width_depth.py --ground_mask ../dataset/pilot2/walkway_masks/ --frame_name ../dataset/pilot2/frames/ --depth_dir ../dataset/pilot2/depths/ --intrinsic_matrix ../dataset/pilot2/intrinsic.json --cloud_dir ../dataset/pilot2/cloud

# 7walkway width visualization
