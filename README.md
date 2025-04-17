# 0TrackingParticipants 
Downsampling the video frames and merge them in windows. files under BEFIT\scripts
`python sample_and_merge_video.py --video E:BEFIT\Pilot5_Mar272025\TPSVideo\Pilot5_Mat_Transit.mp4 --frame_dir E:BEFIT\Pilot5_Mar272025\TPSVideo\downsampling --output E:BEFIT\Pilot5_Mar272025\TPSVideo\merged.mp4`

```conda activate track
# cd Track-Anything to run python `python app.py --device cuda:0 --mask_save False`
python app.py --device cuda:0 --mask_save False
```
Get the mask and tracking results for participants in video frames. \
Masks results are under ./Track-Anything/result/mask/508/mask_00000.npy \ 

# 1Openpose for feet detection
conda activate sam
```
python 1crop_from_masks.py \
  --frame_dir ../dataset/sampled_frames \
  --mask_dir ../dataset/masks \
  --output_dir ../dataset/cropped \
  --padding_scale 1.25
```

```
sudo docker exec -it b9f0bb72fadc /bin/bash
# In Docker run 
./build/examples/openpose/openpose.bin --image_dir /home/houhao/PersonStepMat/dataset/cropped --write_json /home/houhao/PersonStepMat/dataset/openpose_json/ --display 0 --num_gpu 1 --render_pose 0 --number_people_max 1
# if cuda not work, stop and restart
```

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

# 3MaterialClassification
Use the few shot learning to classify the ground material. \
```
python 3mat_classifier.py \
  --support ../support_data/mat_support \
  --infer ../dataset/ground_mask \
  --output ../dataset/mat_predictions \
  --val_ratio 0.0
```
Run visulization to merge frames and overlay masks and predicted materials. 
```
python 4mat_visualize.py   --frame_dir ../dataset/pilot5_cilp_mat/frames   --mask_dir ../dataset/pilot5_cilp_mat/ground_mask   --json_file ../dataset/pilot5_cilp_mat/mat_predictions/temporal_predictions.json   --output_dir ../dataset/pilot5_cilp_mat/visualized   --alpha 0.6   --merge_video   --video_path ../dataset/pilot5_cilp_mat/visualized_mat_overlay.mp4   --fps 1
```