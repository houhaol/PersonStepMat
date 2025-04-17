# openpose-docker
A docker build file for CMU openpose with Python API support

https://hub.docker.com/r/cwaffles/openpose

### Requirements
- Nvidia Docker runtime: https://github.com/NVIDIA/nvidia-docker#quickstart
- CUDA 10.0 or higher on your host, check with `nvidia-smi`

### Example
`docker run -it --rm --gpus all -e NVIDIA_VISIBLE_DEVICES=0 cwaffles/openpose`
The Openpose repo is in `/openpose`


### Leverage tracking results

cd Track-Anything to run python `python app.py --device cuda:0 --mask_save False`
<!-- docker run --gpus all --it openpose:latest -->

Sample frames, crop masks for openpose use
```
python sample_video_frames.py \
  --video_path /home/houhao/workspace/VINS-Mono/dataset/Pilot2/upstairs_eye_tps.mp4 \
  --output_dir ./sampled_frames
  
python crop_from_masks.py \
  --frame_dir ./pilot2_test/sampled_frames \
  --mask_dir ./pilot2_test/masks \
  --output_dir ./pilot2_test/cropped \
  --padding_scale 1.25
```
In Docker run `
 ./build/examples/openpose/openpose.bin --image_dir /home/houhao/pilot2_test/cropped --write_json /home/houhao/pilot2_test/openpose_json/ --display 0 --num_gpu 1 --render_pose 0`

After that, map openpose detected keypoints on full frame
```
python map_keypoints.py \
  --cropped_json_dir ./pilot2_test/openpose_json \
  --offset_json ./pilot2_test/cropped/crop_offsets.json \
  --output_dir ./pilot2_test/output_json_full_frame \
  --full_frame_dir ./pilot2_test/sampled_frames \
  --overlay_dir ./pilot2_test/overlays
```
