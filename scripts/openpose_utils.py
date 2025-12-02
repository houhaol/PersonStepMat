import numpy as np

# ----------------- BODY_25 KEYPOINT CONSTANTS -----------------
BODY_25_PAIRS_RENDER = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (0, 1), (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 21), (11, 22), (14, 19), (19, 20), (11, 24), (22, 23)
]

POSE_COLORS = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170),
    (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
    (128, 128, 255), (128, 255, 128), (255, 128, 128), (255, 255, 128)
]

def get_shoulder_avg(keypoints):
    # Shoulders: 2 (RShoulder), 5 (LShoulder)
    s1 = keypoints[2]
    s2 = keypoints[5]
    return [(s1[0]+s2[0])/2, (s1[1]+s2[1])/2]

def get_lowest_foot(keypoints):
    # Left: 21, 19, 20; Right: 24, 22, 23
    left = [keypoints[i] for i in [19,20,21]]
    right = [keypoints[i] for i in [22,23,24]]
    left_y = [pt[1] for pt in left]
    right_y = [pt[1] for pt in right]
    left_idx = np.argmax(left_y)
    right_idx = np.argmax(right_y)
    left_foot = left[left_idx]
    right_foot = right[right_idx]
    # Pick the lower (max y)
    return left_foot if left_foot[1] > right_foot[1] else right_foot

def get_wrist_avg(keypoints):
    # Wrists: 4 (RWrist), 7 (LWrist)
    w1 = keypoints[4]
    w2 = keypoints[7]
    return [(w1[0]+w2[0])/2, (w1[1]+w2[1])/2]