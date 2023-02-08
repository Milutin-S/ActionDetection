import cv2
import numpy as np
import os
import mediapipe as mp


### Setup data collection
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('datasets/L_final_3x60ns/75')

# Actions that we try to detect
actions = np.array(['noHand'])

# Thirty videos worth of data
no_sequences = 180

# Videos are going to be 30 frames in length
sequence_length = 75

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Loop through actions
for action in actions:
    # Loop through sequences aka videos
    for sequence in range(no_sequences):
        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):
            # Export keypoints
            keypoints = np.asarray(np.zeros(21*3)).flatten()
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

