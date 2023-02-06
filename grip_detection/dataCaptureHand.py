import cv2
import numpy as np
import os
import mediapipe as mp


### Setup data collection
# Path for exported data, numpy arrays
# DATA_PATH = os.path.join('datasets/L_hand_views_60ns_60fs/view_3')
# DATA_PATH = os.path.join('datasets/R_hand_views_60ns_60fs/view_3')
DATA_PATH = os.path.join('datasets/final/rest_fix_all/75')

# Actions that we try to detect
actions = np.array(['rest']) # ['cylindrical', 'pinch', 'lateral', 'rest']

# Thirty videos worth of data
no_sequences = 60

# Videos are going to be xx frames in length
sequence_length = 75

view = 'view_3'

# hello
## 0
## 1
## ...
## 29
# thanks

# i love you


for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence+120)))
        except:
            pass

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
    #                           mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    #                           mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))  # Draw face connections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
    #                           mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))  # Draw pose connections
    # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
    #                           mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))  # Draw left hand connections
    # mp_drawing.draw_landmarks(image, results.multi_hand_landmarks, mp_hands.HAND_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
    #                           mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))  # Draw right hand connections
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints(results):
    # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    # lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # rh = np.array([[res.landmark.x, res.landmark.y, res.landmark.z] for res in results.multi_hand_world_landmarks]).flatten() if results.multi_hand_world_landmarks else np.zeros(21*3)
    rh = []
    if results.multi_hand_landmarks:
        for marker in results.multi_hand_world_landmarks[0].landmark:
            rh.append(np.array([marker.x, marker.y, marker.z]))
        #print(results.multi_hand_world_landmarks[0].landmark[0].x)
    else:
        rh = np.zeros(21*3)

    # return np.concatenate([pose, face, lh, rh])
    return np.asarray(rh).flatten()

cap = cv2.VideoCapture(2) # 0 - webCam, 1 - phone, 2 - realsense


# VIDEO_PATH = os.path.join('datasets/R_final_3x60ns/45/view_1')

# ret, frame = cap.read()
# cv2.imshow('OpenCV Feed', frame)
# Set mediapipe model
with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            nameVid = str(sequence+120)+'.avi'
            # pathVid = os.path.join('videoData/rest_fix_all', str(sequence_length), view, action, nameVid)
            pathVid = os.path.join('videoData/rest_fix_all', action, nameVid)
            out = cv2.VideoWriter(pathVid, fourcc, 30, (640, 480))
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                frameVid = frame.copy()
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # Apply wait logic
                if frame_num == 0:
                    cv2.putText(frame, 'PRESS SPACE TO START COLLECTING', (30, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    while True:
                        if cv2.waitKey(10) & 0xFF == ord(' '):
                            break
                        # elif cv2.waitKey(10) & 0xFF == ord('x'):
                        #     exit()
                    # Read feed again for better results
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                else:
                    cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Make detections
                image, results = mediapipe_detection(frame, hands)
                # print(type(results))
                # print(type(results.multi_hand_world_landmarks))
                # print(results.multi_hand_world_landmarks)
                # print("++++++++++++++++++++++++")
                # print(results.multi_hand_world_landmarks[0].landmark)
                # print("++++++++++++++++++++++++")
                # print(results.multi_hand_world_landmarks[0].landmark[0].x)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Export keypoints
                keypoints = extract_keypoints(results)
                # print("1++++++++++++++++++++++++")
                # print(np.size(keypoints))
                # print("2++++++++++++++++++++++++")
                # print(keypoints)
                npy_path = os.path.join(DATA_PATH, action, str(sequence+120), str(frame_num))
                np.save(npy_path, keypoints)

                # Saving the video

                out.write(frameVid)

                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
cap.release()
cv2.destroyAllWindows()