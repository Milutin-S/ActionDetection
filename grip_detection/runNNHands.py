import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from PIL import ImageFont, ImageDraw, Image

### Setup data collection

# Actions that we try to detect
actions = np.array(['cylindrical', 'pinch', 'lateral', 'rest', 'noHand'])
window_length = 4

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
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints(results):
    rh = []
    zeroTrans = np.zeros(3)
    handLocMarks = []
    idx = 0
    if results.multi_hand_landmarks:
        for marker in results.multi_hand_world_landmarks[0].landmark:

            # To local
            # if idx == 0:
            #     zeroTrans[0], zeroTrans[1], zeroTrans[2] = [marker.x, marker.y, marker.z]
            # idx += 1
            #
            # handLocMarks.append(np.array([marker.x - zeroTrans[0], marker.y - zeroTrans[1], marker.z - zeroTrans[2]]))
            # End local
            rh.append(np.array([marker.x, marker.y, marker.z]))
    else:
        rh = np.zeros(21*3)
        ## Local
        # handLocMarks = np.zeros(21*3)
    # print("+++++++++++++++++++++++++++++++++++++")
    # print(results.multi_hand_world_landmarks) #[0].landmark)
    # print(handLocMarks)
    # print("-------------------------------------")
    # print(rh)

    return np.asarray(rh).flatten()
    # return np.asarray(handLocMarks).flatten()

### Create and load network
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(window_length,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# model.load_weights('weights/date3_10_22_5wl_60e_600ns/my_best_model.epoch53-loss0.12.h5')
# model.load_weights('weights/final/big/my_best_model.epoch89-loss0.01.h5')
# model.load_weights('weights/final/45/3x10/7/my_best_model.epoch89-loss0.01.h5')
# model.load_weights('weights/final/45/3x30/10/my_best_model.epoch88-loss0.00.h5')

# model.load_weights('weights/final/75/3x60/7/my_best_model.epoch84-loss0.01.h5')
# model.load_weights('weights/final/60/3x60/7/my_best_model.epoch89-loss0.01.h5')
# model.load_weights('weights/final/45/3x60/4/my_best_model.epoch90-loss0.03.h5')
# model.load_weights('weights/final/45/3x60/additional/45_4_180e/my_best_model.epoch174-loss0.01.h5')

model.load_weights('weights/final/45/full_4_rest_fix/my_best_model.epoch163-loss0.01.h5')

# model.load_weights('weights/final/60/full_7_rest_fix/my_best_model.epoch90-loss0.02.h5')
# model.load_weights('weights/final/60/full_7_rest_fix/my_best_model.epoch177-loss0.01.h5')

# model.load_weights('weights/final/60/full_7_rest_fix_local/my_best_model.epoch160-loss0.01.h5')
# model.load_weights('weights/final/45/full_4_rest_fix_local/my_best_model.epoch179-loss0.02.h5')


colors = [(245,117,16), (117,245,16), (16, 117, 245), (150, 105, 195), (51, 51, 255)] # orange, green, blue
def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # if res[np.argmax(res)] > treshold:
        (width, height), baseline = cv2.getTextSize(actions[num], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.rectangle(output_frame, (640, num * (height+12)), (640 - int(prob * 110), height + 10 + num * (height+12)), colors[num], -1)
        cv2.putText(output_frame, actions[num], (640 - width, height+5+num*(height+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        # ImageDraw.Draw(output_frame).text((len(actions[num]*100), 85+num*40), actions[num], font=ImageFont.monospace)
    return  output_frame

# Detection variables
sequence = []
sentence = []
treshold = 0.85
predictions = []
res = np.zeros(len(actions))
print("res:", res)
forPlot = []

cap = cv2.VideoCapture(2)
# Set mediapipe model
with mp_hands.Hands(max_num_hands= 1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        # Read feed
        # 480 x 640
        ret, frame = cap.read()
        # print(frame.shape) check
        # Make detections
        # image, results = mediapipe_detection(frame, hands) # right hand, 600
        image, results = mediapipe_detection(cv2.flip(frame, 1), hands) # left hand. 60
        # print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic right hand
        # if results.multi_hand_landmarks:
        #     for idx, res in enumerate(results.multi_handedness):
        #         # print(type(res[1]))
        #         if res.classification[0].label == 'Left':
        #             keypoints = extract_keypoints(results[idx])
        #             #print(res[1].classification[0].label)
        #         elif res.classification[0].label == 'Right':
        #             imageF, resultsF = mediapipe_detection(cv2.flip(frame, 1), hands)
        #             keypoints = extract_keypoints(resultsF[idx])
        #         else:
        #             keypoints = np.asarray(np.zeros(21 * 3)).flatten()
        # else:
        #     keypoints = np.asarray(np.zeros(21 * 3)).flatten()
        keypoints = extract_keypoints(results) # old
        sequence.append(keypoints)
        sequence = sequence[-window_length:]

        if len(sequence) == window_length:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            forPlot.append(res)
            forPlot = forPlot[-120:]
            # print(type(res))
            # print("RES: ", res)
            # print("RES PART: ", forPlot[2])
            # predictions.append(np.argmax(res))


            # Viz probabilities
            image = prob_viz(res, actions, image)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        # print("Frames per second : {0}".format(cap.get(cv2.CAP_PROP_FPS)))

        # Break gracefully on Esc
        if cv2.waitKey(5) & 0xFF == 27:
            # print("RES PART: ", forPlot[2])
            # for i, part  in enumerate(forPlot):
            gripPred = np.array(forPlot)
            # print(gripPred)
            frames = list(range(0, len(forPlot)))
            # plot lines
            plt.plot(frames, gripPred[:,0], color="blue", label="cylindrical")
            plt.plot(frames, gripPred[:, 1], color="green", label="pinch")
            plt.plot(frames, gripPred[:, 2], color="orange", label="lateral")
            plt.plot(frames, gripPred[:, 3], color="pink", label="rest")
            plt.plot(frames, gripPred[:, 4], color="red", label="noHand")
            # plt.plot(frames, x, label="line 2")
            plt.legend()
            plt.show()
            break
cap.release()
cv2.destroyAllWindows()