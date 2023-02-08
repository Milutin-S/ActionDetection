import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:

            # Process results
            label = classification.classification[0].index
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))

            # Extract Coordinates
            coords = tuple(np.multiply(np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)), [640, 480]).astype(int))

            output = text, coords

    return output

def R_L_hand(results):
    for idx, res in enumerate(results.multi_handedness):
        # print(type(res))
        # print(res)
        # print(res.classification[0].label)
        if res.classification[0].label == 'Left':
            print(res.classification[0].label)
            print(idx)

def extract_keypoints(results):
    rh = []
    if results.multi_hand_landmarks:
        for marker in results.multi_hand_world_landmarks[0].landmark:
            rh.append(np.array([marker.x, marker.y, marker.z]))
    else:
        rh = np.zeros(21*3)

    return np.asarray(rh).flatten()

# For webcam input:
cap = cv2.VideoCapture(2)

# length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
# print("Video num of frame: ", length )
# exit()
# cap = cv2.VideoCapture(2)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640,480))

with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    imageVid = image
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # print(results)
    # print(type(results))
    # np.savetxt('testR.out', extract_keypoints(results), delimiter=',')
    # results = hands.process(cv2.flip(image, 1))
    # np.savetxt('testL.out', extract_keypoints(results), delimiter=',')
    # exit()
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        # R_L_hand(results)
    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Hands', image)

    # out.write(imageVid)

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()