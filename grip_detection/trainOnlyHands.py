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
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

### Setup data collection
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('datasets/R_final_3x60ns/45_fix_rest_local')

# Actions that we try to detect
actions = np.array(['cylindrical', 'pinch', 'lateral', 'rest', 'noHand'])

# Thirty videos worth of data
no_sequences = 180 # 540

# Videos are going to be 30 frames in length
sequence_length = 45

window_length = 4

# (sequence_length - window_length + 1)*200*3 ### 23.400, 10, 63; 1 - 7.800, 10, 63

label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)

sequences, labels = [], []
## Whole data
for action in actions:
    for sequence in range(no_sequences):
        for i in range(sequence_length - window_length + 1):
            window = []
            for frame_num in range(i , i + window_length): # sequence_length needs to change in order to take only a part of capture data
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
## Parts of data
# for action in actions:
#     for sequence in range(no_sequences):
#         if (sequence > 49 and sequence < 60) or (sequence > 109 and sequence < 120) or (sequence > 169 and sequence < 180):
#             for i in range(sequence_length - window_length + 1):
#                 window = []
#                 for frame_num in range(i , i + window_length): # sequence_length needs to change in order to take only a part of capture data
#                     res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#                     window.append(res)
#                 sequences.append(window)
#                 labels.append(label_map[action])

## Big data
# for action in actions:
#     for sequence in range(no_sequences):
#         if (sequence >= 0 and sequence < 180):
#             sequence_length = 45
#         elif (sequence >= 180 and sequence < 360):
#             sequence_length = 60
#         else:
#             sequence_length = 75
#             for i in range(sequence_length - window_length + 1):
#                 window = []
#                 for frame_num in range(i , i + window_length): # sequence_length needs to change in order to take only a part of capture data
#                     res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#                     window.append(res)
#                 sequences.append(window)
#                 labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

filepath = 'weights/final/45/full_4_rest_fix_local/my_best_model.epoch{epoch:02d}-loss{loss:.2f}.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

callbacks = [checkpoint, tb_callback]

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(window_length,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=180, callbacks=callbacks)

model.summary()

# Make Predictions
res = model.predict(X_test)
count = 0
for i in range(len(res)):
    if actions[np.argmax(res[i])] == actions[np.argmax(y_test[i])]:
        count += 1

print("TEST RESULTS: " +str(count) +"/"+str(len(res)))

# Save Weights
model.save('weights/final/45/full_4_rest_fix_local/BackUpLASTModelSaver.h5')

### Quick logs
## 1 10 22
# 2 - 7 frames
# 3 - 7 frames
## 2 10 22
# 1 - 8 frames
# 2 - 6 frames, new pinch data
# 3 - 7 frames, 69 epochs

yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print("Multilabel Confusion Matrix (test data): \n", multilabel_confusion_matrix(ytrue, yhat))
print("Accuracy score (test data): \n", accuracy_score(ytrue, yhat))

print("------------------------------------------------")
yhat = model.predict(X_train)
ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print("Multilabel Confusion Matrix (train data): \n", multilabel_confusion_matrix(ytrue, yhat))
print("Accuracy score (train data): \n", accuracy_score(ytrue, yhat))