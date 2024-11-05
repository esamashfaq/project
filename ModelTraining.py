# ModelTraining.py
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense
from itertools import product
from sklearn import metrics

PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))
sequences, frames = 30, 20
label_map = {label: num for num, label in enumerate(actions)}

landmarks, labels = [], []
for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
        temp.append(npy)
    landmarks.append(temp)
    labels.append(label_map[action])

X, Y = np.array(landmarks), to_categorical(labels).astype(int)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

from tensorflow.keras.layers import Input

model = Sequential([
    Input(shape=(frames, 126)),  # Update input shape for hand landmarks only
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(32, return_sequences=True, activation='relu'),
    LSTM(64, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, Y_train, epochs=100)
model.save('my_model.h5')

predictions = np.argmax(model.predict(X_test), axis=1)
test_labels = np.argmax(Y_test, axis=1)
accuracy = metrics.accuracy_score(test_labels, predictions)
print(f"Test Accuracy: {accuracy}")
