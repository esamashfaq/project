import numpy as np
import os
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from KeypointsExtraction import draw_landmarks, image_process, keypoint_extraction
import keyboard

# Path to data and actions defined during training
PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))

# Load the trained model
model = load_model('my_model.h5')

# Initialize prediction and sentence-related lists
sentence, keypoints, last_prediction = [], [], None
cooldown_frames, cooldown_threshold = 0, 20  # Cooldown period of 20 frames after each prediction

# Open camera for capturing
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

with mp.solutions.holistic.Holistic(min_detection_confidence=0.80, min_tracking_confidence=0.80) as holistic:
    while cap.isOpened():
        # Capture frame from camera
        ret, image = cap.read()
        if not ret:
            break
        
        # Process frame and extract keypoints
        results = image_process(image, holistic)
        draw_landmarks(image, results)
        keypoints.append(keypoint_extraction(results))

        # Predict every 10 frames if cooldown is not active
        if len(keypoints) == 20 and cooldown_frames == 0:
            keypoints = np.array(keypoints)
            prediction = model.predict(keypoints[np.newaxis, :, :])
            keypoints = []

            # Check if the prediction exceeds threshold
            if np.max(prediction) >= 0.99:
                predicted_action = actions[np.argmax(prediction)]
                
                # Only append if prediction differs from the last one
                if predicted_action != last_prediction:
                    sentence.append(predicted_action)
                    last_prediction = predicted_action
                    cooldown_frames = cooldown_threshold  # Activate cooldown

        # Decrease cooldown frame count
        cooldown_frames = max(0, cooldown_frames - 1)

        # Limit sentence length for display
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Reset on Spacebar press
        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction = [], [], None

        # Adjust sentence for display
        if sentence:
            sentence[0] = sentence[0].capitalize()

        # Display prediction on image
        display_text = ' '.join(sentence)
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        cv2.putText(image, display_text, (text_x, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the output
        cv2.imshow('Real-time Sign Prediction', image)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()