import os
import pickle
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp

DATA_DIR = './data'
CLIP_LENGTH = 30  # Number of frames per video clip
FEATURE_VECTOR_LENGTH = 63  # 21 landmarks * 3 coordinates per landmark

data = []
labels = []
class_labels = {}

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB and process it with MediaPipe Hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Extract landmark coordinates
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                feature_vector = []
                for landmark in landmarks.landmark:
                    feature_vector.extend([landmark.x, landmark.y, landmark.z])
                if len(feature_vector) == FEATURE_VECTOR_LENGTH:
                    features.append(feature_vector)
    
    cap.release()

    # Pad or truncate features to ensure consistent length
    if len(features) < CLIP_LENGTH:
        features.extend([np.zeros(FEATURE_VECTOR_LENGTH)] * (CLIP_LENGTH - len(features)))
    elif len(features) > CLIP_LENGTH:
        features = features[:CLIP_LENGTH]

    return np.array(features).flatten()

# Collect class labels and data
for dir_index, dir_ in enumerate(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        class_labels[dir_index] = dir_
        for video_path in os.listdir(dir_path):
            video_full_path = os.path.join(dir_path, video_path)
            features = extract_features_from_video(video_full_path)
            data.append(features)
            labels.append(dir_index)

# Encode labels to integers
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Save data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': encoded_labels, 'class_labels': class_labels}, f)
