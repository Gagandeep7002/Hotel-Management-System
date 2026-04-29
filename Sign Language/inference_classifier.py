import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model, scaler, and class labels
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
scaler = model_dict['scaler']
class_labels = model_dict['class_labels']  # Ensure this dictionary has 26 keys

# Initialize video capture and Mediapipe Hands
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

FEATURE_VECTOR_LENGTH = 63  # 21 landmarks * 3 coordinates per landmark

def extract_features_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    feature_vector = np.zeros(FEATURE_VECTOR_LENGTH)  # Initialize with zeros
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for i, landmark in enumerate(landmarks.landmark):
                feature_vector[i*3] = landmark.x
                feature_vector[i*3+1] = landmark.y
                feature_vector[i*3+2] = landmark.z

    return feature_vector

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to improve processing speed
    frame = cv2.resize(frame, (640, 480))

    features = extract_features_from_frame(frame)
    features = features.reshape(1, -1)  # Reshape for the model input

    # Ensure features are scaled consistently
    features_scaled = scaler.transform(features)

    # Predict the sign
    prediction = model.predict(features_scaled)
    predicted_class = int(prediction[0])
    predicted_character = class_labels.get(predicted_class, 'Unknown')

    # Draw hand landmarks and predicted character
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display prediction
    cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
