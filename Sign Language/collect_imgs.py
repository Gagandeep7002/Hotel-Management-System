import os
import cv2
import mediapipe as mp

DATA_DIR = './data'
CLIP_LENGTH = 30  # Number of frames per video clip
DATASET_SIZE = 30  # Number of clips per class

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def create_class_dirs(num_classes):
    for i in range(num_classes):
        class_dir = os.path.join(DATA_DIR, str(i))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

def collect_data(num_classes):
    cap = cv2.VideoCapture(0)
    
    create_class_dirs(num_classes)

    for j in range(num_classes):
        class_dir = os.path.join(DATA_DIR, str(j))
        print(f'Collecting data for class {j}')
        
        for i in range(DATASET_SIZE):
            print(f'Recording clip {i + 1}')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(os.path.join(class_dir, f'{i}.avi'), fourcc, 20.0, (640, 480))

            frame_count = 0
            while frame_count < CLIP_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process the frame and extract hand landmarks
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                
                out.write(frame)
                cv2.imshow('frame', frame)
                frame_count += 1
                cv2.waitKey(30)

            out.release()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    num_classes = int(input("Enter the number of classes: "))
    collect_data(num_classes)
