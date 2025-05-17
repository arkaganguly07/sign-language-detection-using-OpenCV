import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load trained model
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Label mapping
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

while cap.isOpened():
    data_aux = []
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        print("✅ Hand detected!")  # Debugging step

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract hand landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

        print(f"📊 Feature Vector Length: {len(data_aux)}")  # Debugging step

        # Ensure correct input shape for model prediction
        if data_aux:
            try:
                prediction = model.predict([np.array(data_aux)])  # Model inference
                predicted_character = labels_dict[int(prediction[0])]
                
                # Display prediction on frame
                cv2.putText(frame, f'Predicted: {predicted_character}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                print(f'🎯 Predicted Character: {predicted_character}')
            except Exception as e:
                print(f"⚠️ Model Prediction Error: {e}")

    else:
        print("❌ No hands detected")  # Debugging step

    # Display the frame
    cv2.imshow('Hand Sign Recognition', frame)

    # Exit condition: Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources properly
cap.release()
cv2.destroyAllWindows()
