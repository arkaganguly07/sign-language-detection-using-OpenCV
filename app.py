from flask import Flask, render_template, url_for, request, redirect, Response, flash, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import mediapipe as mp
import cv2
import random


app = Flask(__name__)
CORS(app)

# Load model
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get image data from request
        image_data = request.json.get('image_data', '')
        
        # Convert base64 image to numpy array
        if image_data and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
            
        import base64
        jpg_original = base64.b64decode(image_data)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)
        
        # Process the frame with mediapipe
        data_aux = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        prediction_result = ""
        confidence = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            
            # Ensure we have the right number of features
            if data_aux:
                try:
                    # Pad if necessary (adjust based on your model's expected input)
                    max_length = 42 * 2  # 21 landmarks with x,y coordinates
                    if len(data_aux) < max_length:
                        data_aux = np.pad(data_aux, (0, max_length - len(data_aux)))
                    elif len(data_aux) > max_length:
                        data_aux = data_aux[:max_length]
                    
                    # Model prediction
                    prediction = model.predict([np.array(data_aux)])
                    prediction_result = labels_dict[int(prediction[0])]
                    
                    # Get confidence scores if available
                    try:
                        proba = model.predict_proba([np.array(data_aux)])
                        confidence = float(proba[0][int(prediction[0])])
                    except:
                        confidence = 1.0  # Default if not available
                        
                except Exception as e:
                    print(f"Prediction error: {e}")
                    prediction_result = "Error"
        
        return jsonify({
            'prediction': prediction_result,
            'confidence': confidence,
            'hand_detected': bool(results.multi_hand_landmarks)
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

