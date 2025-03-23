from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import csv
import copy
import itertools
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
import base64

# Define the word list
WORDS = [
    "hello", "world", "this", "is", "a", "sign", "language", "typing", "test",
    "practice", "your", "asl", "skills", "and", "have", "fun", "while", "learning",
    "the", "alphabet", "and", "improving", "your", "speed", "and", "accuracy"
]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Initialize game state
game_state = {
    "current_word": "",
    "current_letter_index": 0,
    "score": 0,
    "game_active": False,
    "time_remaining": 60,  # Default duration
    "words": [],
    "current_word_index": 0,
    "completed_words": [],
    "last_detected_letter": None,
    "start_time": None
}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
classifier = KeyPointClassifier()

# Load labels
with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
    labels = [row[0] for row in csv.reader(f)]

def calc_landmark_list(image_width, image_height, landmarks):
    landmark_list = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append([landmark_x, landmark_y])
    return landmark_list

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    
    def normalize_(n):
        return n / max_value
    
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    return temp_landmark_list

@app.route('/')
def index():
    return render_template('web_game.html')

@socketio.on('start_game')
def handle_start_game(data):
    global game_state
    duration = data.get('duration', 60)  # Get duration from client, default to 60 seconds
    game_state = {
        "current_word": "",
        "current_letter_index": 0,
        "score": 0,
        "game_active": True,
        "time_remaining": duration,  # Use the selected duration
        "words": random.sample(WORDS, min(10, len(WORDS))),  # Select 10 random words
        "current_word_index": 0,
        "completed_words": [],
        "last_detected_letter": None,
        "start_time": time.time()
    }
    if game_state["words"]:
        game_state["current_word"] = game_state["words"][0]
    emit('game_state', {"game_state": game_state})

@socketio.on('process_frame')
def handle_frame(data):
    if not game_state['game_active']:
        return

    # Decode base64 image
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Get image dimensions
    image_height, image_width = frame.shape[:2]
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(rgb_frame)
    
    # Prepare response data
    response_data = {
        'game_state': game_state,
        'landmarks': None
    }
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate landmarks
            landmark_list = calc_landmark_list(image_width, image_height, hand_landmarks)
            
            # Pre-process landmarks
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            try:
                # Classify the gesture
                gesture_id = classifier(pre_processed_landmark_list)
                detected_letter = labels[gesture_id].lower()
                game_state['last_detected_letter'] = detected_letter
                
                # Handle the gesture
                if detected_letter == game_state['current_word'][game_state['current_letter_index']]:
                    game_state['current_letter_index'] += 1
                    if game_state['current_letter_index'] >= len(game_state['current_word']):
                        # Word completed
                        game_state['completed_words'].append(game_state['current_word'])
                        game_state['current_word_index'] += 1
                        
                        # Check if we've completed all words
                        if game_state['current_word_index'] >= len(game_state['words']):
                            game_state['game_active'] = False
                        else:
                            # Move to next word
                            game_state['current_word'] = game_state['words'][game_state['current_word_index']]
                            game_state['current_letter_index'] = 0
                
                # Add landmarks to response
                response_data['landmarks'] = landmark_list
                
            except Exception as e:
                print(f"Error classifying gesture: {e}")
    
    # Update game state in response
    response_data['game_state'] = game_state
    emit('game_state', response_data)

@socketio.on('update_timer')
def handle_timer_update():
    if game_state['game_active']:
        game_state['time_remaining'] -= 1
        if game_state['time_remaining'] <= 0:
            game_state['game_active'] = False
        emit('game_state', game_state)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001) 