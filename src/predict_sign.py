import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque, Counter
import time
import pyttsx3
import threading
from wordfreq import top_n_list

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("model/model.pkl", "rb"))

# -----------------------------
# Text to Speech
# -----------------------------
def speak(text):

    if text.strip() == "":
        return

    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine

    t = threading.Thread(target=run)
    t.daemon = True
    t.start()

# -----------------------------
# Variables
# -----------------------------
prediction_buffer = deque(maxlen=15)

sentence = ""
stable_prediction = ""

last_added_time = 0
cooldown = 1.1

last_command_time = 0
command_cooldown = 2

prev_time = 0
two_hand_frames = 0
last_command = ""

english_words = top_n_list("en", 50000)

def get_suggestions(prefix):
    prefix = prefix.lower()
    matches = [w for w in english_words if w.startswith(prefix)]
    return matches[:3]

selected_suggestion = 0
suggestions_enabled = True


# -----------------------------
# MediaPipe
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

def is_fist(hand_landmarks):

    fingers_folded = 0

    tips = [8,12,16,20]
    pips = [6,10,14,18]

    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            fingers_folded += 1

    return fingers_folded >= 3

# -----------------------------
# Main Loop
# -----------------------------
while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img,1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    command = ""

    if results.multi_hand_landmarks:

        hand_count = len(results.multi_hand_landmarks)

        if hand_count == 2:
            two_hand_frames += 1
        else:
            two_hand_frames = 0

        # --------------------------------
        # TWO HAND COMMAND MODE
        # --------------------------------
        if hand_count == 2 and two_hand_frames > 5:

            hand1 = results.multi_hand_landmarks[0]
            hand2 = results.multi_hand_landmarks[1]

            x1 = hand1.landmark[0].x
            y1 = hand1.landmark[0].y

            x2 = hand2.landmark[0].x
            y2 = hand2.landmark[0].y

            x_distance = abs(x1 - x2)
            y_distance = abs(y1 - y2)

            distance = ((x1-x2)**2 + (y1-y2)**2)**0.5

            hand1_fist = is_fist(hand1)
            hand2_fist = is_fist(hand2)

            # SPACE gesture
            if distance > 0.60:
                command = "SPACE"
            
            # CLEAR gesture (hands touching)
            elif distance < 0.15:
                command = "CLEAR"

            # SPEAK gesture (hands vertical)
            elif y_distance > 0.30 and x_distance < 0.25:
                command = "SPEAK"

            # DELETE gesture (two fist)
            elif hand1_fist and hand2_fist:
                command = "DELETE"

            current_time = time.time()

            if command != "" and command != last_command and current_time - last_command_time > command_cooldown:

                if command == "SPACE":
                    sentence += " "

                elif command == "CLEAR":
                    sentence = ""

                elif command == "DELETE":
                    if len(sentence) > 0:
                        sentence = sentence[:-1]

                elif command == "SPEAK":
                    time.sleep(0.2)
                    speak(sentence)

                last_command_time = current_time
                last_command = command

            cv2.putText(img, f"COMMAND: {command}", (10,180),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks,
                                       mp_hands.HAND_CONNECTIONS)

        # --------------------------------
        # ONE HAND LETTER MODE
        # --------------------------------
        elif hand_count == 1:

            last_command = ""
            command = ""

            for hand_landmarks in results.multi_hand_landmarks:

                row = []

                for lm in hand_landmarks.landmark:
                    row.append(lm.x)
                    row.append(lm.y)
                    row.append(lm.z)

                row = np.array(row).reshape(1,-1)

                prediction = model.predict(row)[0]

                prediction_buffer.append(prediction)

                if len(prediction_buffer) == prediction_buffer.maxlen:

                    stable_prediction = Counter(prediction_buffer).most_common(1)[0][0]

                    current_time = time.time()

                    if current_time - last_added_time > cooldown:
                        if stable_prediction != "":
                            sentence += stable_prediction
                            suggestions_enabled = True
                            selected_suggestion = 0
                            last_added_time = current_time
                            prediction_buffer.clear()
                            stable_prediction = ""

                mp_draw.draw_landmarks(img, hand_landmarks,
                                       mp_hands.HAND_CONNECTIONS)

    else:
        prediction_buffer.clear()
        command = ""

    # -----------------------------
    # FPS
    # -----------------------------
    current_time = time.time()
    fps = 1/(current_time-prev_time) if prev_time!=0 else 0
    prev_time = current_time

    # -----------------------------
    # Display
    # -----------------------------
    cv2.putText(img,f"Prediction: {stable_prediction}",(10,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(img,f"Sentence: {sentence}",(10,100),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    
    # -----------------------------
    # Word Suggestions
    # -----------------------------
    current_word = sentence.split(" ")[-1]

    if suggestions_enabled and current_word != "":
        suggestions = get_suggestions(current_word)
    else:
        suggestions = []
    
    if suggestions:
        if selected_suggestion >= len(suggestions):
            selected_suggestion = 0
    else:
        selected_suggestion = -1

    suggestion_text = ""

    for i, word in enumerate(suggestions):
        word = word.upper()

        if i == selected_suggestion:
            suggestion_text += f"[{word}]  "
        else:
            suggestion_text += f"{word}  "

    if suggestions:
        cv2.putText(img,
            f"Suggestions: {suggestion_text}",
            (10,140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2)

    cv2.putText(img,f"FPS: {int(fps)}",(500,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
    
    

    cv2.imshow("Sign Prediction",img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1') and len(suggestions) > 0:
        selected_suggestion = 0
        sentence = sentence[:-len(current_word)] + suggestions[0].upper() + " "
        suggestions_enabled = False


    elif key == ord('2') and len(suggestions) > 1:
        selected_suggestion = 1
        sentence = sentence[:-len(current_word)] + suggestions[1].upper() + " "
        suggestions_enabled = False

    elif key == ord('3') and len(suggestions) > 2:
        selected_suggestion = 2
        sentence = sentence[:-len(current_word)] + suggestions[2].upper() + " "
        suggestions_enabled = False


    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()