import keyboard
import cv2
import mediapipe as mp
import math as m
import time
import pyautogui
import numpy as np
from sklearn.svm import SVC
import joblib
def bruh(landmarks):
    return [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
def extract_features(landmark):
    middle = landmark[9]
    wrist  = landmark[0]

    size = m.hypot(
        middle.x - wrist.x,
        middle.y - wrist.y
    )
    f = []
    for lm in landmark:
        x = (lm.x - wrist.x) / size
        y = (lm.y - wrist.y) / size
        z = (lm.z - wrist.z) / size
        f.extend([x, y, z])
    return f
def finger_up(hand_landmark, tip_id, pip_id):
    tip = hand_landmarks.hand_landmarks[tip_id].y
    pip = hand_landmarks.hand_landmarks[pip_id].y
    return tip > pip

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)
w = cap.get(3)
h = cap.get(4)
data = []
lable = []
can_press = True
clf = joblib.load("clf.pkl")
hand_pos = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
            
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            fingertips = {
                "thumb": hand_landmarks.landmark[4],
                "index": hand_landmarks.landmark[8]
            }
            
            # x0,y0 = fingertips["thumb"].x, fingertips["thumb"].y 
            # x1,y1 = fingertips["index"].x, fingertips["index"].y
            # dist = m.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))

            # if dist < 0.05 and can_press:
            #     if can_press:
            #         print("pause")
            #         pyautogui.press("space")
            #         can_press = False
            # elif dist >= 0.05:
            #     can_press = True
            x_pos = int(fingertips["index"].x * w)
            y_pos = int(fingertips["index"].y * h - 10)
            pred = str(clf.predict([extract_features(hand_landmarks.landmark)])[0])
            cv2.putText( frame,pred, (x_pos, y_pos),cv2.FONT_HERSHEY_COMPLEX,1,(255, 255, 255), 2)
            hand_pos.append(fingertips["index"])
            if len(hand_pos) > 2:
                hand_pos.pop(0)
            if len(hand_pos) > 1:
                dx = hand_pos[1].x - hand_pos[0].x 
                dy = hand_pos[1].y - hand_pos[0].y
                thre = 0.02
                if abs(dx) > abs(dy):
                    if dx > thre:
                        print("swipe right")
                    elif dx < -thre:
                        print("swipe left")
                else:
                    if dy > thre:
                        print("swipe down")
                    elif dy < -thre:
                        print("swipe up")
                
                # if pred == "open_palm":

            if pred == "pinch":
                if can_press:
                    print("pause")
                    pyautogui.press("space")
                    can_press = False
            else:
                can_press = True   
            if keyboard.is_pressed("q"):
                data.append(extract_features(hand_landmarks.landmark))
                lable.append("fist")
            if keyboard.is_pressed("a"):
                data.append(extract_features(hand_landmarks.landmark))
                lable.append("open_palm")
            if keyboard.is_pressed("w"):
                data.append(extract_features(hand_landmarks.landmark))
                lable.append("pinch")
            if keyboard.is_pressed("space"):
                print(clf.predict([extract_features(hand_landmarks.landmark)])[0])
            
                

    # for i in data:
    #     print(i)
    if keyboard.is_pressed("escape"):
        break

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
x = np.array(data)
y = np.array(lable)
np.save("data.npy", x)
np.save("labels.npy", y)
cap.release()
cv2.destroyAllWindows()