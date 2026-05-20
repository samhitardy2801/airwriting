import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
draw_color = (150, 0, 0)  # Dark Blue default

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    mode = "No Hand"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            lm_list = []
            for lm in handLms.landmark:
                h, w, c = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Finger detection
            index_up = lm_list[8][1] < lm_list[6][1]
            middle_up = lm_list[12][1] < lm_list[10][1]
            ring_up = lm_list[16][1] < lm_list[14][1]
            pinky_up = lm_list[20][1] < lm_list[18][1]

            fingers = [index_up, middle_up, ring_up, pinky_up]
            finger_count = fingers.count(True)

            cx, cy = lm_list[8]

            # 🧽 ERASE (4 fingers)
            if finger_count >= 4:
                mode = "ERASE"
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                prev_x, prev_y = 0, 0

            # 🎨 SELECT (2 fingers)
            elif index_up and middle_up:
                mode = "SELECT"
                prev_x, prev_y = 0, 0

                if cy < 80:
                    if 10 < cx < 110:
                        draw_color = (0, 0, 255)      # Red
                    elif 130 < cx < 230:
                        draw_color = (0, 255, 0)      # Green
                    elif 250 < cx < 350:
                        draw_color = (150, 0, 0)      # Dark Blue
                    elif 370 < cx < 470:
                        draw_color = (0, 0, 0)        # Black

            # ✍️ DRAW (1 finger)
            elif index_up:
                mode = "DRAW"

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy

                cv2.line(canvas, (prev_x, prev_y), (cx, cy), draw_color, 5)
                prev_x, prev_y = cx, cy

            # Cursor
            cv2.circle(img, (cx, cy), 10, draw_color, -1)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Merge canvas
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    # 🎨 Draw color bar AFTER merge (fix)
    cv2.rectangle(img, (0, 0), (640, 80), (50, 50, 50), -1)

    # Color boxes
    cv2.rectangle(img, (10, 10), (110, 70), (0, 0, 255), -1)    # Red
    cv2.rectangle(img, (130, 10), (230, 70), (0, 255, 0), -1)   # Green
    cv2.rectangle(img, (250, 10), (350, 70), (150, 0, 0), -1)   # Dark Blue
    cv2.rectangle(img, (370, 10), (470, 70), (0, 0, 0), -1)     # Black

    # Highlight selected color
    if draw_color == (0, 0, 255):
        cv2.rectangle(img, (10, 10), (110, 70), (255, 255, 255), 3)
    elif draw_color == (0, 255, 0):
        cv2.rectangle(img, (130, 10), (230, 70), (255, 255, 255), 3)
    elif draw_color == (150, 0, 0):
        cv2.rectangle(img, (250, 10), (350, 70), (255, 255, 255), 3)
    elif draw_color == (0, 0, 0):
        cv2.rectangle(img, (370, 10), (470, 70), (255, 255, 255), 3)

    # Mode display
    cv2.putText(img, f'Mode: {mode}', (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Air Writing AI", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()