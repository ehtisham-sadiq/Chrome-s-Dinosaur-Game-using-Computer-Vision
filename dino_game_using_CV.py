import numpy as np
import cv2

import math
import pyautogui


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # collect hand gestures
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 0)
    hand_img = frame[100:300, 100:300]

    blur = cv2.GaussianBlur(hand_img, (3, 3), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #binary image with where white will be skin colors and rest is black
    skin = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    
    kernel = np.ones((5, 5))

    dilation = cv2.dilate(skin, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(hand_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        hull = cv2.convexHull(contour)
        draw = np.zeros(hand_img.shape, np.uint8)
        cv2.drawContours(draw, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(draw, [hull], -1, (0, 0, 255), 0)

        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            if angle <= 90:
                count_defects += 1
                cv2.circle(hand_img, far, 1, [0, 0, 255], -1)

            cv2.line(hand_img, start, end, [0, 255, 0], 2)

        # if the codition matches, press space
        if count_defects >= 4:
            pyautogui.press('space')
            cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

    except:
        pass
    
    cv2.imshow("Gesture", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()