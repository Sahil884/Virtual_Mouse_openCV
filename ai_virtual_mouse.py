import cv2
import numpy as np
import time
import HandTrackingModule as htm
import autopy
import pyautogui

pyautogui.PAUSE = 0.45
wCam, hCam = 640, 480
frameR = 100    # Frame Reduction
smoothening = 6


ptime = 0
previous_loc_x, previous_loc_y = 0, 0
current_loc_x, current_loc_y = 0, 0


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
w_screen, h_screen = autopy.screen.size()  # this will give the size of the screen of device currently using
# print(w_screen, h_screen)


while True:
    ret, frame = cap.read()
    # find the hand landmarks

    frame = detector.findHands(frame)
    lmList, bbox = detector.findPosition(frame)

    # get the tip of index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # check which fingers are up
        fingers = detector.fingers_up()
        # print(fingers)
        cv2.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # only index finger -- moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # converts coordinate

            x3 = np.interp(x1, (frameR, wCam-frameR), (0,w_screen))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, h_screen))

            # smoothen values
            current_loc_x = previous_loc_x + (x3 - previous_loc_x) / smoothening
            current_loc_y = previous_loc_y + (y3 - previous_loc_y) / smoothening

        # move mouse
            autopy.mouse.move(w_screen-current_loc_x, current_loc_y)
            cv2.circle(frame, (x1,y1), 15, (255,0,255), cv2.FILLED)
            previous_loc_x, previous_loc_y = current_loc_x, current_loc_y

        # if index and middle fingers are up -- clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # find distance between fingers
            length, frame, lineInfo = detector.findDistance(8,12, frame)
            print(length)
            # click mouse if distance is short
            if length < 24:
                cv2.circle(frame, (lineInfo[4],lineInfo[5]), 15, (0,255,0), cv2.FILLED)
                pyautogui.click()  # perform mouse click

        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1 and fingers[0] == 1:
            pyautogui.hotkey('ctrl', 'v')

        if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0 and fingers[0] == 0:

            pyautogui.hotkey('ctrl', 'c')

    fps = cap.get(cv2.CAP_PROP_FPS)

    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,185), 3)

    cv2.namedWindow('Video')

    cv2.moveWindow('Video', 40,30)
    cv2.imshow('Video', frame)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()