import cv2
import numpy as np
import handTrackMod as htm
import time
import pyautogui
from pynput.keyboard import Key, Controller

##########################
wCam, hCam = 1920, 1080
frameR = 100  # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()
click_switch = True
keyboard = Controller()

y0 = 0
y1 = 0
while True:
    #Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    #Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

    #Check which fingers are up
    fingers = detector.fingersUp()

    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),(255, 0, 255), 2)
    #Index/second Finger Moving cursor
    if (fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0):
        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        #Move Mouse
        pyautogui.moveTo(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

    #clicking w/ index finger & releasing the click w/ first 2 fingers
    if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
        pyautogui.mouseDown(button='left')
        pyautogui.mouseUp(button='left')

    # ScrollING
    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
        time0 = time.time()
        y0 = detector.lmList[1][2]

        if y0 > y1:
            pyautogui.scroll(75)
            print('up')
        if y0 < y1:
            pyautogui.scroll(-75)
            print('down')

        print(time0)
        print(y0)
        print(y1)

        y1 = detector.lmList[1][2]

    #Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    #Display
    cv2.imshow("Image", img)
    cv2.moveWindow("Image", 0, 0)
    # cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)