import cv2
import mediapipe as mp
import time

#video input
cap = cv2.VideoCapture(0)

#creating hands visuals
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#supporting smooth fps pt0
pTime = 0
cTime = 0

while True:
    #grabs img converts to proper color values
    success, img = cap.read()
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRgb)

    #adds landmarks and hand connections to img
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
    #supports smooth fps pt1
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.imshow("Image", img)
    cv2.waitKey(1)