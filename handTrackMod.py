import cv2
import mediapipe as mp
import time

class handDetec():
    def __init__(self, mode=False, maxHands=2, detecCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detecCon = detecCon
        self.trackCon = trackCon

        #creating hands visuals
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detecCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def finsHands(self, img, draw=True):


        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:

                    # adds landmarks and hand connections to img
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNum=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
           myHand = self.results.multi_hand_landmarks[handNum]

            # converts from the decimal prints the id and position
            for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmList



def main():
    # supporting smooth fps pt0
    pTime = 0
    cTime = 0
    # video input
    cap = cv2.VideoCapture(0)
    detec = handDetec

    while True:
        # grabs img converts to proper color values
        success, img = cap.read()
        img = detec.findHands(img)
        lmList = detec.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # supports smooth fps pt1
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()