import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 

# Created a Hand Detector Module so we can reuse code in other projects 
# basically by rewriting the main part and importing the module :>
class HandDetector():
    # constructor
    def __init__(self, mode = False, max_hands = 2, detection_confidence = 0.5, tracking_confidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    # Find hand and return the drawn hands frame
    def findHands(self,frame,draw = True):
        imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame,handlandmarks,self.mphands.HAND_CONNECTIONS)
        return frame

    # Find the position of hands and returns a list of the different 
    # points landmark for the input frame
    def findPosition(self,frame,handNo = 0,draw = True):
        lmkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                    h,w,c = frame.shape
                    cx,cy = int(lm.x * w), int(lm.y * h)
                    lmkList.append([id,cx,cy])
                    if draw:
                        if((id == 8)):
                            cv2.circle(frame,(cx,cy),15,(255,255,0),cv2.FILLED)
        return lmkList
    
    # def fingersUp(self):
    #     fingers = []
    #     # Thumb
    #     if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
    #         fingers.append(1)
    #     else:
    #         fingers.append(0)
    #     # Fingers
    #     for id in range(1, 5):
    #         if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
    #             fingers.append(1)
    #         else:
    #             fingers.append(0)
    #     return fingers


def main():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    prevTime = 0
    currTime = 0
    while(True):
        ret,frame = cap.read()
        if not ret:
            continue

        frame = detector.findHands(frame)

        lmklist = detector.findPosition(frame)
        if(len(lmklist)!=0):
            print(lmklist[8])

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime
        cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,
                    5,(45,4,210),3)

        cv2.imshow("Cam",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()