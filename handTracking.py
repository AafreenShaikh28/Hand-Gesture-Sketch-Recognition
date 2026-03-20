import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 

#------------------------------------------------------------------
# Track hand Motions then register strokes drawn 
#------------------------------------------------------------------

# creates a video capture object (0-> default webcam)
cap = cv2.VideoCapture(0)

prevTime = 0
currTime = 0

mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while(True):
# reads frames from the camera
    ret,frame = cap.read()
    if not ret:
        continue
    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    # 21 landmark points.
    # if results frame has a hand, multi_hand_landmarks does give coordinates
    if results.multi_hand_landmarks:
        for handlandmarks in results.multi_hand_landmarks:
        # handlandmarks.landmarks gives id and x,y,z coordinates ratio
            for id,lm in enumerate(handlandmarks.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                if((id == 8)):
                    cv2.circle(frame,(cx,cy),15,(255,255,0),cv2.FILLED)
            mpDraw.draw_landmarks(frame,handlandmarks,
            mphands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime
    cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,
                3,(255,255,0),3)
# displays the framees in a window called "Cam"
    cv2.imshow("Cam",frame)

# waits 1 milliesec and checks for q key
# close cam on q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# frees camera and closes al cv windows
cap.release()
cv2.destroyAllWindows()

