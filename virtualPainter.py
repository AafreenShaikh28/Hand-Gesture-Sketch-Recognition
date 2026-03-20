import cv2
import numpy as np 
import time
import os
import handTrackingModule as htm 

folder_path = "virtualpainter"
myList = os.listdir(folder_path)
myList.pop(0)
img_overlay =[]
for impath in myList:
    image = cv2.imread(f'{folder_path}/{impath}')
    img_overlay.append(image)
print(len(img_overlay))

default_header = img_overlay[2]
draw_color = (230, 216, 173)
brush_thickness = 15
index_x_p,index_y_p = 0,0

imgCanvas = np.zeros((720,1280,3),np.uint8)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = htm.HandDetector(detection_confidence=0.85)

while(True):
    # 1) reading frames
    ret,frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    # 2) Finding hand landmarks
    frame = detector.findHands(frame)
    lmkList = detector.findPosition(frame,0,False)
    if(len(lmkList)!=0):
        index_x,index_y = lmkList[8][1:]
        middle_x,middle_y = lmkList[12][1:]
    # 3) Check which fingers are up

        # BOTH FINGERS ARE UP (SELECTION)
        if((lmkList[8][2] < lmkList[5][2]) and (lmkList[12][2] < lmkList[9][2]) ):
            cv2.putText(frame,"Both Fingers are up!",(500,70),cv2.FONT_HERSHEY_PLAIN,
                3,draw_color,3)
            # BLUE
            if(0<index_x <200 and 0<index_y < 180): 
                default_header = img_overlay[2]
                draw_color = (230, 216, 173)
                brush_thickness = 20
            # PINK
            elif(0<index_x < 200 and 180<index_y < 360):
                default_header = img_overlay[1]
                #255, 182, 193
                draw_color = (193, 182, 255)
                brush_thickness = 20
            # YELLOW
            elif(0<index_x < 200 and 360 <index_y < 540):
                default_header = img_overlay[0]
                draw_color = (224, 255, 255)
                brush_thickness = 20
            # ERASER
            elif(0<index_x < 200 and 540<index_y < 720):
                default_header = img_overlay[3]
                draw_color = (0, 0, 0)

            cv2.rectangle(frame, (index_x, index_y), (index_x+30, index_y+30), draw_color, cv2.FILLED)

        # INDEX FINGER (DRAWING)
        if((lmkList[8][2] < lmkList[5][2]) and (lmkList[12][2] > lmkList[9][2]) ):
            cv2.putText(frame,"Index Finger is up!",(500,70),cv2.FONT_HERSHEY_PLAIN,
                3,(0,0,0),3)
            cv2.circle(frame, (index_x, index_y), 15, draw_color, cv2.FILLED)
            if(index_x_p == 0 and index_y_p == 0):
                index_x_p = index_x
                index_y_p = index_y
            if(draw_color==(0,0,0)):
                cv2.line(frame,(index_x_p,index_y_p),(index_x,index_y),draw_color,100)
                cv2.line(imgCanvas,(index_x_p,index_y_p),(index_x,index_y),draw_color,100)
            else:
                # to reduce jitter
                dist = ((index_x - index_x_p)**2 + (index_y - index_y_p)**2) ** 0.5
                if dist > 5: 
                    cv2.line(frame,(index_x_p,index_y_p),(index_x,index_y),draw_color,brush_thickness)
                    cv2.line(imgCanvas,(index_x_p,index_y_p),(index_x,index_y),draw_color,brush_thickness)
            index_x_p  = index_x
            index_y_p = index_y
        else:
            index_x_p, index_y_p = 0, 0

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame,imgInv)
    frame = cv2.bitwise_or(frame,imgCanvas)
    # 4) check selection
    # 5) drawing mode? 

    # setting palatte 
    frame[0:720,0:200] = default_header
    # frame = cv2.addWeighted(frame,0.5,imgCanvas,0.5,0)
    cv2.imshow("Camera",frame)
    # cv2.imshow("Canvas",imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    # we can have the user change size of the brush with distance b/w two fingers