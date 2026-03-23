import cv2

class DigitRecognition:
    # def __init__(model):
    #     self.model = model

    def imgPreprocessing(self,imgBW):
        # coords = cv2.findNonZero(imgBW)
        # if coords is not None:
        #     x, y, w, h = cv2.boundingRect(coords)
        #     imgBW = imgBW[y:y+h, x:x+w]
        imgResized = cv2.resize(imgBW, (28, 28))
        return imgResized

    # def predictor(self,img):
    #     prediction = self.model.predict(img)
    #     return prediction