import cv2
import os
import tensorflow as tf   # or torch, sklearn — whatever you're using
import numpy as np

class DigitRecognition:
    def __init__(self, model_path="mnist_model.keras"):
        # prefer the .keras file saved by training; fall back to .h5 if necessary
        model_file = model_path
        if not os.path.exists(model_file):
            if os.path.exists("mnist_model.h5"):
                model_file = "mnist_model.h5"
        self.model = tf.keras.models.load_model(model_file)

    def imgPreprocessing(self, imgBW):
        # ensure single-channel and resize to 28x28 suitable for the model
        imgResized = cv2.resize(imgBW, (28, 28), interpolation=cv2.INTER_AREA)
        if len(imgResized.shape) == 3:
            imgResized = cv2.cvtColor(imgResized, cv2.COLOR_BGR2GRAY)
        imgResized = imgResized.astype('uint8')
        return imgResized

    def predict(self, img):
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)          # batch + channel dims
        probs = self.model.predict(img)[0]
        digit = int(np.argmax(probs))
        confidence = float(probs[digit])
        return digit, confidence