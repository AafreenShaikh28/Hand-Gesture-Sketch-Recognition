import tensorflow as tf   # or torch, sklearn — whatever you're using
import numpy as np

class DigitRecognition:
    def __init__(self):
        self.model = tf.keras.models.load_model("mnist_model.h5")

    def imgPreprocessing(self, imgBW):
        imgResized = cv2.resize(imgBW, (28, 28))
        return imgResized

    def predict(self, img):
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)          # batch + channel dims
        probs = self.model.predict(img)[0]
        digit = int(np.argmax(probs))
        confidence = float(probs[digit])
        return digit, confidence