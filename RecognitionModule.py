import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv("digit-recognizer/train.csv")
test = pd.read_csv("digit-recognizer/test.csv")

x_train = train.drop("label", axis=1)
y_train = train["label"]

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

x_test = test.drop("label", axis=1)
y_test = test["label"]
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# class Recognition:
#     def __init__(y_img):
#         self.y_img = y_img

#     def modelPredict(self):
#         train = pd.read_csv("digit-recognizer/train.csv")
#         test = pd.read_csv("digit-recognizer/test.csv")

#         x_train = train.drop("label", axis=1)
#         y_train = train["label"]

#         model = LogisticRegression(max_iter=1000)
#         model.fit(x_train, y_train)



    
