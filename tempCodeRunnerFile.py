x_test = test.drop("label", axis=1)
y_test = test["label"]
y_pred = model.predict(x_test)