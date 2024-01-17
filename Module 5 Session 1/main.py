#importing necessary libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression

data = pd.read_csv("seattle_weather_1948-2017.csv")
data.head(5)

data.shape

data.describe()

data.isnull().any()

features = ["DATE","PRCP","TMAX","TMIN","RAIN"]

X = data[features]
y = data.iloc[:, 5]

X.head(5)

y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 0)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

naiveBayes = GaussianNB()
naiveBayes.fit(X_train, y_train)

y_preds = naiveBayes.predict(X_test)
y_preds

print("Training Score (accuracy): ", accuracy_score(y_train, naiveBayes.predict(X_train)))
print("Testing Score (accuracy): ", accuracy_score(y_test, y_preds))

confusion = confusion_matrix(y_test, y_preds, labels = naiveBayes.classes_)
display = ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels = naiveBayes.classes_)

display.plot()