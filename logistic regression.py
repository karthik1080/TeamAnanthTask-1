
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import csv

x = []
y = []

with open("./IRIS.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        x.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        y.append(float(row[4]))

trainx, testx, trainy, testy = train_test_split(x, y, test_size = 0.25)

trainx = StandardScaler.fit_transform(trainx)
testx = StandardScaler.transform(testx)

LogisticRegression.fit(trainx, trainy)
ypred = LogisticRegression.predict(testx)

print("Accuracy score:", accuracy_score(testy, ypred))
print("Classification report:\n", classification_report(testy, ypred))
