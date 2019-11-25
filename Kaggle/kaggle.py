from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
import numpy as np
from sklearn import metrics
from random import randrange, uniform, randint
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

arr = []
arr2 = []
arr3 = []
train = open("train_features.csv")
label = open("train_label.csv")
test = open("test_features.csv")

submission = open("sample_Submission.csv")
df = pd.read_csv(submission)
df.index = df.index + 1


try:
    for line in train:
        line = line.strip()
        line_arr = line.split(",")
        arr.append(line_arr)
finally:
    train.close()

try:
    for line in label:
        line = line.strip()
        line_arr = line.split(",")
        arr2.append(line_arr)
finally:
    label.close()

try:
    for line in test:
        line = line.strip()
        line_arr = line.split(",")
        arr3.append(line_arr)
finally:
    test.close()

arr2.pop(0)

def normalize(features):
    minmax = []
    for i in range(features.shape[1]):
        col = features[:,i]
        min_value = np.min(col)
        max_value = np.max(col)
        minmax.append([min_value,max_value])
    for row in features:
        for i in range(len(row)):
            row[i] = (row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])
    return features

input_data = np.array(arr).astype(np.float)
input_data = input_data[:,1:]
print(input_data.shape)
labels = np.array(arr2).astype(np.int)
labels = labels[:,1:]

labels = labels.reshape(418,)
test_data = np.array(arr3).astype(np.float)
test_data = test_data[:,1:]

#clf = AdaBoostClassifier(n_estimators=750, random_state=0, base_estimator=RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)) #0.86505
clf = AdaBoostClassifier(n_estimators=750, random_state=0, base_estimator=RandomForestClassifier(n_estimators=95, max_depth=2,random_state=0)) 

model = clf.fit(input_data,labels)
test_pred = model.predict(test_data)
df["label"] = test_pred

df.to_csv("submission.csv", index=False)