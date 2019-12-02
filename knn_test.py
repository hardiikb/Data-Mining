import numpy as np
from math import sqrt
from math import pi
from math import exp

arr = []
arr1 = []
accu = []
prec = []
reca = []
fscore = []
reader = open("project3_dataset3_train.txt")
reader1 = open("project3_dataset3_test.txt")

try:
    for line in reader:
        line = line.strip()
        line_arr = line.split("\t")
        arr.append(line_arr)
finally:
    reader.close()

try:
    for line in reader1:
        line = line.strip()
        line_arr = line.split("\t")
        arr1.append(line_arr)
finally:
    reader1.close()

input_data = np.array(arr)
rows = input_data.shape[0]
columns = input_data.shape[1]

test_data = np.array(arr1)
test_rows = test_data.shape[0]
test_columns = test_data.shape[1]


def handle_categorical_attributes(params):
    string_indices = []
    for i in range(len(params[0])):
        flag = False
        try:
            complex(params[0][i])  # for int, long, float and complex
        except ValueError:
            flag = True
        if flag:
            string_indices.append(i)
    for i in string_indices:
        unique_strings = np.unique(params[:, i])
        replacement_vals = list(range(len(unique_strings)))
        dictionary = dict(zip(unique_strings, replacement_vals))
        for j in range(len(params[:, i])):
            params[j][i] = dictionary.get(params[j][i])
    return params

def euclidean(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

def neighbors(train, test, num_neighbors):
    distances = []
    for row in train:
        dist = euclidean(test, row)
        distances.append((row,dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

input_data = handle_categorical_attributes(input_data)
test_data = handle_categorical_attributes(test_data)

features = input_data[:,0:columns-1]
labels = input_data[:,columns-1:columns]
test_features = test_data[:, 0:test_columns-1]
test_labels = test_data[:,test_columns-1:test_columns]

## separating features and labels 
features = features.astype(np.float)
labels = labels.astype(np.int)
test_features = test_features.astype(np.float)
test_labels = test_labels.astype(np.int)

combined = np.concatenate((features,labels), axis=1)
test_combined = np.concatenate((test_features,test_labels), axis=1)

def get_values(actual,predictions):
    try:
        a = 0
        b = 0
        c = 0
        d = 0
        for i in range(len(actual)):
            if(actual[i]==1 and predictions[i]==1):
                a += 1
            elif(actual[i]==1 and predictions[i]==0):
                b += 1
            elif(actual[i]==0 and predictions[i]==1):
                c += 1
            else:
                d += 1

        accu.append(float(a + d)/(a + b + c + d))
        fscore.append(float(2 * a) / ((2 * a) + b + c))
        prec.append(float(a)/(a + c))
        reca.append(float(a)/(a + b))
    except ZeroDivisionError:
        pass

scores = []

test, train = test_combined,combined

predictions = []
for row in test:
    neighbors_list = neighbors(train, row, 10)
    output_values = [row[-1] for row in neighbors_list]
    prediction = max(set(output_values), key=output_values.count)
    predictions.append(prediction)

actual = [row[-1] for row in test]
# correct = 0
# for i in range(len(actual)):
#     if(actual[i]==predictions[i]):
#         correct += 1
# scores.append(correct/float(len(actual))*100.0)
get_values(actual,predictions)

#print(scores)
print("**** 10 Fold Accuracy ****")
print(accu)
print("Mean Accuracy: " + str(np.mean(accu)))
print("")

print("**** 10 Fold Precision ****")
print(prec)
print("Mean Precision: " + str(np.mean(prec)))
print("")

print("**** 10 Fold Recall ****")
print(reca)
print("Mean Recall: " + str(np.mean(reca)))
print("")

print("**** 10 Fold Fscore ****")
print(fscore)
print("Mean Fscore: " + str(np.mean(fscore)))
