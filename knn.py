import numpy as np
from math import sqrt
arr = []
reader = open("project3_dataset2.txt")
iterationCounter = 0
try:
    for line in reader:
        line = line.strip()
        line_arr = line.split("\t")
        arr.append(line_arr)
finally:
    reader.close()

input_data = np.array(arr)
rows = input_data.shape[0]
columns = input_data.shape[1]

def handle_categorical_attributes():
    string_indices = []
    for i in range(len(input_data[0])):
        flag = False
        try:
            complex(input_data[0][i])  # for int, long, float and complex
        except ValueError:
            flag = True
        if flag:
            string_indices.append(i)
    for i in string_indices:
        unique_strings = np.unique(input_data[:, i])
        replacement_vals = list(range(len(unique_strings)))
        dictionary = dict(zip(unique_strings, replacement_vals))
        for j in range(len(input_data[:, i])):
            input_data[j][i] = dictionary.get(input_data[j][i])
    return string_indices

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

handle_categorical_attributes()

features = input_data[:,0:columns-1]
labels = input_data[:,columns-1:columns]

## separating features and labels 
features = features.astype(np.float)
labels = labels.astype(np.int)

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

### normalizing the features 
normalize(features)

## combining features and labels to perform cross validation
combined = np.concatenate((features,labels), axis=1)
cross_valid = np.array_split(combined, 10)
scores = []

for fold in range(len(cross_valid)):
    x = cross_valid[fold]
    y = np.vstack([d for i,d in enumerate(cross_valid) if i!=fold])
    test, train = np.asarray(x),np.asarray(y)
    
    predictions = []
    for row in test:
        neighbors_list = neighbors(train, row, 10)
        output_values = [row[-1] for row in neighbors_list]
        prediction = max(set(output_values), key=output_values.count)
        predictions.append(prediction)

    actual = [row[-1] for row in x]
    correct = 0
    for i in range(len(actual)):
        if(actual[i]==predictions[i]):
            correct += 1
    scores.append(correct/float(len(actual))*100.0)

print(scores)

