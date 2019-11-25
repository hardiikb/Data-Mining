import numpy as np
from math import sqrt
from math import pi
from math import exp

arr = []
accu = []
prec = []
reca = []
fscore = []
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
combined = np.concatenate((features,labels), axis=1)
cross_valid = np.array_split(combined, 10)

### separate the data based on labels
def separate_by_label(combined):
    label_dict = {}
    for i in range(len(combined)):
        row = combined[i]
        label = row[-1]
        if(label not in label_dict):
            label_dict[label] = []
        label_dict[label].append(row)
    return label_dict

def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean_p, stdev_p):
    exponent = exp(-((x-mean_p)**2 / (2 * stdev_p**2 )))
    return (1 / (sqrt(2 * pi) * stdev_p)) * exponent

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summary = []
    for i in range(dataset.shape[1]):
        column = dataset[:,i]
        summary.append((mean(column), stdev(column), len(column)))
    del(summary[-1])
    return summary

def summarize_by_label(dataset):
    label_dict = separate_by_label(dataset)
    summary_by_label = {}
    for label_value,rows in label_dict.items():
        summary_by_label[label_value] = summarize_dataset(np.array(rows))
    return summary_by_label

def get_values(actual,predictions):
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


# summary = summarize_by_label(combined)
# for label in summary:
# 	print(label)
# 	for row in summary[label]:
# 		print(row)
scores = []

for fold in range(len(cross_valid)):
    x = cross_valid[fold]
    y = np.vstack([d for i,d in enumerate(cross_valid) if i!=fold])
    test, train = np.asarray(x),np.asarray(y)

    summary = summarize_by_label(train)
    predictions = []
    for row in test:
        #calculate probabilities
        no_of_rows = sum([summary[label][0][2] for label in summary])
        probabilities = {}
        for label,label_summary in summary.items():
            probabilities[label] = summary[label][0][2]/float(no_of_rows)
            
            for i in range(len(label_summary)):
                mean_val, stdev_val, total_val= label_summary[i]
                probabilities[label] *= calculate_probability(row[i], mean_val, stdev_val)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        predictions.append(best_label)
    actual = [row[-1] for row in x]
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
