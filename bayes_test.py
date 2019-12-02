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
    try:
        exponent = exp(-((x-mean_p)**2 / (2 * stdev_p**2 )))
        return (1 / (sqrt(2 * pi) * stdev_p)) * exponent
    except ZeroDivisionError:
        pass

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
try:
    test, train = test_combined, combined

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
        total = 0
        for val in probabilities.values():
            total += val
        for key in probabilities.keys():
            probabilities[key] /= total
        print(probabilities)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        predictions.append(best_label)
    actual = [row[-1] for row in test]
    # correct = 0
    # for i in range(len(actual)):
    #     if(actual[i]==predictions[i]):
    #         correct += 1
    # scores.append(correct/float(len(actual))*100.0)
    get_values(actual,predictions)
except TypeError:
    pass

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
