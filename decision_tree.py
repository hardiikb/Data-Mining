# Created by Varun at 08/11/19
import numpy as np
import pandas as pd


class TreeNode():
    def __init__(self):
        self.split_index = None
        self.split_value = None
        self.left = []
        self.right = []
        self.attribute_column = None


class PerformanceMetrics():
    def __init__(self):
        self.accu = []
        self.prec = []
        self.reca = []
        self.fscore = []
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0

    def get_values(self, test_results, ground_truth):
        for i, j in zip(ground_truth, test_results):
            if i == 1 and j == 1:
                self.a += 1
            elif i == 1 and j == 0:
                self.b += 1
            elif i == 0 and j == 1:
                self.c += 1
            elif i == 0 and j == 0:
                self.d += 1
        self.accuracy()
        self.precision()
        self.recall()
        self.f_score()
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0

    def accuracy(self):
        self.accu.append(float(self.a + self.d)/(self.a + self.b + self.c + self.d))

    def f_score(self):
        self.fscore.append(float(2 * self.a) / ((2 * self.a) + self.b + self.c))

    def precision(self):
        self.prec.append(float(self.a)/(self.a + self.c))

    def recall(self):
        self.reca.append(float(self.a)/(self.a + self.b))


def get_tree_node(data):
    current_min_error = float('inf')
    node = TreeNode()
    for col in range(len(data[0])-1):
        for r in range(len(data)):
            left = []
            right = []
            split_value = data[r][col]
            for row in range(len(data)):
                if col not in string_indices:
                    if data[row][col] <= split_value:
                        left.append(data[row])
                    else:
                        right.append(data[row])
                else:
                    if data[row][col] == split_value:
                        left.append(data[row])
                    else:
                        right.append(data[row])
            left, right = np.asarray(left), np.asarray(right)
            total_tree_nodes = len(left) + len(right)
            left0, left1, right0, right1 = 0, 0, 0, 0
            if len(left) > 0:
                left1 = count_method(left, 1)
                left0 = count_method(left, 0)
            if len(right) > 0:
                right1 = count_method(right, 1)
                right0 = count_method(right, 0)
            gini_index_left = 1.0 - ((left1 * left1) + (left0 * left0))
            gini_index_right = 1.0 - ((right1 * right1) + (right0 * right0))
            gini_index = ((gini_index_left * len(left)) + (gini_index_right * len(right))) / total_tree_nodes

            if current_min_error > gini_index:
                current_min_error = gini_index
                node.split_value = data[r][col]
                node.split_index = r
                node.attribute_column = col
                node.left = left
                node.right = right
    return node


# noinspection PyTypeChecker
def generate_decision_tree(root, depth):
    left_tree = root.left
    right_tree = root.right

    def count_0_1(count0, count1, left):
        count0 += list(left[:,-1]).count(0)
        count1 += list(left[:,-1]).count(1)
        return count0, count1

    def node_class(left,right):
        count0 = 0
        count1 = 0
        if len(left):
            count0, count1 = count_0_1(count0, count1, left)
        if len(right):
            count0, count1 = count_0_1(count0, count1, right)
        return 1 if count1 > count0 else 0

    def check_tree(node, side, string_side):
        if len(side) > 0:
            flag = False
            if len(np.unique(side[:,-1])) == 1:
                flag = True
            if string_side == "left":
                if flag:
                    node.left = node_class(side, list())
                else:
                    node.left = generate_decision_tree(get_tree_node(side), depth + 1)
            else:
                if flag:
                    node.right = node_class(side, list())
                else:
                    node.right = generate_decision_tree(get_tree_node(side), depth + 1)

    if len(left_tree) == 0 or len(right_tree) == 0:
        root.left = root.right = node_class(left_tree, right_tree)
        return root
    check_tree(root, left_tree, 'left')
    check_tree(root, right_tree, 'right')
    return root


def count_method(side, element):
    return float(list(side[:,-1]).count(element)) / len(side)


def handle_categorical_attributes():
    string_indices = []
    for i in range(len(df[0])):
        flag = False
        try:
            complex(df[0][i])  # for int, long, float and complex
        except ValueError:
            flag = True
        if flag:
            string_indices.append(i)
    for i in string_indices:
        unique_strings = np.unique(df[:, i])
        replacement_vals = list(range(len(unique_strings)))
        dictionary = dict(zip(unique_strings, replacement_vals))
        for j in range(len(df[:, i])):
            df[j][i] = dictionary.get(df[j][i])

    return string_indices


def predict(root, test_row):
    def check_tree_condition(node, test_data_point, string_tree_side):
        if string_tree_side == "left":
            if node.left == 0 or node.left == 1:
                return node.left
            else:
                return predict(node.left, test_data_point)
        elif string_tree_side == "right":
            if node.right == 0 or node.right == 1:
                return node.right
            else:
                return predict(node.right, test_data_point)

    if root == 0 or root == 1:
        return root
    elif root.attribute_column not in string_indices:
        if test_row[root.attribute_column] < root.split_value:
            return check_tree_condition(root, test_row, 'left')
        else:
            return check_tree_condition(root, test_row, 'right')
    else:
        if test_row[root.attribute_column] == root.split_value:
            return check_tree_condition(root, test_row, 'left')
        else:
            return check_tree_condition(root, test_row, 'right')


filename = input("Enter the dataset file name")
df = [line.strip().split('\t') for line in open(filename, 'r')]
df = np.asarray(df)
output_class = df[:,-1].reshape((df.shape[0],1)).astype(int)
df = df[:,0:-1]
string_indices = handle_categorical_attributes()
df = df.astype(float)
df = np.append(df,output_class,axis=1)
number_of_folds = int(input("Enter the number of k-folds for CV"))
cross_valid = np.array_split(df, number_of_folds)
metrics = PerformanceMetrics()
for fold in range(len(cross_valid)):
    x = cross_valid[fold]
    y = np.vstack([d for i,d in enumerate(cross_valid) if i != fold])
    test, train = np.asarray(x),np.asarray(y)
    predicted_classes = []
    root = get_tree_node(data=train)
    root = generate_decision_tree(root, 1)
    for test_row in test:
        predicted_classes.append(predict(root, test_row))
    metrics.get_values(predicted_classes, test[:,-1])
print(metrics.accu, metrics.fscore)
print(np.mean(metrics.accu))
print(np.mean(metrics.prec))
print(np.mean(metrics.reca))
print(np.mean(metrics.fscore))


