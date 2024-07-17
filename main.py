import itertools
from collections import Counter
from sklearn.model_selection import train_test_split

k = 3

# Read the vectors from vectors.txt
with open('vectors.txt', 'r') as file:
    data = [[int(x) for x in line.split()] for line in file]

# Split data into features (vectors) and labels (last element of each vector)
vectors = [vector[:-1] for vector in data]
labels = [vector[-1] for vector in data]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=42)


# Function to calculate error rate
def error_rate(predicted_labels, true_labels):
    return sum(1 for pred, true in zip(predicted_labels, true_labels) if pred != true) / len(true_labels)


# Function to generate specific binary trees of depth k
def generate_trees(k):
    trees = []
    for tree in itertools.product([0, 1], repeat=k):
        trees.append(tree)
    return trees


# Function to predict labels based on a given tree
def predict_labels(tree, X):
    predictions = []
    for vector in X:
        node = 0
        depth = 0
        while depth < len(tree):
            if tree[node] == 0:
                predictions.append(0)
                break
            elif tree[node] == 1:
                predictions.append(1)
                break
            else:
                node = 2 * node + 1 + vector[tree[node] - 2]
                depth += 1
    return predictions


# Print all specific tree structures and their error rates on the training set
print("All generated trees and their training error rates:")
for tree in generate_trees(k):
    predictions = predict_labels(tree, X_train)
    current_error = error_rate(predictions, y_train)
    print(f"Tree: {tree}, Training error rate: {current_error:.4f}")

# Find the best tree within the desired structure on the training set
best_tree = None
min_error = float('inf')

for tree in generate_trees(k):
    predictions = predict_labels(tree, X_train)
    current_error = error_rate(predictions, y_train)
    if current_error < min_error:
        min_error = current_error
        best_tree = tree

# Evaluate the best tree on the test set
test_predictions = predict_labels(best_tree, X_test)
test_error = error_rate(test_predictions, y_test)

# Print results
print("\nResults:")
print(f"Best tree: {best_tree}")
print(f"Training error rate: {min_error:.4f}")
print(f"Test error rate: {test_error:.4f}")