import numpy as np
from itertools import combinations


# Read data from file
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            data.append(list(map(int, line.strip().split())))
    return np.array(data)


# Decision tree node class
class TreeNode:
    def __init__(self, depth=0):
        self.left = None
        self.right = None
        self.feature = None
        self.label = None
        self.depth = depth

    def is_leaf(self):
        return self.label is not None

    def predict(self, x):
        if self.is_leaf():
            return self.label
        if x[self.feature] == 0:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


# Function to compute the error
def compute_error(node, X, y):
    if node.is_leaf():
        predictions = np.full(y.shape, node.label)
        return np.sum(predictions != y)
    left_mask = X[:, node.feature] == 0
    right_mask = ~left_mask
    left_error = compute_error(node.left, X[left_mask], y[left_mask])
    right_error = compute_error(node.right, X[right_mask], y[right_mask])
    return left_error + right_error


# Function to create a tree given a combination of features
def create_tree(X, y, features, depth=0, max_depth=3, parent_label=None):
    # Check if y is empty
    if len(y) == 0:
        # Return a leaf with the parent node's majority label
        leaf = TreeNode(depth)
        leaf.label = parent_label
        return leaf

    if depth == max_depth or len(np.unique(y)) == 1:
        leaf = TreeNode(depth)
        leaf.label = np.argmax(np.bincount(y))
        return leaf

    best_tree = None
    best_error = float('inf')

    for feature in features:
        node = TreeNode(depth)
        node.feature = feature

        left_mask = X[:, feature] == 0
        right_mask = ~left_mask

        left_parent_label = np.argmax(np.bincount(y[left_mask])) if len(y[left_mask]) > 0 else parent_label
        right_parent_label = np.argmax(np.bincount(y[right_mask])) if len(y[right_mask]) > 0 else parent_label

        node.left = create_tree(X[left_mask], y[left_mask], features, depth + 1, max_depth, left_parent_label)
        node.right = create_tree(X[right_mask], y[right_mask], features, depth + 1, max_depth, right_parent_label)

        error = compute_error(node, X, y)
        if error < best_error:
            best_tree = node
            best_error = error

    return best_tree


# Generate all possible trees with k=3 levels
def brute_force_decision_tree(X, y, max_depth=3):
    n_features = X.shape[1]
    all_features = range(n_features)

    best_tree = None
    best_error = float('inf')

    for feature_comb in combinations(all_features, max_depth):
        tree = create_tree(X, y, feature_comb, max_depth=max_depth)
        error = compute_error(tree, X, y)
        print(f"error {error}")
        if error < best_error:
            best_tree = tree
            best_error = error

    return best_tree, best_error


# Print the tree
def print_tree(node, depth=0):
    indent = "  " * depth
    if node.is_leaf():
        print(f"{indent}Label: {node.label}")
    else:
        print(f"{indent}Feature: {node.feature}")
        print(f"{indent}Left:")
        print_tree(node.left, depth + 1)
        print(f"{indent}Right:")
        print_tree(node.right, depth + 1)


# File path
file_path = 'vectors.txt'
data = read_data(file_path)

# Separate features and labels
X = data[:, :-1]
y = data[:, -1]

# Run the algorithm on the vector data-set with k=3
tree, error = brute_force_decision_tree(X, y, max_depth=3)
print(f'Best tree error: {error}')

# Print the tree
print_tree(tree)
