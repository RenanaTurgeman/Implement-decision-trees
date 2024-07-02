import numpy as np
from scipy.stats import entropy

# Load the data
file_path = 'vectors.txt'
data = np.loadtxt(file_path, dtype=int)

# Separate features and labels
X = data[:, :-1]
y = data[:, -1]


# Function to calculate error for a given split
def calculate_error(y_left, y_right):
    n_left = len(y_left)
    n_right = len(y_right)
    if n_left == 0 or n_right == 0:
        return float('inf')
    p_left = np.sum(y_left) / n_left
    p_right = np.sum(y_right) / n_right
    error_left = np.minimum(p_left, 1 - p_left) * n_left
    error_right = np.minimum(p_right, 1 - p_right) * n_right
    return error_left + error_right


# A. Brute-force Method
def brute_force_decision_tree(X, y, k):
    n_samples, n_features = X.shape
    min_error = float('inf')
    best_tree = None

    # Recursive function to generate all possible trees
    def generate_trees(depth, X, y):
        nonlocal min_error, best_tree

        if depth == k or len(np.unique(y)) == 1:
            return [{'is_leaf': True, 'prediction': np.argmax(np.bincount(y))}]

        trees = []
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                y_left = y[left_indices]
                y_right = y[right_indices]
                error = calculate_error(y_left, y_right)
                if error < min_error:
                    min_error = error
                    left_tree = generate_trees(depth + 1, X[left_indices], y_left)
                    right_tree = generate_trees(depth + 1, X[right_indices], y_right)
                    trees.append({
                        'is_leaf': False,
                        'feature': feature,
                        'threshold': threshold,
                        'left': left_tree,
                        'right': right_tree
                    })
        return trees

    best_tree = generate_trees(0, X, y)
    return best_tree, min_error


# B. Binary Entropy Method (ID3 Algorithm)
def binary_entropy_decision_tree(X, y, k):
    n_samples, n_features = X.shape

    # Function to calculate binary entropy
    def calculate_entropy(y):
        p = np.sum(y) / len(y)
        return entropy([p, 1 - p], base=2)

    # Function to calculate the best split
    def best_split(X, y):
        min_entropy = float('inf')
        best_feature = None
        best_threshold = None
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                y_left = y[left_indices]
                y_right = y[right_indices]
                entropy_left = calculate_entropy(y_left)
                entropy_right = calculate_entropy(y_right)
                weighted_entropy = (len(y_left) / len(y)) * entropy_left + (len(y_right) / len(y)) * entropy_right
                if weighted_entropy < min_entropy:
                    min_entropy = weighted_entropy
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    # Recursive function to build the tree
    def build_tree(depth, X, y):
        if depth == k or len(np.unique(y)) == 1:
            return {'is_leaf': True, 'prediction': np.argmax(np.bincount(y))}

        feature, threshold = best_split(X, y)
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        left_tree = build_tree(depth + 1, X[left_indices], y[left_indices])
        right_tree = build_tree(depth + 1, X[right_indices], y[right_indices])
        return {
            'is_leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }

    tree = build_tree(0, X, y)
    return tree


# Function to evaluate the tree
def evaluate_tree(tree, X, y):
    def predict(tree, x):
        if tree['is_leaf']:
            return tree['prediction']
        if x[tree['feature']] <= tree['threshold']:
            return predict(tree['left'], x)
        else:
            return predict(tree['right'], x)

    predictions = [predict(tree, x) for x in X]
    error = np.mean(predictions != y)
    return error


# Function to visualize the tree
def print_tree(tree, depth=0):
    if tree['is_leaf']:
        print(f'{"|   " * depth}Leaf: Predict {tree["prediction"]}')
    else:
        print(f'{"|   " * depth}Node: Feature {tree["feature"]}, Threshold {tree["threshold"]}')
        print_tree(tree['left'], depth + 1)
        print_tree(tree['right'], depth + 1)


# Run the brute-force method
brute_force_tree, brute_force_error = brute_force_decision_tree(X, y, 3)
print("Brute-force method:")
print(f"Error: {brute_force_error}")
print_tree(brute_force_tree)

# Run the binary entropy method
binary_entropy_tree = binary_entropy_decision_tree(X, y, 3)
binary_entropy_error = evaluate_tree(binary_entropy_tree, X, y)
print("\nBinary entropy method:")
print(f"Error: {binary_entropy_error}")
print_tree(binary_entropy_tree)
