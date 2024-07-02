import numpy as np
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Load the data
file_path = 'vectors.txt'
data = np.loadtxt(file_path, dtype=int)

# Separate features and labels
X = data[:, :-1]
y = data[:, -1]

def calculate_error(y_left, y_right):
    """
    Function to calculate error for a given split
    :param y_left: A list or array of labels (0s and 1s) corresponding to the data points in the left subset after a split.
    :param y_right: A list or array of labels (0s and 1s) corresponding to the data points in the right subset after a split.
    :return: The total error between the two subset.
    """

    n_left = len(y_left)
    n_right = len(y_right)
    # logger.info(f" the number of elements in each subset: left {n_left}, right {n_right} ")
    if n_left == 0 or n_right == 0:
        # If either subset is empty, the error is set to infinity.
        return float('inf')
    p_left = np.sum(y_left) / n_left
    p_right = np.sum(y_right) / n_right
    error_left = np.minimum(p_left, 1 - p_left) * n_left
    error_right = np.minimum(p_right, 1 - p_right) * n_right
    return error_left + error_right


# A. Brute-force Method

def brute_force_decision_tree(X, y, k):
    """
    Brute-force Method
    :param X: the features
    :param y: the labels
    :param k: the decision tree will have up to k levels
    :return:
    """
    n_samples, n_features = X.shape
    min_error = float('inf')
    best_tree = None

    def generate_trees(depth, X, y, current_min_error):
        """
        Recursive function to generate all possible trees
        :param depth:
        :param X:
        :param y:
        :param current_min_error: track the minimum error within this scope
        :return:
        """
        if depth == k or len(np.unique(y)) == 1:
            return {'is_leaf': True, 'prediction': np.argmax(np.bincount(y))}

        local_min_error = current_min_error
        best_split = None
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            print(f"Thresholds: {thresholds}")
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                y_left = y[left_indices]
                y_right = y[right_indices]
                error = calculate_error(y_left, y_right)
                if error < local_min_error:
                    local_min_error = error
                    left_tree = generate_trees(depth + 1, X[left_indices], y_left, local_min_error)
                    right_tree = generate_trees(depth + 1, X[right_indices], y_right, local_min_error)
                    best_split = {
                        'is_leaf': False,
                        'feature': feature,
                        'threshold': threshold,
                        'left': left_tree,
                        'right': right_tree
                    }
        return best_split

    best_tree = generate_trees(0, X, y, min_error)
    return best_tree, min_error

# B. Binary Entropy Method (ID3 Algorithm)

def generate_trees(depth, X, y, min_error, k):

    """
    Recursive function to generate all possible trees
    :param depth:
    :param X:
    :param y:
    :param current_min_error: track the minimum error within this scope
    :return:
    """
    n_samples, n_features = X.shape

    if depth == k or len(np.unique(y)) == 1:
        print("in the first if")
        return {'is_leaf': True, 'prediction': np.argmax(np.bincount(y))}

    best_split = None
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        print(f"thresholds: {thresholds}---------")
        for threshold in thresholds:
            left_indices = X[:, feature] <= threshold
            right_indices = X[:, feature] > threshold
            y_left = y[left_indices]
            y_right = y[right_indices]
            error = calculate_error(y_left, y_right)
            if error < min_error:
                min_error = error
                left_tree, left_error = generate_trees(depth + 1, X[left_indices], y_left, min_error,k)
                right_tree, right_error = generate_trees(depth + 1, X[right_indices], y_right, min_error, k)
                best_split = {
                    'is_leaf': False,
                    'feature': feature,
                    'threshold': threshold,
                    'left': left_tree,
                    'right': right_tree
                }
    return best_split, min_error

def brute_force_decision_tree(X, y, k):
    """
    Brute-force Method
    :param X: the features
    :param y: the labels
    :param k: the decision tree will have up to k levels
    :return:
    """
    min_error = float('inf')

    best_tree, min_error = generate_trees(0, X, y, min_error, k)
    return best_tree, min_error

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
# binary_entropy_tree = binary_entropy_decision_tree(X, y, 3)
# binary_entropy_error = evaluate_tree(binary_entropy_tree, X, y)
# print("\nBinary entropy method:")
# print(f"Error: {binary_entropy_error}")
# print_tree(binary_entropy_tree)
