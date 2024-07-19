import numpy as np
from scipy.stats import entropy
from collections import Counter


def binary_entropy(labels):
    """
       Calculate the binary entropy of a set of labels.

       :param: labels: An array of labels (integers) for which to calculate the entropy.
       :returns: The binary entropy of the input labels.

       >>> labels = np.array([0, 1, 0, 1, 1, 0])
       >>> binary_entropy(labels)
       1.0
    """
    label_counts = np.bincount(labels)
    probabilities = label_counts / np.sum(label_counts)
    return entropy(probabilities, base=2)


def compute_error(tree, features, labels):
    """
      Calculate the error rate of a decision tree.

      This function computes the error rate of a decision tree by comparing the tree's
      predictions on the given features to the true labels.

      :param: tree: The decision tree used for making predictions.
      :param: features: The feature matrix where each row is a sample and each column is a feature.
      :param: labels: The true labels corresponding to each sample in the feature matrix.

      :returns: The error rate of the decision tree.
    """
    incorrect_predictions = 0
    total_predictions = len(labels)

    for i, feature in enumerate(features):
        prediction = predict(tree, [feature])[0]
        if prediction != labels[i]:
            incorrect_predictions += 1

    error_rate = incorrect_predictions / total_predictions
    return error_rate


def predict(tree, features):
    """
        Predict labels using a decision tree for a set of features.

        This function traverses the decision tree to predict labels for each sample
        in the feature matrix.

        :param: tree: The decision tree used for making predictions.
                      The tree is expected to be in dictionary format where each node contains 'feature', 'left',
                     'right', and 'value' keys.
        :param: features: The feature matrix where each row is a sample and each column is a feature.

        :returns: An array containing predicted labels for each sample in the feature matrix.

        Example:
        >>> tree = {
        ...     'feature': 4,
        ...     'left': {
        ...         'feature': 1,
        ...         'left': {'value': 0},
        ...         'right': {'value': 1}
        ...     },
        ...     'right': {
        ...         'feature': 0,
        ...         'left': {'value': 1},
        ...         'right': {'value': 0}
        ...     }
        ... }
        >>> features = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
        >>> predict(tree, features)
        array([0, 1])
        """
    predictions = []
    for feature in features:
        node = tree
        while isinstance(node, dict) and "feature" in node:
            split_feature = node["feature"]
            node = node["left"] if feature[split_feature] == 0 else node["right"]
        predictions.append(node["value"])
    return np.array(predictions)


def find_best_split(features, labels):
    """
        Find the best feature to split on based on minimum entropy.

        This function evaluates each feature in the feature matrix to determine
        which one provides the best split based on minimum weighted entropy.

        :param: features: The feature matrix where each row is a sample and each column is a feature.
        :param: labels: The true labels corresponding to each sample in the feature matrix.

        :returns: The index of the optimal feature to split on, or None if no valid split is found.

        Example:
        >>> features = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]])
        >>> labels = np.array([0, 1, 0])
        >>> find_best_split(features, labels)
        0
        """
    optimal_feature = None
    minimum_entropy = float('inf')
    for i in range(features.shape[1]):
        left_mask = features[:, i] == 0
        right_mask = features[:, i] == 1
        if len(labels[left_mask]) == 0 or len(labels[right_mask]) == 0:
            continue
        left_entropy = binary_entropy(labels[left_mask])
        right_entropy = binary_entropy(labels[right_mask])
        weighted_entropy = (len(left_mask) * left_entropy + len(right_mask) * right_entropy) / len(labels)
        if weighted_entropy < minimum_entropy:
            minimum_entropy = weighted_entropy
            optimal_feature = i
    return optimal_feature


def divide_node(node, features, labels, current_depth, max_depth):
    """
      Recursively divides a node in a decision tree based on the best split feature.

      This function recursively builds the decision tree by splitting nodes based on
      the best feature found using minimum entropy. It stops splitting when the maximum
      depth is reached or when all labels in a node are the same.

      :param: node: The current node of the decision tree represented as a dictionary.
                   Contains keys 'feature', 'left', 'right', and 'value'.
      :param: features: The feature matrix where each row is a sample and each column is a feature.
      :param: labels: The true labels corresponding to each sample in the feature matrix.
      :param: current_depth: The current depth of the node in the tree.
      :param: max_depth: The maximum depth allowed for the tree.

      Example:
      >>> node = {}
      >>> features = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]])
      >>> labels = np.array([0, 1, 0])
      >>> divide_node(node, features, labels, 0, 2)
      """

    if current_depth >= max_depth or len(set(labels)) == 1:
        counter = Counter(labels)
        node["value"] = counter.most_common(1)[0][0]
        return

    split_feature = find_best_split(features, labels)
    if split_feature is None:
        counter = Counter(labels)
        node["value"] = counter.most_common(1)[0][0]
        return

    left_mask = np.where(features[:, split_feature] == 0)[0]
    right_mask = np.where(features[:, split_feature] == 1)[0]

    if len(left_mask) == 0 or len(right_mask) == 0:
        counter = Counter(labels)
        node["value"] = counter.most_common(1)[0][0]
        return

    node["feature"] = split_feature
    node["left"] = {}
    node["right"] = {}

    divide_node(node["left"], features[left_mask], labels[left_mask], current_depth + 1, max_depth)
    divide_node(node["right"], features[right_mask], labels[right_mask], current_depth + 1, max_depth)


def construct_tree(features, labels, max_depth):
    """
       Constructs a decision tree based on the given features and labels.

       This function constructs a decision tree recursively by calling the divide_node
       function to split nodes based on the best feature found using minimum entropy.

       :param: features: The feature matrix (2D array) where each row is a sample and each column is a feature.
       :param: labels: The true labels (1D array) corresponding to each sample in the feature matrix.
       :param: max_depth: The maximum depth allowed for the tree.

       :returns: The root node of the constructed decision tree represented as a dictionary.
    """
    root = {}
    divide_node(root, features, labels, 0, max_depth)
    return root


def brute_force_method(available_features, depth, memo):
    """
        Enumerates all possible binary trees up to a given depth using the available features.

        This function recursively generates all possible binary trees by combining features
        at each level up to the specified depth. Each tree node is represented as a dictionary
        containing keys 'feature', 'left', and 'right'.

        :param available_features: The features available for splitting nodes in the tree.
        :param depth: The maximum depth of the trees to generate.
        :param memo: A dictionary to memoize results of subtrees to avoid redundant calculations.

        :returns: A list of dictionaries, each representing a binary tree.
       """

    if depth == 0:
        return [{"value": 0}, {"value": 1}]
    if depth in memo:
        return memo[depth]

    trees = []
    for feature in available_features:
        for left_depth in range(depth):
            for right_depth in range(depth):
                left_trees = brute_force_method(available_features, left_depth, memo)
                right_trees = brute_force_method(available_features, right_depth, memo)
                for left_tree in left_trees:
                    for right_tree in right_trees:
                        root = {"feature": feature, "left": left_tree, "right": right_tree}
                        trees.append(root)
    memo[depth] = trees
    return trees


def print_tree(tree, depth=0, is_left=None):
    result = ""
    if depth > 0:
        if is_left:
            result += "│   " * (depth - 1) + "├── "
        else:
            result += "│   " * (depth - 1) + "└── "

    if "feature" in tree:
        result += f"Split feature: {tree['feature']}\n"
        result += print_tree(tree["left"], depth + 1, True)
        result += print_tree(tree["right"], depth + 1, False)
    else:
        result += f"Leaf value: {tree['value']}\n"
    return result


if __name__ == '__main__':
    # Load the vectors from the file
    data_vectors = np.loadtxt('vectors.txt', dtype=int)

    # Split the data into features and labels
    feature_matrix = data_vectors[:, :-1]
    label_vector = data_vectors[:, -1]

    # Parameters
    max_depth = 2  # because it goes over 0, 1, 2
    available_features = range(feature_matrix.shape[1])

    # Generate all possible trees of max_depth levels
    memo = {}
    possible_trees = brute_force_method(available_features, max_depth, memo)

    # Find the tree with the smallest error
    optimal_tree = None
    smallest_error = float('inf')
    for tree in possible_trees:
        error = compute_error(tree, feature_matrix, label_vector)
        if error < smallest_error:
            smallest_error = error
            optimal_tree = tree

    # Output the best tree and its error
    print("Results of Brute-Force Method:")
    print("Optimal Tree:")
    print(print_tree(optimal_tree))
    print(f"Brute-Force Smallest Error: {smallest_error:.4f}")
    optimal_accuracy = (1 - smallest_error) * 100
    print(f"Brute-Force Optimal Prediction Accuracy: {optimal_accuracy:.4f}%")

    print("\nResults of Entropy Method:")
    # Construct entropy-based decision tree
    entropy_based_tree = construct_tree(feature_matrix, label_vector, max_depth)

    # Calculate the error of the binary entropy tree
    entropy_tree_error = compute_error(entropy_based_tree, feature_matrix, label_vector)

    print("Binary Entropy Tree Structure:")
    print(print_tree(entropy_based_tree))
    print(f"Binary Entropy Tree Error: {entropy_tree_error:.4f}")
    print(f"Binary Entropy Tree Accuracy: {(1 - entropy_tree_error) * 100:.4f}%")