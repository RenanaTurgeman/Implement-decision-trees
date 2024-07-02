from itertools import combinations, product
def load_data(file_path):
    vectors = []
    with open(file_path, 'r') as file:
        for line in file:
            vector = list(map(int, line.strip().split()))
            vectors.append(vector)
    return vectors


# Generate all possible splits
def generate_splits(data, labels, max_depth):
    features = list(range(len(data[0]) - 1))
    possible_splits = []
    for depth in range(1, max_depth + 1):
        for split_features in combinations(features, depth):
            for split_conditions in product([0, 1], repeat=depth):
                possible_splits.append((split_features, split_conditions))
    return possible_splits

def calculate_error_for_split(data, labels, split):
    split_features, split_conditions = split
    predictions = []
    for vector in data:
        for feature, condition in zip(split_features, split_conditions):
            if vector[feature] != condition:
                predictions.append(0)
                break
        else:
            predictions.append(1)
    return sum(pred != label for pred, label in zip(predictions, labels)) / len(labels)

def brute_force_tree(data, labels, max_depth):
    best_split = None
    min_error = float('inf')
    splits = generate_splits(data, labels, max_depth)
    for split in splits:
        error = calculate_error_for_split(data, labels, split)
        if error < min_error:
            min_error = error
            best_split = split
    return best_split, min_error


# Function to print the decision tree
def print_tree(tree):
    for feature, value in tree:
        print(f"If feature {feature} == {value}, then ", end="")
    print("label is 1")
    print("Otherwise, label is 0")


if __name__ == '__main__':

    # Prepare data and labels
    vectors = load_data('vectors.txt')
    data = [vector[:-1] for vector in vectors]
    labels = [vector[-1] for vector in vectors]

    # Find the best split and calculate error
    max_depth = 3
    best_split, min_error = brute_force_tree(data, labels, max_depth)
    print(f"Best split for strategy A (Brute-force): {best_split} with error {min_error}")

