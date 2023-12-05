import argparse
import csv
import math
from collections import Counter
from fractions import Fraction
from collections import defaultdict


# Function to load data from a CSV file
def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Skip rows that don't have the expected number of columns
            if len(row) < 2:  # Assuming at least one feature and one class label
                continue

            # Convert feature values to float and add the class label
            try:
                numeric_row = [float(item) for item in row[:-1]] + [row[-1]]
                data.append(numeric_row)
            except ValueError:
                # Handle the case where conversion to float fails
                print(f"Warning: Skipping row due to invalid data - {row}")
    return data


# Helper function to calculate Euclidean distance squared
def euclidean_distance_squared(point1, point2):
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))

def manhattan_distance(point1, point2):
    return sum(abs(p1 - p2) for p1,p2 in zip(point1, point2))


# Function stub for kNN algorithm
def knn_classify(test_data, train_data, k, verbose=True):
    # Implement kNN classification here
    predictions = []
    for test_point in test_data:
        # Calculate squared distances from the test point to all training points
        distances = []
        for train_point in train_data:
            distance = euclidean_distance_squared(test_point[:-1], train_point[:-1])
            distances.append((distance, train_point[-1]))

        # Sort by distance and select the top k distances
        k_nearest = sorted(distances, key=lambda x: x[0])[:k]

        # Get the labels of the k nearest neighbors
        k_nearest_labels = [label for _, label in k_nearest]

        # Use majority vote for the predicted label
        most_common = Counter(k_nearest_labels).most_common(1)
        predicted_label = most_common[0][0]

        if verbose:
            actual_label = test_point[-1]
            print(f"want={actual_label} got={predicted_label}")

        predictions.append(predicted_label)

    return predictions

def train_naive_bayes(train_data, laplacian_correction):
    feature_counts = defaultdict(lambda: defaultdict(lambda: 0))
    class_counts = defaultdict(lambda: 0)
    feature_value_counts = defaultdict(lambda: set())

    for row in train_data:
        class_label = row[-1]
        class_counts[class_label] += 1
        for i, feature in enumerate(row[:-1]):
            feature_counts[class_label][i, feature] += 1
            feature_value_counts[i].add(feature)

    # Apply Laplacian correction if needed
    if laplacian_correction > 0:
        for class_label in class_counts:
            for feature_index, feature_values in feature_value_counts.items():
                q = len(feature_values)
                for value in feature_values:
                    feature_counts[class_label][feature_index, value] += laplacian_correction
                class_counts[class_label] += q * laplacian_correction

    return feature_counts, class_counts

def predict_naive_bayes(test_data, feature_counts, class_counts, verbose):
    predictions = []
    total_instances = sum(class_counts.values())

    for row in test_data:
        best_class, best_prob = None, 0
        for class_label, class_count in class_counts.items():
            # Calculate prior probability P(C)
            prior_prob = Fraction(class_count, total_instances)
            if verbose:
                print(f"P(C={class_label}) = [{prior_prob}]")

            # Calculate Naive Bayes probability NB(C)
            nb_prob = prior_prob
            for i, feature in enumerate(row[:-1]):
                # Calculate conditional probability P(Ax | C)
                cond_prob = Fraction(feature_counts[class_label][i, feature], class_counts[class_label])
                nb_prob *= cond_prob
                if verbose:
                    print(f"P(A{i}={feature} | C={class_label}) = {cond_prob}")

            # Compare and store the best probability
            if nb_prob > best_prob:
                best_class, best_prob = class_label, nb_prob

            if verbose:
                print(f"NB(C={class_label}) = {float(nb_prob)}")

        predictions.append(best_class)
        if verbose:
            actual_label = row[-1]
            result = "match" if best_class == actual_label else "fail"
            print(f"{result}: got \"{best_class}\" != want \"{actual_label}\"")

    return predictions


# Function stub for Naive Bayes algorithm
def naive_bayes_classify(test_data, train_data, laplacian_correction, verbose=True):
    # Implement Naive Bayes classification here
    feature_counts, class_counts = train_naive_bayes(train_data,laplacian_correction)
    predictions = predict_naive_bayes(test_data, feature_counts, class_counts, verbose)
    return predictions

def parse_centroids(centroid_args):
    centroids = []
    for centroid_str in centroid_args:
        # Convert string to a tuple of floats
        parts = centroid_str.split(',')
        try:
            point = tuple(float(part) for part in parts)
            centroids.append(point)
        except ValueError:
            raise ValueError(f"Invalid centroid format: {centroid_str}")
    return centroids



def assign_clusters(data, centroids, distance_func):
    clusters = {}
    for point in data:
        distances = [distance_func(point[:-1], centroid) for centroid in centroids]
        closest_centroid_index = min(range(len(distances)), key=distances.__getitem__)
        clusters.setdefault(closest_centroid_index, []).append(point)
    return clusters

def update_centroids(clusters, num_features):
    new_centroids = []
    for points in clusters.values():
        centroid = [sum(point[i] for point in points) / len(points) for i in range(num_features)]
        new_centroids.append(tuple(centroid))
    return new_centroids

# Function stub for k-Means clustering (extra credit)
def k_means_cluster(data, initial_centroids, distance_metric):
    centroids = parse_centroids(initial_centroids)

    # Choose the distance function
    if distance_metric == 'e2':
        distance_func = euclidean_distance_squared
    elif distance_metric == 'manh':
        distance_func = manhattan_distance
    else:
        # Default to Euclidean squared if not specified
        distance_func = euclidean_distance_squared

    # Implement k-Means clustering here
    num_features = len(data[0]) - 1  # Assuming last element is a label

    while True:
        clusters = assign_clusters(data, centroids, distance_func)
        new_centroids = update_centroids(clusters, num_features)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters

def calculate_metrics(predictions, actuals):
    true_positives = {}
    false_positives = {}
    false_negatives = {}

    for pred, actual in zip(predictions, actuals):
        if pred == actual:
            true_positives[pred] = true_positives.get(pred, 0) + 1
        else:
            false_positives[pred] = false_positives.get(pred, 0) + 1
            false_negatives[actual] = false_negatives.get(actual, 0) + 1

    metrics = {}
    labels = set(predictions + actuals)
    for label in labels:
        tp = true_positives.get(label, 0)
        fp = false_positives.get(label, 0)
        fn = false_negatives.get(label, 0)

        precision = Fraction(tp, tp + fp) if (tp + fp) > 0 else 0
        recall = Fraction(tp, tp + fn) if (tp + fn) > 0 else 0

        metrics[label] = {"Precision": precision, "Recall": recall}

    return metrics

def print_metrics(metrics):
    for label in sorted(metrics.keys()):
        metric = metrics[label]
        print(f"Label={label} Precision={metric['Precision']} Recall={metric['Recall']}")

def print_kmeans_results(centroids, clusters):
    for i, cluster in enumerate(clusters.values()):
        print(f"C{i+1} = {{", end="")
        print(','.join(point[-1] for point in cluster), end="}\n")
    for centroid in centroids:
        print(centroid)


# Main function
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Machine Learning Program")
    parser.add_argument('-train', required=True, help='Path to the training data file')
    parser.add_argument('-test', help='Path to the testing data file')
    parser.add_argument('-K', type=int, default=-1, help='Indicates to use kNN and the value of K (0 for Naive Bayes)')
    parser.add_argument('-C', type=int, default=0, help='Laplacian correction for Naive Bayes (0 means no correction)')
    parser.add_argument('-v', action='store_true', help='Verbose output flag')
    parser.add_argument('-d', choices=['e2', 'manh'], help='Distance metric for kMeans (e2 for Euclidean, manh for Manhattan)')
    parser.add_argument('centroids', nargs='*', type=str, help='List of centroids for kMeans (Extra Credit)')
    args = parser.parse_args()

    # Load training and testing data
    train_data = load_data(args.train)
    test_data = load_data(args.test) if args.test else None

    # Perform the selected algorithm based on the arguments
    if args.K > 0:
        predictions = knn_classify(test_data, train_data, args.K, args.v)
    elif args.K == 0:
        predictions = naive_bayes_classify(test_data, train_data, args.C, args.v)
    elif args.d:
        final_centroids, clusters = k_means_cluster(train_data, args.centroids, args.d)
        print_kmeans_results(final_centroids, clusters)


    # Calculate and print metrics
    if test_data:
        actual_labels = [row[-1] for row in test_data]
        metrics = calculate_metrics(predictions, actual_labels)
        print_metrics(metrics)


if __name__ == "__main__":
    main()
