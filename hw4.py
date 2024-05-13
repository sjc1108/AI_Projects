
import math
import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):
    with open(filepath, "r") as f:
        read = csv.DictReader(f)
        data = [dict(row) for row in read]

    return data

def calc_features(row):
    """takes in one row dict from the data loaded from the previous function then calculates the \
        corresponding feature vector for that country as specified above, and returns it as a NumPy \
            array of shape (6,). The dtype of this array should be float64"""
    
    features = np.array([
        float(row["Population"]),
        float(row["Net migration"]),
        float(row["GDP ($ per capita)"]),
        float(row["Literacy (%)"]),
        float(row["Phones (per 1000)"]),
        float(row["Infant mortality (per 1000 births)"])
    ], dtype = np.float64)

    return features

def eu_distance(x, y):
    """maximum straight dist between elements of two clusters"""
    max_dist = float('-inf')  # to - inf
    for a in x:
        for b in y:
            dist = np.linalg.norm(a - b)  #euclidean dist
            if dist > max_dist:
                max_dist = dist  # Update if current dist is greater

    return max_dist

def hac(features):
    """Performs complete linkage hierarchical agglomerative clustering on the country with the
       (x1, . . . , x6) feature representation, and returns a NumPy array representing the clustering."""
    clusters = {}  # dict where ea key-value pair is cluster
    for cluster_index in range(len(features)):  # each feature vector its own cluster initially
        clusters[cluster_index] = [features[cluster_index]]

    Z = np.empty((len(features) -1, 4))  # numpy array storing details of cluster mergings
    for merge_step in range(len(features)-1):  # find and merge two closest clusters
        closest_pair = None  # closest pair of clusters
        min_dist = float("inf")
        for clus_a in clusters.keys():
            for clus_b in clusters.keys():
                if clus_a < clus_b:
                    distance = eu_distance(clusters[clus_a], clusters[clus_b])
                    if distance < min_dist:
                        closest_pair, min_dist = (clus_a, clus_b), distance
        clus_a, clus_b = closest_pair
        Z[merge_step] = [clus_a, clus_b, min_dist, len(clusters[clus_a]) + len(clusters[clus_b])]
        clusters[len(features) + merge_step] = clusters.pop(clus_a) + clusters.pop(clus_b)
    return Z


def fig_hac(Z, names):
    """visualizes the hierarchical agglomerative clustering on the country’s feature representation."""
    fig = plt.figure()
    dendrogram(Z, labels = names, leaf_rotation = 90)
    plt.tight_layout()
    return fig

def normalize_features(features):
    """takes a list of feature vectors and computes the normalized values. The output should be a list \
        of normalized feature vectors in the same format as the input."""
    features = np.vstack(features)
    feature_means = features.mean(axis = 0) #mean for each feature
    feature_std = features.std(axis = 0)
    normalized_features = (features -feature_means)/ feature_std #ea vector is normalized

    return [np.array(row) for row in normalized_features]

