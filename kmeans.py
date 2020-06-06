import numpy as np
import matplotlib.pyplot as plt

def kmeans(data, k=-1):
    """A function that performs k-means clustering given an mxn input NumPy array."""
    if k < 0:
        print("Please enter a valid k value")
    else:
        cluster_labels = np.zeros(data.shape[0], dtype=float)
        cluster_labels = kmeans_proc(data, int(k))

    return cluster_labels

def kmeans_proc(data, k):
    """A function that performs the k-means procedure"""
    # Number of columns (m) and number of rows (n)
    m = data.shape[0]
    n = data.shape[1]
    
    # Array that contains the cluster classification of each observation
    labels = np.zeros(m, dtype=float)
    new_labels = np.zeros(m, dtype=float)
    
    # Array that contains the distance of each observation to the cluster centroid of its current cluster
    distances = np.zeros(m, dtype=float)
    
    # Select initial k cluster centers from the set of observations using random sampling
    np.random.seed(100)
    cluster_centers = data[np.random.choice(m, k, replace=False)]

    # Procedure will continue until there are observations that 'migrates' to another cluster
    n_migrations = 1
    while n_migrations != 0:

        # Compute distances of each observation to the cluster centroids
        for i in range(0, m):
            dist = compute_cluster_dist(data[i], cluster_centers)
            
            # Assigns each observation to the cluster with minimum centroid distance
            new_labels[i] = np.argmin(dist)
        
            # Stores the distance of each observation to its current cluster centroid
            distances[i] = np.min(dist)
            
        # Checks how many observations migrated to another cluster
        n_migrations = sum(labels != new_labels)

        # Assign the newly computed labels as the current label
        labels = new_labels.copy()

        # Computed the sum of square distances of each observation from their corresponding cluster centers
        variance = sum(map(lambda x:x*x, distances))
        
        # Computes for the new cluster centers
        if(n_migrations != 0): 
            for i in range(0, k):
                temp_sum = np.zeros(n)
                for j in range(0, m):
                    if (labels[j] == i):
                        temp_sum += data[j]
                cluster_centers[i] = temp_sum / len(labels[labels == i])
        
    return labels

def compute_cluster_dist(obs, cluster_centers):
    """A function that computes the distances of an obsevation to the cluster centers"""
    m = len(cluster_centers)
    dist = np.zeros(m)
    
    # Computes the eulidean distance of each observation to each cluster centers.
    for i in range(0, m):
        dist[i] = compute_euclidean_dist(obs, cluster_centers[i])
        
    return dist

def compute_euclidean_dist(arr1, arr2):
    """A function that computes the euclidean distance between two points"""
    sum_sqdiff = 0
    for i in range(0, len(arr1)):
        sum_sqdiff += (arr1[i] - arr2[i]) ** 2
    return np.sqrt(sum_sqdiff)