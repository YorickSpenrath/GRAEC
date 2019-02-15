"""
Created on Wed Oct 17 11:57:43 2018

@author: 179040

Graduation Project Yorick Spenrath
Eindhoven University of Technology
For the degree "Master of Science"
In the programs Business Information Systems
                Operations Management and Logistics
In association with Kropman B.V., Nijmegen
For more info please contact the original author at "yorick.spenrath@gmail.com"

"""

"""
Functions to handle K medoid reduction
"""
import scipy
import numpy as np


def reduce_to_medoids(data_X, data_y, factor=2, dfun='jaccard', return_indices=False):
    # Get all classes
    classes, class_size = np.unique(data_y, return_counts=True)
    max_clust_size = factor * np.min(class_size)
    indices = []
    for c, c_count in zip(classes, class_size):
        if (c_count <= max_clust_size):
            indices.extend(np.where(data_y == c)[0].tolist())
        else:
            indices_cluster = np.where(data_y == c)[0].tolist()
            cluster_X = data_X[indices_cluster]
            dists_X = generate_boolean_distances(cluster_X, t=dfun)
            clusters, medoids = cluster(dists_X, max_clust_size)
            indices.extend([indices_cluster[i] for i in medoids])

    return (data_X[indices], data_y[indices]) + ((indices,) if return_indices else ())


def generate_boolean_distances(data_x, t='jaccard'):
    return scipy.spatial.distance.cdist(data_x, data_x, metric=t)


def getTimeRepresentatives(data_x, k_rep):
    distances = generate_euclidean_distances(data_x)
    return data_x[cluster(distances, k_rep)[1]]


def generate_euclidean_distances(data_x):
    return scipy.spatial.distance.cdist(data_x, data_x, metric='euclidean')


def cluster(distances, k=3):
    m = distances.shape[0]  # number of points

    # Pick k random medoids.
    curr_medoids = np.random.choice(np.arange(m), k, replace=False)

    old_medoids = np.array([-1] * k)  # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1] * k)

    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    return clusters, curr_medoids


def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:, medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters


def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster, cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)
