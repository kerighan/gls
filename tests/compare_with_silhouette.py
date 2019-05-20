from compare_accuracy import (
    generate_data, find_optimal_number_of_clusters, cluster)
import matplotlib.pyplot as plt


points = generate_data(4)
results = find_optimal_number_of_clusters(points)

n_clusters_silhouette = results[0]
n_clusters_gaussian_likelihood_pca = results[3]

labels = cluster(points, n_clusters_gaussian_likelihood_pca)
plt.scatter(points[:, 0], points[:, 1], c=labels)
plt.show()

labels = cluster(points, n_clusters_silhouette)
plt.scatter(points[:, 0], points[:, 1], c=labels)
plt.show()
