from evaluate import find_optimal_number_of_clusters
from sklearn import datasets


# illustration of the failure of the method on non-normal data
n_samples = 1000
noisy_circles, _ = datasets.make_circles(n_samples=n_samples,
                                         factor=.5, noise=.05)
noisy_moons, _ = datasets.make_moons(n_samples=n_samples, noise=.05)


results = find_optimal_number_of_clusters(noisy_moons)
# uncomment the line below to test with noisy circles
# results = find_optimal_number_of_clusters(noisy_circles)

n_clusters_silhouette = results[0]
n_clusters_davies_bouldin = results[1]
n_clusters_gaussian_likelihood = results[2]
n_clusters_gaussian_likelihood_pca = results[3]

print(n_clusters_silhouette, "silhouette guess")
print(n_clusters_gaussian_likelihood_pca, "GLS-PCA guess")
