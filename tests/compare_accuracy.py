from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from metric import gaussian_likelihood_score, find_best_cluster
from tqdm import tqdm
import numpy as np


np.seterr(divide='ignore')
MAX_NUMBER_OF_CLUSTERS = 9


def generate_data(n_clusters):
    """generate blobs of data given a number of clusters."""
    X, y = make_blobs(n_samples=500,
                      n_features=2,
                      centers=n_clusters,
                      cluster_std=1,
                      center_box=(-10.0, 10.0),
                      shuffle=True)
    return X


def cluster(points, k):
    clustering = AgglomerativeClustering(n_clusters=k).fit(points)
    labels = clustering.labels_
    return labels


def find_optimal_number_of_clusters(points):
    # evaluate scores for all possible numbers of clusters
    scores_silhouette = []
    scores_davies_bouldin = []
    scores_gaussian_likelihood = []
    scores_gaussian_likelihood_pca = []
    for i in range(2, MAX_NUMBER_OF_CLUSTERS):
        labels = cluster(points, i)
        scores_silhouette.append(silhouette_score(points, labels))
        scores_davies_bouldin.append(davies_bouldin_score(points, labels))
        scores_gaussian_likelihood.append(
            gaussian_likelihood_score(points, labels, pca=False))
        scores_gaussian_likelihood_pca.append(
            gaussian_likelihood_score(points, labels, pca=True))

    # find the optimal number of clusters given the scores
    n_clusters_silhouette = np.argmax(scores_silhouette) + 2
    n_clusters_davies_bouldin = np.argmin(scores_davies_bouldin) + 2
    n_clusters_gaussian_likelihood = find_best_cluster(
        scores_gaussian_likelihood) + 2
    n_clusters_gaussian_likelihood_pca = find_best_cluster(
        scores_gaussian_likelihood_pca) + 2
    return (n_clusters_silhouette, n_clusters_davies_bouldin,
            n_clusters_gaussian_likelihood, n_clusters_gaussian_likelihood_pca)


def step(n_clusters):
    # let's generate some data
    points = generate_data(n_clusters)
    # let's find the optimal number of clusters
    # according to each scoring method
    results = find_optimal_number_of_clusters(points)
    n_clusters_silhouette = results[0]
    n_clusters_davies_bouldin = results[1]
    n_clusters_gaussian_likelihood = results[2]
    n_clusters_gaussian_likelihood_pca = results[3]

    # check wether a method is right or not
    silhouette_right = n_clusters_silhouette == n_clusters
    davies_bouldin_right = n_clusters_davies_bouldin == n_clusters
    gaussian_likelihood_right = n_clusters_gaussian_likelihood == n_clusters
    gaussian_likelihood_pca_right = n_clusters_gaussian_likelihood_pca \
        == n_clusters

    return (silhouette_right, davies_bouldin_right,
            gaussian_likelihood_right, gaussian_likelihood_pca_right)


def evaluate_scores(num_iter=1000):
    """
    1. randomly select a number of clusters.
    2. generate blobs using this number
    3. get the score of each method for all possible values of k
    4. find the optimal k according to each method
    5. add 1 to all successful guesses, 0 otherwise
    6. compare accuracy
    """
    accuracy_silhouette = 0
    accuracy_davies_bouldin = 0
    accuracy_gaussian_likelihood = 0
    accuracy_gaussian_likelihood_pca = 0
    for _ in tqdm(range(num_iter), "Generating samples"):
        # choose a random number of clusters
        k = np.random.randint(2, MAX_NUMBER_OF_CLUSTERS)
        results = step(k)

        # add successful inference
        accuracy_silhouette += results[0]
        accuracy_davies_bouldin += results[1]
        accuracy_gaussian_likelihood += results[2]
        accuracy_gaussian_likelihood_pca += results[3]

    accuracy_silhouette *= 100. / num_iter
    accuracy_davies_bouldin *= 100. / num_iter
    accuracy_gaussian_likelihood *= 100. / num_iter
    accuracy_gaussian_likelihood_pca *= 100. / num_iter

    print(accuracy_silhouette, "silhouette accuracy")
    print(accuracy_davies_bouldin, "Davies Bouldin accuracy")
    print(accuracy_gaussian_likelihood, "Gaussian Likelihood accuracy")
    print(accuracy_gaussian_likelihood_pca,
          "Gaussian Likelihood with PCA accuracy")


if __name__ == "__main__":
    evaluate_scores()
