import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA


__version__ = "1.0.0"


def gaussian_likelihood_score(points, labels, resolution=50, pca=True):
    probability = 0
    n_clusters = len(np.unique(labels))
    for j in range(n_clusters):
        cluster_points = points[labels == j]
        if pca:
            cluster_points = PCA(1).fit_transform(cluster_points)
            centroid = np.mean(cluster_points)
            spread = np.std(cluster_points)

            rv = norm(loc=centroid, scale=spread)
            hist, bins = np.histogram(cluster_points[:, 0], bins=resolution)
            probability_pca = np.sum(rv.logpdf(bins[:-1]) * hist)
            if np.isnan(probability_pca):
                probability_pca = 0.
            probability += probability_pca
        else:
            centroid = np.mean(cluster_points, axis=0)
            spread = np.std(cluster_points, axis=0)

            rv_1 = norm(loc=centroid[0], scale=spread[0])
            rv_2 = norm(loc=centroid[1], scale=spread[1])

            hist_1, bins_1 = np.histogram(
                cluster_points[:, 0], bins=resolution)
            hist_2, bins_2 = np.histogram(
                cluster_points[:, 1], bins=resolution)

            probability_axis_1 = np.sum(rv_1.logpdf(bins_1[:-1]) * hist_1)
            if np.isnan(probability_axis_1):
                probability_axis_1 = 0.
            probability_axis_2 = np.sum(rv_2.logpdf(bins_2[:-1]) * hist_2)
            if np.isnan(probability_axis_2):
                probability_axis_2 = 0.
            probability += probability_axis_1 + probability_axis_2
    return probability


def find_best_cluster(scores):
    scores = np.array(scores)
    # derivative of score
    d = scores - np.hstack([[0], scores[:-1]])
    dd = d[1:] / (d[:-1] + 1e-10)
    if dd[0] < 0:
        dd[0] = np.inf
    optimal_number = np.argmin(dd)
    return optimal_number
