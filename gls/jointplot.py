import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np



def make_close_blobs(num_clusters):
    centroids = np.array([[0, 0], [.3, .3]])
    sigmas = [.12, .12]
    num_points = [100, 100]

    # generate points
    points = []
    for i in range(num_clusters):
        cluster_points = np.random.normal(loc=centroids[i], scale=sigmas[i],
                                          size=(num_points[i], 2))
        points.append(cluster_points)
    points = np.vstack(points)
    return points


def plot_points(points):
    df = pd.DataFrame()
    df["x"] = points[:, 0]
    df["y"] = points[:, 1]
    sns.set(style="darkgrid")
    g = sns.jointplot("x", "y", data=df,
                    color="m", height=7)
    plt.show()


if __name__ == "__main__":
    points = make_close_blobs(2)
    plot_points(points)

    points_fitted = PCA(2).fit_transform(points)
    plot_points(points_fitted)
