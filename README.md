# Gaussian Likelihood Score

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* scikit-learn
* scipy
* numpy


### Installing

You can install the method by typing:
```
pip install gls
```

### Basic usage

```python
from gls import gaussian_likelihood_score, find_best_cluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs


# define number of clusters
k = 5

# generate dummy data
X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=k,
                  cluster_std=1,
                  center_box=(-15.0, 15.0),
                  shuffle=True)

# get score for each possible k
gls_scores = []
for i in range(2, 10):
    clusterer = AgglomerativeClustering(n_clusters=i).fit(X)
    labels = clusterer.labels_
    # append score for the hypothesis `i`
    gls_scores.append(gaussian_likelihood_score(X, labels))

# find best cluster. `+ 2` because we start with i == 2
guess = find_best_cluster(gls_scores) + 2
print(f"GLS guess: {guess}")
```

## Authors

Maixent Chenebaux