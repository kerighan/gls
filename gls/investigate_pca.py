import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD


# x = np.linspace(0, 1, 100)
# y = 3 * x + 50 + .2 * np.random.normal(size=100)
# vectors = np.vstack([y, x]).T

x = np.random.normal(size=(100, 2))
x_2 = np.random.normal(size=(100, 2))
x_2[:, 0] += 4
x_2[:, 1] += 4
vectors = np.vstack([x, x_2])
print(vectors.shape)

y_r = PCA(2).fit_transform(vectors)

plt.scatter(vectors[:, 0], vectors[:, 1])
plt.show()

plt.scatter(y_r[:, 0], y_r[:, 1])
plt.show()
