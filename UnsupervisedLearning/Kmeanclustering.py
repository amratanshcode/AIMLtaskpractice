'''K-Means is an unsupervised learning algorithm used to group similar data points into clusters.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample data
X = np.array([
    [1,2],
    [1,4],
    [1,0],
    [10,2],
    [10,4],
    [10,0]
])

# Model
kmeans = KMeans(n_clusters=2)

# Train
kmeans.fit(X)

# Predict clusters
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_

# Plot
plt.scatter(X[:,0], X[:,1], c=labels)

# Plot centroids
plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200)

plt.title("K-Means Clustering")

plt.show()

