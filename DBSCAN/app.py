import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Generate sample data (non-spherical clusters)
X, _ = make_moons(n_samples=300, noise=0.07, random_state=42)
X = StandardScaler().fit_transform(X)

# Apply DBSCAN
db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X)

# Number of clusters (ignoring noise, labeled -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")

# Plot
plt.figure(figsize=(7, 5))
unique_labels = set(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        color = 'k'  # noise points in black
    mask = labels == label
    plt.scatter(X[mask, 0], X[mask, 1], c=[color], s=30,
                label=f"Cluster {label}" if label != -1 else "Noise")

plt.title("DBSCAN Clustering")
plt.legend()
plt.show()