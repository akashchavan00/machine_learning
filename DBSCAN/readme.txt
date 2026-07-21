# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN is a clustering algorithm that groups together points that are closely packed (high 
density), while marking points in low-density regions as outliers/noise. Unlike K-Means, it 
doesn't require you to specify the number of clusters beforehand, and it can find clusters 
of arbitrary shape.

## Core Concepts

DBSCAN relies on two parameters:

1. **eps (ε)** — the radius of the neighborhood around a point.
2. **minPts** — the minimum number of points required within that radius for a point to be 
considered a "dense" region (a core point).

Based on these, every point in the dataset is classified into one of three types:

- **Core point**: A point that has at least `minPts` points (including itself) within 
distance `eps`.
- **Border point**: A point that has fewer than `minPts` neighbors within `eps`, but lies 
within the neighborhood of a core point.
- **Noise point**: A point that is neither a core point nor a border point — it doesn't 
belong to any cluster.

## Algorithm Steps

1. Pick an arbitrary unvisited point `p`.
2. Retrieve all points within distance `eps` of `p` — this is `p`'s neighborhood.
3. If the neighborhood has at least `minPts` points, `p` is a core point, and a new cluster 
is started.
   - Add all points in the neighborhood to the cluster.
   - Recursively expand the cluster: for each neighbor that is also a core point, add *its* 
   neighbors to the cluster too (this is how clusters "grow" through connected dense regions).
4. If the neighborhood has fewer than `minPts` points, `p` is temporarily labeled noise 
(it may later be claimed as a border point of another cluster).
5. Repeat until every point has been visited.

At the end, points are grouped into clusters or labeled as noise.

## Why It's Useful

- **No need to specify number of clusters** in advance.
- **Finds arbitrarily shaped clusters** (not just spherical, unlike K-Means).
- **Robust to outliers** — naturally identifies noise points.
- **Sensitive to two intuitive parameters** (`eps`, `minPts`) instead of an arbitrary 
cluster count.

## Limitations

- Struggles when clusters have **varying densities** (a single `eps` may not suit all clusters).
- Performance and results are sensitive to **parameter selection**.
- Can be slow on very large datasets without spatial indexing (though KD-Trees/Ball-Trees help).
- Struggles in **high-dimensional spaces** where distance metrics become less meaningful.
