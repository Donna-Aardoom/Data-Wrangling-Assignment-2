"""
Reproduction of Figure 9 from Breunig et al. (2000), "LOF: Identifying Density-Based Local Outliers".
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.default_rng(42)

def gaussian(center, sigma, n):
    return rng.normal(center, sigma, size=(n, 2))

def uniform(center, radius, n):
    r = radius * np.sqrt(rng.uniform(size=n))
    theta = rng.uniform(0, 2 * np.pi, size=n)
    return center + np.column_stack([r * np.cos(theta), r * np.sin(theta)])

# clusters (coordinates estimated from Figure 9)
X = np.vstack([
    gaussian((4.5, 5.75), 0.50, 200),    # sparse Gaussian
    gaussian((8.85, 8.10), 0.20, 500),   # dense Gaussian
    uniform((2.4, 4.3), 0.50, 500),      # dense uniform
    uniform((7.3, 2.25), 1.15, 500),     # sparse uniform
    [[0.9, 7.1], [1.65, 5.75], [3.75, 2.0],
     [7.1, 6.25], [7.3, 5.95], [10.35, 2.35]],  # outliers
])

# LOF 
lof = LocalOutlierFactor(n_neighbors=40).fit(X)
scores = -lof.negative_outlier_factor_

# Figure 9: dataset + LOF
fig = plt.figure(figsize=(13, 5.5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(X[:, 0], X[:, 1], s=2, c="k")
ax1.set(xlim=(0, 11), ylim=(0, 9), aspect="equal")
ax1.set_title("Dataset")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.bar3d(X[:, 0], X[:, 1], 0, 0.08, 0.08, scores, color="dimgray", shade=True)
ax2.set(xlim=(0, 11), ylim=(0, 9), zlim=(0, max(16, scores.max())),
        title="Outlier factors (MinPts=40)")
ax2.set_zlabel("outlier factor LOF", labelpad=2)  
ax2.zaxis.set_label_position("lower")
ax2.zaxis.set_ticks_position("lower")
ax2.view_init(elev=15, azim=-60)

fig.tight_layout()
fig.savefig("lof_figure9.png", dpi=150, bbox_inches="tight")   
