"""
       
               Time embedding of pituitary calcium time series 

        This code asume you already have obtained your cell classes to 
             adjust the time delay embedding parameters

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D  # Not necessary in newer matplotlib versions
import matplotlib.gridspec as gridspec


## Time series comming from previous 
PopA = time_series_1[:,:600].T
PopB = time_series_2[:,:600].T

CellA = len(PopA[0,:])
CellB = len(PopB[0,:])

#### rename column to Cell A + number to refer to those cells from pop A 
c = []

for i in range(len(PopA[0,:])):
    c.append("A_cell_" + str(i))
    
dfA = pd.DataFrame(PopA, columns=c)

#### rename column to Cell B + number to refer to those cells from pop b 

c = []

for i in range(len(PopB[0,:])):
    c.append("B_cell_" + str(i))
    
dfB = pd.DataFrame(PopB, columns=c)

###### once labeled each column, merge both df

df = pd.concat([dfA,dfB], axis = 1)

# Time-delay embedding function
def time_delay_embedding(x, m, tau):
    N = len(x) - (m - 1) * tau
    return np.array([x[i:i + m * tau:tau] for i in range(N)])

# Parameters
m, tau = 2, 1

# Embed each cell
embeddings = [time_delay_embedding(dfA[col].values, m, tau) for col in dfA.columns]
min_len = min(e.shape[0] for e in embeddings)
embedded_data = [e[:min_len] for e in embeddings]

# Build full multivariate embedding
X = np.hstack(embedded_data)

# Correct label assignment
labels = np.array(["A"] * CellA + ["B"] * CellB)
group_labels = np.repeat(labels, min_len)

# PCA projection
X_centered = X - np.mean(X, axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
X_pca = X_centered @ Vt.T[:, :2]

# Split A and B indices
a_idx = np.where(group_labels == "A")[0]
b_idx = np.where(group_labels == "B")[0]

# Ensure indices are within bounds
a_idx = a_idx[a_idx < X_pca.shape[0]]
b_idx = b_idx[b_idx < X_pca.shape[0]]

# Plot
plt.figure()
plt.scatter(X_pca[a_idx, 0], X_pca[a_idx, 1], s=20, alpha=0.8, label="Population A", c="lightpink")
plt.scatter(X_pca[a_idx, 0], X_pca[a_idx, 1], s=20, alpha=0.5, label="Population B", c="aquamarine")
plt.scatter(X_pca[0, 0], X_pca[0, 1], color="green", s=200, label="Start", marker='*')
plt.scatter(X_pca[-1, 0], X_pca[-1, 1], color="black", s=200, label="End", marker='X')

plt.title("PCA of Embedded Neural Activity by Population")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


"""
Time-delay embedded population trajectories
Comparison of Population A vs Population B
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression

# =========================
# Input time series
# =========================
# shape assumed: (cells, time)
PopA = time_series_1[:, :600]
PopB = time_series_2[:, :600]

PopA = PopA.T  # (time, cells)
PopB = PopB.T

CellA = PopA.shape[1]
CellB = PopB.shape[1]

# =========================
# Build DataFrames
# =========================
dfA = pd.DataFrame(
    PopA, columns=[f"A_cell_{i}" for i in range(CellA)]
)

dfB = pd.DataFrame(
    PopB, columns=[f"B_cell_{i}" for i in range(CellB)]
)

# =========================
# Time-delay embedding
# =========================
def time_delay_embedding(x, m, tau):
    N = len(x) - (m - 1) * tau
    return np.array([x[i:i + m * tau:tau] for i in range(N)])

def population_embedding(df, m, tau):
    embeddings = [
        time_delay_embedding(df[col].values, m, tau)
        for col in df.columns
    ]
    min_len = min(e.shape[0] for e in embeddings)
    return np.hstack([e[:min_len] for e in embeddings])

# embedding parameters
m = 2
tau = 1

XA = population_embedding(dfA, m, tau)
XB = population_embedding(dfB, m, tau)

# align trajectory lengths
T = min(len(XA), len(XB))
XA = XA[:T]
XB = XB[:T]

# =========================
# Joint PCA (shared phase space)
# =========================
X_all = np.vstack([XA, XB])
X_centered = X_all - X_all.mean(axis=0)

U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

XA_pca = XA @ Vt.T[:, :2]
XB_pca = XB @ Vt.T[:, :2]

# =========================
# Phase diagram
# =========================
plt.figure(figsize=(7, 6))

plt.plot(XA_pca[:, 0], XA_pca[:, 1], '-o',
         label='Population A', alpha=0.8)
plt.plot(XB_pca[:, 0], XB_pca[:, 1], '-o',
         label='Population B', alpha=0.8)

plt.scatter(XA_pca[0, 0], XA_pca[0, 1],
            c='green', s=120, marker='*', label='Start')
plt.scatter(XA_pca[-1, 0], XA_pca[-1, 1],
            c='black', s=120, marker='X', label='End')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Population Phase-Space Trajectories")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Quantitative comparisons
# =========================

# ---- 1. Distance between trajectories
dist_t = np.linalg.norm(XA_pca - XB_pca, axis=1)

# ---- 2. Velocity (speed in phase space)
vA = np.linalg.norm(np.diff(XA_pca, axis=0), axis=1)
vB = np.linalg.norm(np.diff(XB_pca, axis=0), axis=1)

# ---- 3. Directional similarity (cosine of velocity angle)
dXA = np.diff(XA_pca, axis=0)
dXB = np.diff(XB_pca, axis=0)

cos_angle = np.sum(dXA * dXB, axis=1) / (
    np.linalg.norm(dXA, axis=1) * np.linalg.norm(dXB, axis=1)
)

# ---- 4. ε-intersections (shared states)
epsilon = 0.5
treeA = cKDTree(XA_pca)
close_encounters = treeA.query_ball_point(XB_pca, r=epsilon)
num_close_states = sum(len(i) > 0 for i in close_encounters)

# ---- 5. Trajectory geometry (convex hull area)
areaA = ConvexHull(XA_pca).volume
areaB = ConvexHull(XB_pca).volume

# =========================
# Output summary
# =========================
print("Mean distance between trajectories:", np.mean(dist_t))
print("Mean speed Pop A:", np.mean(vA))
print("Mean speed Pop B:", np.mean(vB))
print("Mean directional similarity:", np.mean(cos_angle))
print("Number of close encounters:", num_close_states)
print("Convex hull area Pop A:", areaA)
print("Convex hull area Pop B:", areaB)
