
"""
                 Time-delay embedded population trajectories
                  Comparison of Population A vs Population B
                  
                  
                  updated version; with comparision metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression

# Input time series
# shape assumed: (cells, time)
PopA = time_series_1[:, :600]
PopB = time_series_2[:, :600]

PopA = PopA.T  # (time, cells)
PopB = PopB.T

CellA = PopA.shape[1]
CellB = PopB.shape[1]


# Build DataFrames

dfA = pd.DataFrame(
    PopA, columns=[f"A_cell_{i}" for i in range(CellA)]
)

dfB = pd.DataFrame(
    PopB, columns=[f"B_cell_{i}" for i in range(CellB)]
)

# Time-delay embedding


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
### embeding population in a separate way 
XA = population_embedding(dfA, m, tau)
XB = population_embedding(dfB, m, tau)

# align trajectory lengths
T = min(len(XA), len(XB))
XA = XA[:T]
XB = XB[:T]

### Version Corrected

# Center each trajectory 

XA_c = XA - XA.mean(axis = 0)
XB_c = XB - XB.mean(axis = 0)

K = XA_c @ XA_c.T + XB_c @ XB_c.T

K.shape == (T,T)

eigvals, eigvecs = np.linalg.eigh(K)

# Sort descending 
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

### select the number of dimensions 

components = 23

Z = eigvecs[:,:components] * np.sqrt(eigvals[:components])

ZA = Z
ZB = Z

plt.figure(figsize=(7,6))
plt.scatter(ZA[0,0], ZA[0,1], c='green', s=500, marker='*', label='Start')
plt.scatter(ZA[-1,0], ZA[-1,1], c='black', s=500, marker='X', label='End')

plt.plot(ZA[:,0], ZA[:,1], 'o', label='Class 1', alpha =0.3)
plt.plot(ZB[:,0], ZB[:,1], 'o', label='Class 2', alpha =0.2)

plt.xlabel("Latent dim 1")
plt.ylabel("Latent dim 2")
plt.title("Shared Population Phase Space (Time-aligned PCA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#####  CrossValidationValue to inferred influence of both populations

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# predict XB(t+1)
Y = XB[1:]
XB_now = XB[:-1]
XA_now = XA[:-1]

# model 1: self-prediction
model_self = Ridge(alpha=1.0)
score_self = cross_val_score(
    model_self, XB_now, Y, cv=5, scoring='r2'
).mean()

# model 2: with A
X_joint = np.hstack([XB_now, XA_now])
model_joint = Ridge(alpha=1.0)
score_joint = cross_val_score(
    model_joint, X_joint, Y, cv=5, scoring='r2'
).mean()

print("B self-prediction:", score_self)
print("B prediction with A:", score_joint)




"""
#### Synchronous similarity between geomitral space 

"""
# Normalize embedding times per population 
XA_n = (XA - XA.mean(axis=0)) / XA.std(axis=0)
XB_n = (XB - XB.mean(axis=0)) / XB.std(axis=0)

## Time PCA 
# Time–time covariance
K = XA_n @ XA_n.T + XB_n @ XB_n.T   # shape (T, T)

# Eigen-decomposition
eigvals, eigvecs = np.linalg.eigh(K)

# sort descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# latent coordinates
n_components = 3
Z = eigvecs[:, :n_components] * np.sqrt(eigvals[:n_components])


# verify the shape correspondo to Z 


Z.shape == (T, n_components)

#Project population onto temporal models 

ZA = XA_n @ XA_n.T @ eigvecs[:, :n_components]
ZB = XB_n @ XB_n.T @ eigvecs[:, :n_components]

#Normalize

ZA /= np.linalg.norm(ZA, axis=0, keepdims=True)
ZB /= np.linalg.norm(ZB, axis=0, keepdims=True)

# Time aligned distance

d_t = np.linalg.norm(ZA - ZB, axis=1)


plt.figure(figsize=(7,4))
plt.plot(d_t)
plt.xlabel("Time")
plt.ylabel("Population distance")
plt.title("Time-aligned dynamical distance")
plt.grid(True)
plt.tight_layout()
plt.show()



# Similar State epochs 

epsilon = np.percentile(d_t, 10)
similar_times = np.where(d_t < epsilon)[0]

# Plot 
plt.figure()
plt.scatter(Z[:,0], Z[:,1], c='lightgray')
plt.scatter(Z[similar_times,0], Z[similar_times,1],
            c='red', label='Similar states')
plt.legend()
plt.show()
