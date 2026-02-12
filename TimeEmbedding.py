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


