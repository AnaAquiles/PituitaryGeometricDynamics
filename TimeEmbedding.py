"""
                     EMBEDDING TIME 2.0 correction

 to perform same logic computation with all my time series datasets divided by popA and PopB, how could I do ? 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes, svd
from scipy.stats import gaussian_kde

#### Select the condition and the number of subject 
time_series_1 = data_g1['Condition_2'] 
time_series_1 = time_series_1[3]

time_series_2 = data_g2['Condition_2'] 
time_series_2 = time_series_2[3]
# INPUT
# 
PopA = time_series_1[:, :600].T   # (T, cells)
PopB = time_series_2[:, :600].T

CellA, CellB = PopA.shape[1], PopB.shape[1]
dfA = pd.DataFrame(PopA, columns=[f"A_cell_{i}" for i in range(CellA)])
dfB = pd.DataFrame(PopB, columns=[f"B_cell_{i}" for i in range(CellB)])

# 
# TIME-DELAY EMBEDDING
# 
def time_delay_embedding(x, m, tau):
    N = len(x) - (m - 1) * tau
    return np.array([x[i:i + m * tau:tau] for i in range(N)])

def population_embedding(df, m, tau):
    embeddings = [time_delay_embedding(df[col].values, m, tau) for col in df.columns]
    min_len = min(e.shape[0] for e in embeddings)
    return np.hstack([e[:min_len] for e in embeddings])

m, tau = 2, 1
XA = population_embedding(dfA, m, tau)
XB = population_embedding(dfB, m, tau)

T = min(len(XA), len(XB))
XA, XB = XA[:T], XB[:T]

#PCA PER POPULATION (feature space, not time space)
#
n_components = min(10, XA.shape[1], XB.shape[1])

pca_A = PCA(n_components=n_components)
pca_B = PCA(n_components=n_components)

ZA = pca_A.fit_transform(XA)   # (T, n_components) — A's trajectory in its own latent space
ZB = pca_B.fit_transform(XB)   # (T, n_components) — B's trajectory in its own latent space

# 
# PROCRUSTES ALIGNMENT
# optimal orthogonal rotation R to bring ZB into ZA's coordinate frame

R, _ = orthogonal_procrustes(ZB, ZA)
ZB_aligned = ZB @ R   # ZA and ZB_aligned now live in the same coordinate frame

time_axis = np.arange(T)

# 
#  GEOMETRIC SYNCHRONY METRIC: pointwise trajectory distance
# d_t = ||ZA(t) - ZB_aligned(t)||  — how far apart are the two population
# states at each time point in the shared aligned space

d_t = np.linalg.norm(ZA - ZB_aligned, axis=1)   # (T,)

# 
#  DOMINANCE: principal subspace angles
# Measures how much the two populations share geometric structure
# Small angle → synchrony / shared geometry
# Large angle → dominance / divergence
# 
def subspace_angles(A, B):
    QA, _ = np.linalg.qr(A)
    QB, _ = np.linalg.qr(B)
    _, sigma, _ = svd(QA.T @ QB, full_matrices=False)
    sigma = np.clip(sigma, -1, 1)
    return np.degrees(np.arccos(sigma))   # principal angles in degrees

angles = subspace_angles(ZA, ZB_aligned)
mean_angle = np.mean(angles)

# 
# SLIDING WINDOW SUBSPACE ALIGNMENT SCORE
# Captures LOCAL geometric synchrony over time — more informative than
# pointwise distance alone; reveals when synchrony transitions happen
# 
window = 20  ## to change 
sync_score = []
sync_times = []

for t in range(window, T - window):
    seg_A = ZA[t - window:t + window]
    seg_B = ZB_aligned[t - window:t + window]
    QA, _ = np.linalg.qr(seg_A)
    QB, _ = np.linalg.qr(seg_B)
    _, sigma, _ = svd(QA.T @ QB, full_matrices=False)
    sync_score.append(np.mean(np.clip(sigma, -1, 1)))
    sync_times.append(t)

sync_score = np.array(sync_score)
sync_times = np.array(sync_times)

 
######## PLOTS
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

colors = {'A': 'royalblue', 'B': 'tomato', 'sync': 'mediumseagreen',
          'div': 'darkorange', 'angle': 'mediumpurple'}

# Shared phase space after Procrustes alignment 
ax1 = fig.add_subplot(gs[0, :])   # full width
ax1.plot(ZA[:, 0], ZA[:, 1], 'o', color=colors['A'], alpha=0.35, ms=3, label='Class A')
ax1.plot(ZB_aligned[:, 0], ZB_aligned[:, 1], 'o', color=colors['B'], alpha=0.35, ms=3, label='Class B (Procrustes aligned)')
ax1.scatter(*ZA[0, :2],  c='green', s=250, marker='*', zorder=5, label='Start')
ax1.scatter(*ZA[-1, :2], c='black', s=250, marker='X', zorder=5, label='End')
ax1.set(xlabel="Latent dim 1", ylabel="Latent dim 2",
        title="Shared Population Phase Space — Procrustes Aligned (Dim 1 vs 2)")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.4)

# Geometric distance d_t over time 
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_axis, d_t, lw=1.2, color='purple', alpha=0.85)
ax2.axhline(d_t.mean(), ls='--', color='gray', lw=1, label=f'Mean = {d_t.mean():.2f}')
ax2.fill_between(time_axis, d_t, d_t.mean(),
                 where=(d_t < d_t.mean()), alpha=0.25, color=colors['sync'], label='High sync')
ax2.fill_between(time_axis, d_t, d_t.mean(),
                 where=(d_t >= d_t.mean()), alpha=0.25, color=colors['div'],  label='Low sync')
ax2.set(xlabel="Time", ylabel="Δ(t)  ||ZA − ZB_aligned||",
        title="Pointwise Geometric Distance Δ(t)")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)

#  Sliding window subspace alignment score 
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(sync_times, sync_score, lw=1.5, color=colors['sync'])
ax3.axhline(sync_score.mean(), ls='--', color='gray', lw=1,
            label=f'Mean = {sync_score.mean():.3f}')
ax3.fill_between(sync_times, sync_score, sync_score.mean(),
                 where=(sync_score > sync_score.mean()), alpha=0.25,
                 color=colors['sync'], label='High alignment')
ax3.fill_between(sync_times, sync_score, sync_score.mean(),
                 where=(sync_score <= sync_score.mean()), alpha=0.25,
                 color=colors['div'], label='Low alignment')
ax3.set(xlabel="Time", ylabel="Mean cos(principal angle)",
        title=f"Sliding Window Subspace Alignment (window={window})")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.4)

# Principal subspace angles (global dominance) 
ax4 = fig.add_subplot(gs[2, 0])
ax4.bar(range(len(angles)), angles, color=colors['angle'], alpha=0.8, edgecolor='white')
ax4.axhline(45, ls='--', color='gray', lw=1, label='45° threshold (random)')
ax4.axhline(mean_angle, ls=':', color='black', lw=1.2,
            label=f'Mean = {mean_angle:.1f}°')
ax4.set(xlabel="Principal angle index", ylabel="Angle (degrees)",
        title="Principal Subspace Angles — Global Geometric Dominance\n"
              "(< 45° → shared structure, > 45° → divergent geometry)")
ax4.legend(fontsize=8); ax4.grid(True, axis='y', alpha=0.4)

#  KDE — when does synchrony occur 
ax5 = fig.add_subplot(gs[2, 1])

# High-sync moments: top quartile of alignment score
high_sync_threshold = np.percentile(sync_score, 75)
low_sync_threshold  = np.percentile(sync_score, 25)
high_sync_times = sync_times[sync_score >= high_sync_threshold]
low_sync_times  = sync_times[sync_score <= low_sync_threshold]

for times, color, label in [
    (high_sync_times, colors['sync'], 'High sync (top 25%)'),
    (low_sync_times,  colors['div'],  'Low sync / dominance (bottom 25%)')
]:
    if len(times) > 1:
        kde = gaussian_kde(times, bw_method=0.2)
        t_range = np.linspace(0, T, 400)
        ax5.plot(t_range, kde(t_range), lw=2, color=color, label=label)
        ax5.fill_between(t_range, kde(t_range), alpha=0.2, color=color)

ax5.set(xlabel="Time", ylabel="Density",
        title="Temporal KDE — When Does Sync / Dominance Occur?")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.4)

plt.suptitle("Geometric Synchrony in Population Dynamics", fontsize=14, y=1.01, fontweight='bold')
plt.show()
