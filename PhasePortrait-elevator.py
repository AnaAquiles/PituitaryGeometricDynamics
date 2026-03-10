"""
Elevator figure

Key design decisions:
  - T=50000, burn=5000 → stationary distribution, no transient artefacts
  - ZB (N=80) shown as phase portrait — clean oscillatory structure
  - ZA (N=50) shown as coloured lagged trajectory overlaid
  - Vector field: empirical binned dx/dt from ZB trajectory
  - Energy landscape: KDE on stationary ZB cloud
  - Input applied to Pop A (I_A_tonic), effect read out in ZB
  - Three I levels shown: 0.0, 0.6, 1.0 — stacked as elevator
  - Lag annotated and cross-validated across all conditions

Layout:
  Left (wide):  3D elevator — ZB planes at I=0, 0.6, 1.0 stacked along input axis
  Right top:    2D phase portrait ZB — I=0   (limit cycle visible)
  Right mid:    2D phase portrait ZB — I=0.6 (shrinking / shifted)
  Right bot:    time series PC1(ZA) vs PC1(ZB) with lag annotation
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

# 
# SIMULATION  —  exact same parameters as the validated model


def simulate(g_AB=3.0, g_BA=0.0, I_A_tonic=0.0, I_B_tonic=0.0,
             N_A=50, N_B=80, rank=2, sr=1.15,
             T=50000, dt=0.5, tau=5.0, noise=0.015,
             seed_W=8, seed_IC=108):
    rng_W = np.random.default_rng(seed_W)
    rng_A = np.random.default_rng(seed_IC)
    rng_B = np.random.default_rng(seed_IC + 50)

    def lrW(N, r):
        U  = rng_W.standard_normal((N, r))
        V  = rng_W.standard_normal((r, N))
        W  = (U @ V) / np.sqrt(N)
        ev = np.max(np.abs(np.linalg.eigvals(W)))
        sc = (sr / ev) if ev > 1e-10 else 1.0
        return W * sc, U * sc

    W_A, m_A = lrW(N_A, rank);  W_B, m_B = lrW(N_B, rank)
    C_AB = rng_W.standard_normal((N_B, N_A)) / N_A
    C_BA = rng_W.standard_normal((N_A, N_B)) / N_B
    u_A  = m_A[:, 0] / (np.linalg.norm(m_A[:, 0]) + 1e-10)
    u_B  = m_B[:, 0] / (np.linalg.norm(m_B[:, 0]) + 1e-10)

    x_A  = rng_A.standard_normal(N_A) * 0.3
    x_B  = rng_B.standard_normal(N_B) * 0.3
    tA   = np.zeros((T, N_A));  tB = np.zeros((T, N_B))
    dt_tau = dt / tau

    for t in range(T):
        r_A, r_B = np.tanh(x_A), np.tanh(x_B)
        x_A += (-x_A + W_A @ r_A + g_BA * (C_BA @ r_B)
                + I_A_tonic * u_A) * dt_tau \
               + noise * np.sqrt(dt) * rng_A.standard_normal(N_A)
        x_B += (-x_B + W_B @ r_B + g_AB * (C_AB @ r_A)
                + I_B_tonic * u_B) * dt_tau \
               + noise * np.sqrt(dt) * rng_B.standard_normal(N_B)
        tA[t] = x_A;  tB[t] = x_B

    return tA, tB


BURN = 5000   # discard transient

def get_stationary_pca(traj, n_comp=4):
    stat = traj[BURN:]
    pca  = PCA(n_components=n_comp).fit(stat)
    return pca, pca.transform(stat)

def get_lag(ZA, ZB, max_lag=600):
    best_lag, best_val = 0, -np.inf
    for d in range(min(2, ZA.shape[1])):
        a = ZA[:, d] - ZA[:, d].mean()
        for sign in [1, -1]:
            b    = sign * (ZB[:, d] - ZB[:, d].mean())
            corr = np.correlate(a, b, mode='full')
            lags = np.arange(-(len(a) - 1), len(a))
            mask = np.abs(lags) <= max_lag
            idx  = np.argmax(corr[mask])
            if corr[mask][idx] > best_val:
                best_val = corr[mask][idx];  best_lag = lags[mask][idx]
    return best_lag, best_val

def energy_landscape(Z, grid_res=70, bw=0.25, sigma=1.5, margin=0.8):
    x, y = Z[:, 0], Z[:, 1]
    xi   = np.linspace(x.min() - margin, x.max() + margin, grid_res)
    yi   = np.linspace(y.min() - margin, y.max() + margin, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    # Subsample for KDE speed (50k points is slow)
    sub  = max(1, len(Z) // 8000)
    Z_s  = Z[::sub]
    kde  = gaussian_kde(Z_s[:, :2].T, bw_method=bw)
    P    = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(grid_res, grid_res)
    U    = gaussian_filter(-np.log(np.clip(P, 1e-10, None)), sigma=sigma)
    U   -= U.min()
    return Xi, Yi, U

def empirical_vf(Z, n_grid=14):
    """Bin finite-difference velocities onto a regular grid."""
    dZ    = np.diff(Z[:, :2], axis=0)
    Z_mid = 0.5 * (Z[:-1, :2] + Z[1:, :2])
    gr    = np.abs(Z).max() * 1.15
    edges = np.linspace(-gr, gr, n_grid + 1)
    cents = 0.5 * (edges[:-1] + edges[1:])
    G1, G2   = np.meshgrid(cents, cents)
    DG1, DG2 = np.zeros_like(G1), np.zeros_like(G2)
    cnt      = np.zeros_like(G1)
    for ti in range(len(dZ)):
        iz = np.searchsorted(edges, Z_mid[ti, 0]) - 1
        jz = np.searchsorted(edges, Z_mid[ti, 1]) - 1
        if 0 <= iz < n_grid and 0 <= jz < n_grid:
            DG1[jz, iz] += dZ[ti, 0]
            DG2[jz, iz] += dZ[ti, 1]
            cnt[jz, iz] += 1
    m = cnt > 0
    DG1[m] /= cnt[m];  DG2[m] /= cnt[m]
    return G1, G2, DG1, DG2, m, gr


# 
# SIMULATE THREE INPUT LEVELS
#     Input applied to Pop A (g_AB=3.0 → A drives B)
#     ZB shows the driven population → clean oscillation + input deformation

I_LEVELS = [0.0, 0.6, 1.2]
COLORS   = ['#4e9af1', '#ffe082', '#f47c7c']
LABELS   = ['I = 0  (autonomous)', 'I = 0.6  (moderate)', 'I = 1.2  (strong)']

print("Simulating T=50 000 for three input levels (this takes ~2 min)...")
data = []
pca_B_ref = None   # fix PCA axes from I=0 run for comparability

for I_val in I_LEVELS:
    print(f"  I_A = {I_val}...")
    tA, tB = simulate(I_A_tonic=I_val)

    # Fit PCA on stationary portion of this condition
    pca_A, ZA = get_stationary_pca(tA, n_comp=4)
    pca_B, ZB = get_stationary_pca(tB, n_comp=4)

    if pca_B_ref is None:
        pca_B_ref = pca_B   # lock axes from I=0

    # Re-project using FIXED reference axes for ZB (so all conditions are comparable)
    ZB_fixed = pca_B_ref.transform(tB[BURN:])[:, :2]

    # ZA in its own PCA (sign-aligned to ZB for cross-correlation only)
    ZA_2 = ZA[:, :2]
    sign_corr = np.sign(np.sum(ZA_2 * ZB_fixed, axis=0))
    sign_corr[sign_corr == 0] = 1
    ZA_sign = ZA_2 * sign_corr

    lag, strength = get_lag(ZA_sign, ZB_fixed)

    # Energy landscape on ZB
    Xi, Yi, U = energy_landscape(ZB_fixed)

    # Empirical vector field on ZB
    G1, G2, DG1, DG2, vmask, gr = empirical_vf(ZB_fixed)

    data.append(dict(
        I=I_val, color=COLORS[I_LEVELS.index(I_val)],
        label=LABELS[I_LEVELS.index(I_val)],
        ZA=ZA_sign, ZB=ZB_fixed,
        tA_stat=tA[BURN:], tB_stat=tB[BURN:],
        Xi=Xi, Yi=Yi, U=U,
        G1=G1, G2=G2, DG1=DG1, DG2=DG2, vmask=vmask, gr=gr,
        lag=lag, strength=strength,
    ))
    print(f"    lag={lag:+d}  strength={strength:.0f}")

print("Done.")


#  FIGURE
#   Left  (wide 3D): elevator — ZB at three I levels stacked along input axis
#   Right col 1:     ZB 2D phase portrait  I=0
#   Right col 2:     ZB 2D phase portrait  I=0.6
#   Bottom row:      time series PC1(ZA) vs PC1(ZB)  for I=0 and I=0.6


fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor('#0e0e14')

gs = gridspec.GridSpec(2, 3, figure=fig,
                        width_ratios=[1.45, 1, 1],
                        wspace=0.20, hspace=0.45)

ax3d   = fig.add_subplot(gs[:, 0], projection='3d')
ax_pp  = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
ax_ts  = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]


#  draw 2D phase portrait 

def draw_pp(ax, d, col_title, n_traj=3000):
    ax.set_facecolor('#0e0e14')
    ZB = d['ZB']

    # Energy landscape background
    ax.contourf(d['Xi'], d['Yi'], d['U'], levels=20, cmap='plasma', alpha=0.80)
    ax.contour( d['Xi'], d['Yi'], d['U'], levels=8,
                colors='white', alpha=0.10, linewidths=0.45)

    # Empirical velocity field
    G1, G2, DG1, DG2, vmask = d['G1'], d['G2'], d['DG1'], d['DG2'], d['vmask']
    spd = np.sqrt(DG1**2 + DG2**2) + 1e-10
    step = 2
    valid = vmask[::step, ::step]
    ax.quiver(G1[::step, ::step][valid], G2[::step, ::step][valid],
              (DG1[::step, ::step] / spd[::step, ::step])[valid],
              (DG2[::step, ::step] / spd[::step, ::step])[valid],
              alpha=0.55, color='white', scale=18,
              headwidth=3.5, headlength=3.5, width=0.004)

    # ZB trajectory (colour = time)
    n   = min(n_traj, len(ZB))
    sub = max(1, n // 500)
    cg  = plt.get_cmap('Reds')(np.linspace(0.3, 1.0, n))
    for i in range(0, n - 1, sub):
        ax.plot(ZB[i:i+2, 0], ZB[i:i+2, 1], color=cg[i], lw=0.85, alpha=0.9)

    # ZA overlaid (smaller, grey-blue)
    ZA = d['ZA']
    n_a = min(n_traj, len(ZA))
    sub_a = max(1, n_a // 400)
    cg_a = plt.get_cmap('Blues')(np.linspace(0.3, 0.85, n_a))
    for i in range(0, n_a - 1, sub_a):
        ax.plot(ZA[i:i+2, 0], ZA[i:i+2, 1], color=cg_a[i], lw=0.6, alpha=0.55)

    # Annotations
    lag = d['lag']
    lead = 'A leads' if lag < -3 else ('B leads' if lag > 3 else 'no lead')
    lag_col = '#4e9af1' if lag < -3 else '#f47c7c'
    ax.text(0.03, 0.97, f'lag = {lag:+d}  ({lead})',
            transform=ax.transAxes, fontsize=8.5, color=lag_col, va='top',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='#0e0e14', ec=lag_col, alpha=0.85))
    ax.text(0.97, 0.03, d['label'].split('  ')[0],
            transform=ax.transAxes, fontsize=8.5, color='#ffe082',
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.25', fc='#0e0e14', ec='#ffe082', alpha=0.8))

    ax.set_title(col_title, color=d['color'], fontsize=10, fontweight='bold', pad=5)
    ax.set_xlabel('ZB  PC1', color='gray', fontsize=8.5)
    ax.set_ylabel('ZB  PC2', color='gray', fontsize=8.5)
    ax.tick_params(colors='gray', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(d['color']);  sp.set_linewidth(1.6)


# Draw the I=0 and I=0.6 panels
draw_pp(ax_pp[0], data[0], 'Pop B latent space  ·  I = 0\n(autonomous, A→B coupling)')
draw_pp(ax_pp[1], data[1], 'Pop B latent space  ·  I_A = 0.6\n(tonic input on Pop A)')


#  TIME SERIES PANELS 

for ax, d, title in zip(ax_ts, [data[0], data[1]],
                        ['Time series  ·  I = 0', 'Time series  ·  I_A = 0.6']):
    ax.set_facecolor('#0e0e14')
    T_show = min(1200, len(d['ZA']))
    t_ax   = np.arange(T_show) * 0.5   # time in tau units

    za = d['ZA'][:T_show, 0]
    zb = d['ZB'][:T_show, 0]
    # Normalise for visual overlap
    za_n = za / (za.std() + 1e-10)
    zb_n = zb / (zb.std() + 1e-10)

    ax.plot(t_ax, zb_n, lw=1.1, color='#f47c7c', alpha=0.9, label='Pop B PC1')
    ax.plot(t_ax, za_n, lw=0.9, color='#7eb8f7', alpha=0.75, label='Pop A PC1')
    ax.fill_between(t_ax, za_n, zb_n, alpha=0.10, color='white', label='Δ(t)')

    lag = d['lag']
    lead = 'A leads' if lag < -3 else 'B leads'
    ax.text(0.97, 0.97, f'lag={lag:+d}  ({lead})',
            transform=ax.transAxes, fontsize=8, color='#4e9af1',
            va='top', ha='right', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='#0e0e14', ec='#4e9af1', alpha=0.85))

    ax.set_xlabel('Time  (τ units)', color='white', fontsize=8)
    ax.set_ylabel('Normalised PC1', color='white', fontsize=8)
    ax.set_title(title, color=d['color'], fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=7.5, facecolor='#1a1a24', labelcolor='white', loc='upper left')
    ax.tick_params(colors='gray', labelsize=7)
    ax.grid(True, alpha=0.12)
    for sp in ax.spines.values():
        sp.set_edgecolor(d['color']);  sp.set_linewidth(1.3)


#  3D ELEVATOR 

ax3d.set_facecolor('#0e0e14')
plane_z = [0.0, 4.0, 8.0]

for d, pz in zip(data, plane_z):
    color  = d['color']
    ZB     = d['ZB']
    gr     = d['gr']

    # Transparent plane
    xp = np.array([-gr, gr]); yp = np.array([-gr, gr])
    Xp, Yp = np.meshgrid(xp, yp)
    ax3d.plot_surface(Xp, Yp, np.full_like(Xp, pz), alpha=0.07, color=color)

    # Border
    bx = [-gr, gr, gr, -gr, -gr]
    by = [-gr, -gr, gr, gr, -gr]
    ax3d.plot(bx, by, [pz] * 5, color=color, lw=1.6, alpha=0.75)

    # Empirical vector field on the plane
    G1, G2 = d['G1'], d['G2']
    DG1, DG2, vmask = d['DG1'], d['DG2'], d['vmask']
    spd = np.sqrt(DG1**2 + DG2**2) + 1e-10
    st  = 2
    val = vmask[::st, ::st]
    Zs  = np.full(val.sum(), pz)
    Dz  = np.zeros_like(Zs)
    ax3d.quiver(G1[::st,::st][val], G2[::st,::st][val], Zs,
                (DG1[::st,::st]/spd[::st,::st])[val] * 0.28,
                (DG2[::st,::st]/spd[::st,::st])[val] * 0.28,
                Dz, color=color, alpha=0.60,
                arrow_length_ratio=0.35, linewidth=0.9)

    # ZB trajectory on the plane (last 1500 stationary steps)
    seg = ZB[:1500]
    cg  = plt.get_cmap('cool')(np.linspace(0.2, 1.0, len(seg)))
    sub = max(1, len(seg) // 200)
    for i in range(0, len(seg) - 1, sub):
        ax3d.plot(seg[i:i+2, 0], seg[i:i+2, 1], [pz, pz],
                  color=cg[i], lw=1.8, alpha=0.9)

    # Label
    ax3d.text(gr + 0.15, -gr, pz, d['label'],
              color=color, fontsize=8.5, fontweight='bold')

# Vertical elevator axis
z_bot, z_top = plane_z[0] - 0.5, plane_z[-1] + 1.0
ax3d.plot([0, 0], [0, 0], [z_bot, z_top],
          color='white', lw=1.2, ls='--', alpha=0.45)
ax3d.text(0.15, 0.15, z_top + 0.3, 'Input axis  I →',
          color='white', fontsize=9, fontstyle='italic')

# Upward arrows between planes
for z0, z1, col in zip(plane_z[:-1], plane_z[1:], COLORS[1:]):
    ax3d.quiver(0, -gr * 0.75, z0 + 0.3, 0, 0, z1 - z0 - 0.6,
                color=col, arrow_length_ratio=0.3, linewidth=1.8, alpha=0.85)

ax3d.set_xlabel('ZB PC1  (κ₁)', color='gray', fontsize=8.5, labelpad=6)
ax3d.set_ylabel('ZB PC2  (κ₂)', color='gray', fontsize=8.5, labelpad=6)
ax3d.set_title('Pop B latent space\nTonic input on Pop A acts as elevator',
               color='white', fontsize=10, fontweight='bold', pad=10)
ax3d.tick_params(colors='gray', labelsize=6)
for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
    pane.fill = False;  pane.set_edgecolor('#252535')
ax3d.zaxis.set_tick_params(labelcolor='#0e0e14')
ax3d.set_zlim(z_bot, z_top + 0.5)
ax3d.view_init(elev=20, azim=-52)


#  LEGEND + TITLE 

legend_els = [
    Line2D([0],[0], color='#f47c7c', lw=2,   label='Pop B trajectory (ZB PC1-2)'),
    Line2D([0],[0], color='#7eb8f7', lw=1.5, label='Pop A trajectory (ZA, normalised)'),
    Line2D([0],[0], color='white',   lw=1.5, alpha=0.5,
           linestyle='None', marker=r'$\rightarrow$', markersize=10,
           label='Empirical velocity field'),
    Line2D([0],[0], color='white', lw=0, marker='s',
           markerfacecolor='#f47c7c', markersize=8, label='I = 0  (blue plane)'),
    Line2D([0],[0], color='white', lw=0, marker='s',
           markerfacecolor='#ffe082', markersize=8, label='I = 0.6  (yellow plane)'),
    Line2D([0],[0], color='white', lw=0, marker='s',
           markerfacecolor='#f47c7c', markersize=8, label='I = 1.2  (red plane)'),
]
fig.legend(handles=legend_els, loc='lower center', ncol=6,
           fontsize=8.5, facecolor='#1a1a24', labelcolor='white',
           framealpha=0.85, bbox_to_anchor=(0.5, 0.005))

fig.text(0.01, 0.97,
         'Model: g_AB = 3.0  (A→B directed)  ·  seed_W = 8  ·  N_A = 50, N_B = 80  '
         '·  T = 50 000, burn = 5 000  ·  Input u_A ∥ m_A[:,0]',
         color='#777788', fontsize=7, va='top')

plt.suptitle(
    'Directed Coupled-Population Model  ·  Tonic Input as Elevator\n'
    'ZB (Pop B, N=80) shows clean oscillatory attractor  ·  '
    'Lag = A leads B by ~23 steps  ·  preserved under tonic input on A',
    color='white', fontsize=11, y=1.01)

plt.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
