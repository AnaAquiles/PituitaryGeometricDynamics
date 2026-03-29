# 🧫 Population Dynamics in Spontaneous Calcium Activity  

**Spoiler:** This is not about brain geometry — it’s about rhythmic synchrony in the pituitary gland 💥  

---

## 👀 Overview  
This repository contains all the code used in our study on *population geometry coupling* (preprint coming soon). In this work, we identify **two classes of spontaneous calcium activity** by evaluating two types of structural interactions:

- **Homotypic interactions**: within the same cell group  
- **Heterotypic interactions**: across all cells exhibiting spontaneous calcium activity  

Our goal is to better understand how collective dynamics emerge from these interactions and how they relate to functional behavior.

---

## 🧪 Data  
All time series data come from **in vitro calcium imaging experiments**.  
For full experimental details, please refer to the Methods section of the manuscript (coming soon 📄).

---

## 🔬 Methodological Framework  

This repository provides a complete framework to explore the **intrinsic dynamics of spontaneous calcium activity**, particularly in pituitary cells that can secrete hormones *without hypothalamic input*.  

The pipeline is structured as follows:


### 🧹 0. Preprocessing  
Initial conditioning and preparation of calcium signals.

- `preprocessing.py`



### 🌊 1. Signal Decomposition  
Separation of **periodic** and **aperiodic** components, and spectral characterization.

- `AperiodicFit.py` → extraction of aperiodic components  
- `PAC.py` → phase-amplitude coupling analysis  
- `SignalClasses.py` → classification of oscillatory regimes  



### 📊 2. Feature Extraction  

#### 🔹 Aperiodic & Entropy-Based Metrics  
- `AperiodicEntropyCorr.py` → entropy and correlation structure  
- `ContingencyAperiodicEntropy.py` → contingency analysis between entropy features  

#### 🔹 Connectivity & Correlation  
- `AperiodicClusterAdjacency.py` → adjacency structure from aperiodic features  
- `SurrogateCorrelationSynchrony.py` → surrogate-based validation of synchrony  



### 🧭 3. Population-Level & Geometric Analysis  

Exploration of the **geometry of the activity landscape** and synchrony structure.

- `GeometricSyncrony.py` → geometric description of synchrony  
- `QuasipotentialDominance.py` → dominance landscape / quasipotential analysis  
- `PhasePortrait-elevator.py` → dynamical phase space exploration  



### ⚖️ 4. Dynamical Regimes & Bistability  

Characterization of emergent collective states.

- `BistabilityAnalysis.py` → detection of bistable regimes  

This analysis revealed a **transient bistable state** in heterotypic interactions, suggesting dynamics consistent with a **Hopf oscillator**, where one class of signals may act as a *leader* driving secretory pulses.



## 🤖 Modeling  

To validate these findings, we implemented a **null model generator**:

- `rnnNullModel.py` → low-rank recurrent neural network (RNN)  

This allows testing whether observed dynamics arise from structured biological interactions or simpler generative mechanisms.

---

## ⚙️ Environment  

- `enviroment.yml` → reproducible computational environment  

---

## 🚀 How to Use This Repository  

💡 **Suggested workflow:**
1. Start with preprocessing  
2. Perform signal decomposition  
3. Extract features (entropy, correlations, coupling)  
4. Analyze synchrony and geometry  
5. Explore dynamical regimes and validate with null models  

---

## 🌱 Final Thoughts  
This repository is intended as both a **research companion** and a **starting point** for exploring how spontaneous activity encodes meaningful biological dynamics.  

Feel free to adapt, extend, and apply these tools to your own questions!

---

## 📬 Contact  
For questions, collaborations, or feedback:  
**anaaquiles@ciencias.unam.mx**
