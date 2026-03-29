# 🧫 Population Dynamics in Spontaneous Calcium Activity  

**Spoiler:** This is not about brain geometry — it’s about rhythmic synchrony in the pituitary gland 💥  

---

## 🧠 Overview  
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

More broadly, the pipeline is designed to move from **general signal features to biologically meaningful insights**:

### 1. Signal Decomposition  
- Separation of **periodic** and **aperiodic** components  
- Spectral analysis to classify oscillatory patterns  

### 2. Feature Extraction  
- **Nonlinear metrics**: entropy, directional information  
- **Linear metrics**: Spearman correlation for connectivity inference  

### 3. Population-Level Analysis  
- Exploration of the **geometry of the activity landscape**  
- Characterization of **synchrony patterns**  

This approach revealed a **transient bistable state** in heterotypic interactions, suggesting dynamics consistent with a **Hopf oscillator**.  

In this regime, one class of signals may act as a *leader*, potentially initiating and driving secretory pulses.

---

## 🤖 Modeling  
To validate these findings, we implemented a **null model generator** based on a **low-rank recurrent neural network (RNN)**.  
This allows us to test whether the observed dynamics arise from structured interactions or can be explained by simpler generative processes.

---

## 🚀 How to Use This Repository  
You can use this framework to explore calcium dynamics (or similar time series data) in your own systems.  

💡 **Suggested approach:**
1. Start with simple feature extraction  
2. Gradually incorporate spectral and nonlinear analyses  
3. Explore connectivity and population geometry  
4. Interpret results in the context of your biological system  

---

## 🌱 Final Thoughts  
This repository is intended as both a **research companion** and a **starting point** for exploring how spontaneous activity encodes meaningful biological dynamics.  

Feel free to adapt, extend, and apply these tools to your own questions!

---

## 📬 Contact  
For questions, collaborations, or feedback:  
**anaaquiles@ciencias.unam.mx**
