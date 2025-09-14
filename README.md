# Bayesian Deep Clustering Network for SME Credit Risk Assessment

## 📌 Overview
This project implements a **Bayesian Deep Clustering Network (BDCN)** for **SME credit risk assessment** in Bangladesh, addressing the country’s rising Non-Performing Loan (NPL) crisis.  
Our approach uses **non-deterministic unsupervised deep learning** to segment SMEs into risk clusters while **quantifying prediction uncertainty** — a critical feature for financial decision-making.

---

## 🎯 Motivation
Bangladesh’s NPLs have reached **24.13% of all outstanding loans**, significantly affecting SMEs that contribute **25% to GDP** but face a **$2.8B financing gap**.  
Traditional deterministic models (Logistic Regression, Decision Trees, Neural Networks) lack the ability to quantify uncertainty, leading to poor credit risk assessment and reduced financial inclusion.

---

## 🚀 Research Objectives
1. **Design** a non-deterministic unsupervised neural network for SME risk profiling.
2. **Implement** uncertainty quantification in credit risk assessment.
3. **Compare** performance against traditional models (VAE + K-means).
4. **Demonstrate** business impact through improved risk-based loan approval.
5. **Provide** actionable recommendations for NPL reduction in Bangladesh.

---

## 🧠 Methodology

### Problem Formulation
- **Input:** 14 financial & operational SME features  
- **Output:** Cluster assignment (Low, Medium, High risk)  
- **Objective:** Learn natural risk groupings & quantify confidence in predictions  

### Model Architecture
- **Bayesian Encoder:** Dense variational layers → latent space  
- **Clustering Layer:** Learnable cluster centers with soft assignment  
- **Decoder:** Reconstruction of input features  
- **Loss Function:**  
L = L_recon + λ₁L_cluster + λ₂L_KL

- Reconstruction Loss (MSE)
- Clustering Loss
- KL Divergence for Bayesian regularization  

### Uncertainty Estimation
- **Monte Carlo Dropout** during inference
- **Predictive Entropy** & **Epistemic Uncertainty** analysis

---

## 🧪 Experimental Setup

- **Dataset:** Synthetic dataset (2,000 SMEs, 14 features)
- **Framework:** TensorFlow 2.x + TensorFlow Probability
- **Hardware:** GPU-accelerated training (Colab Pro)
- **Training:**  
- Epochs: 150  
- Batch size: 64  
- Optimizer: Adam (LR=0.001)  
- Early stopping based on validation loss  

---

## 📊 Results

| Model                   | Silhouette Score | Adjusted Rand Index | Normalized MI |
|------------------------|-----------------|---------------------|---------------|
| **Bayesian Deep Clustering** | **0.687**         | **0.724**           | **0.698**     |
| VAE + K-means          | 0.542           | 0.631               | 0.609         |
| **Improvement**        | **+26.7%**      | **+14.7%**          | **+14.6%**    |

- **High Confidence Predictions (>80%)**: 68.3%  
- **Uncertain Cases (Manual Review Required)**: 12.3%  
- **Projected NPL Reduction:** ~3.5 percentage points (≈ $500M unlocked financing)

---

## 📈 Business Impact
- **Risk-Based Lending:** Automate low-risk approvals
- **Human-in-the-loop Review:** Focus on uncertain/high-risk cases
- **Financial Inclusion:** Better credit access for deserving SMEs
- **Economic Growth:** Strengthened SME contribution to GDP

---

## 🔮 Future Work
- Real-world validation with actual SME financial data  
- Integration of **Explainable AI** techniques for model interpretability  
- Support for **multi-modal data** (e.g., text, alternative data sources)  
- **Federated Learning** for privacy-preserving credit risk modeling  
- Real-time credit decision systems with streaming data  

---

## 📚 References
- Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR.  
- Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. ICML.  

---

## 👤 Author
**Md. Salman Pasha**  
Student ID: 24141081  
Course: Neural Networks  
Assignment: Non-Deterministic Unsupervised Neural Network Model  
Date: September 14, 2025

