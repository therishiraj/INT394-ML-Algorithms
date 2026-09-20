# 📘 Machine Learning — Complete B.Tech Notes

Unit-wise notes with definitions, derivations, every key formula, and fully solved numericals.

> **Math rendering:** These notes use GitHub's native LaTeX support (`$...$` inline and fenced `math` code blocks). They render correctly on github.com. For local viewing use VS Code (with *Markdown+Math*) or Typora.

---

## 📑 Table of Contents

| Unit | Topics | File |
|:---:|---|---|
| **I** | Foundations of ML, Types of Learning, Challenges, Statistical Learning Framework, ERM, Inductive Bias, PAC Learning | [Unit-1-Foundations.md](Unit-1-Foundations.md) |
| **II** | Classification, Decision Boundaries, Linear Classifiers, OvA / OvO, Bayes Theorem, Naïve Bayes, Bayesian Decision Theory | [Unit-2-Classification.md](Unit-2-Classification.md) |
| **III** | KNN, Decision Trees (ID3, C4.5, CART), Pruning, Overfitting, Ensembles (Bagging, Boosting, Stacking, Voting) | [Unit-3-NonParametric-and-Ensembles.md](Unit-3-NonParametric-and-Ensembles.md) |
| **IV** | Regression (Linear, Non-linear, Loss Functions, Non-parametric), Clustering (Distances, K-Means, Hierarchical, Validation) | [Unit-4-Regression-and-Clustering.md](Unit-4-Regression-and-Clustering.md) |
| **V** | Reinforcement Learning, MDPs, Value Functions, Bellman Equations, Q-Learning, TD Learning | [Unit-5-Reinforcement-Learning.md](Unit-5-Reinforcement-Learning.md) |
| **VI** | VC Dimension, Rademacher Complexity, Bias–Variance, Regularization, Cross-Validation, SRM, Gradient Descent & Variants, Convergence | [Unit-6-Model-Complexity-and-Optimization.md](Unit-6-Model-Complexity-and-Optimization.md) |

---

## 🧭 How to Use These Notes

1. Read the **concept** section first, then work through each **solved numerical** with pen and paper.
2. Every unit ends with a **Formula Sheet** for quick revision before exams.
3. **Exam tips** (marked 💡) highlight frequently asked questions and common mistakes.

## 📐 Notation Used Throughout

| Symbol | Meaning |
|---|---|
| $\mathcal{X}$, $\mathcal{Y}$ | Input (feature) space, output (label) space |
| $x_i$, $y_i$ | $i$-th training example and its label |
| $m$ or $n$ | Number of training examples |
| $d$ | Number of features (dimensions) |
| $\mathcal{D}$ | Unknown data distribution |
| $S$ | Training sample |
| $h$, $\mathcal{H}$ | A hypothesis, the hypothesis class |
| $L_{\mathcal{D}}(h)$, $L_S(h)$ | True risk, empirical risk |
| $\mathbf{w}$, $b$ | Weight vector, bias |
| $\eta$ (or $\alpha$) | Learning rate |
| $\lambda$ | Regularization strength |
| $\gamma$ | Discount factor (RL) |

---

⭐ If these notes help you, star the repository and share with classmates!
