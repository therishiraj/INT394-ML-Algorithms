# Unit II — Classification & Probabilistic Approaches

[⬅ Back to Index](README.md)

## Contents
1. [Overview of Classification](#1-overview-of-classification)
2. [Evaluation Metrics for Classification](#2-evaluation-metrics-for-classification)
3. [Decision Boundaries and Their Properties](#3-decision-boundaries-and-their-properties)
4. [Linear Classifiers](#4-linear-classifiers)
5. [Multi-class Classification Strategies](#5-multi-class-classification-strategies)
6. [Bayes Theorem](#6-bayes-theorem)
7. [Naïve Bayes Classifier](#7-naïve-bayes-classifier)
8. [Bayesian Decision Theory](#8-bayesian-decision-theory)
9. [Formula Sheet](#9-formula-sheet)

---

## 1. Overview of Classification

**Classification** is a supervised learning task where the output $y$ belongs to a finite set of **classes** (categories). The model learns a mapping $h:\mathcal{X}\rightarrow\lbrace C_1, C_2, \dots, C_K\rbrace$.

### 1.1 Types of Classification

| Type | Description | Example |
|---|---|---|
| **Binary** | Two classes | Spam / Not spam, Disease / Healthy |
| **Multi-class** | More than two mutually exclusive classes | Digit recognition (0–9), species of iris |
| **Multi-label** | Each instance can have multiple labels | A movie tagged as both *Action* and *Comedy* |
| **Imbalanced** | One class heavily outnumbers others | Fraud detection (0.1% fraud) |

### 1.2 Classification Workflow

```mermaid
flowchart LR
    A[Collect labelled data] --> B[Preprocess: clean, encode, scale]
    B --> C[Split: train / validation / test]
    C --> D[Train classifier]
    D --> E[Tune hyperparameters]
    E --> F[Evaluate on test set]
    F --> G[Deploy & monitor]
```

### 1.3 Types of Classifiers

- **Discriminative models** learn $P(y \mid x)$ or the boundary directly — Logistic Regression, SVM, Perceptron, Decision Trees.
- **Generative models** learn $P(x \mid y)$ and $P(y)$, then use Bayes' theorem to get $P(y \mid x)$ — Naïve Bayes, Gaussian Discriminant Analysis.

---

## 2. Evaluation Metrics for Classification

### 2.1 Confusion Matrix (binary)

|  | **Predicted Positive** | **Predicted Negative** |
|---|---|---|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |

### 2.2 Metrics

```math
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
```

```math
\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall (Sensitivity, TPR)} = \frac{TP}{TP + FN}
```

```math
\text{Specificity (TNR)} = \frac{TN}{TN + FP} \qquad \text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}
```

```math
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \qquad F_\beta = \frac{(1+\beta^2)\cdot P \cdot R}{\beta^2 P + R}
```

### 📝 Solved Numerical 2.1 — Confusion Matrix Metrics

A classifier on 100 patients gives TP = 30, FN = 10, FP = 5, TN = 55.

```math
\text{Accuracy} = \frac{30 + 55}{100} = 0.85
```

```math
\text{Precision} = \frac{30}{30+5} = 0.857, \qquad \text{Recall} = \frac{30}{30+10} = 0.75
```

```math
\text{Specificity} = \frac{55}{55+5} = 0.917, \qquad F_1 = \frac{2(0.857)(0.75)}{0.857+0.75} = \frac{1.2855}{1.607} = 0.80
```

---

## 3. Decision Boundaries and Their Properties

### 3.1 Definition

A **decision boundary** is the surface in feature space that separates regions assigned to different classes. On the boundary, the classifier is equally inclined to both classes.

For a classifier with discriminant functions $g_1(x), g_2(x)$: predict class 1 if $g_1(x) > g_2(x)$. The boundary is:

```math
g_1(x) = g_2(x) \quad \Longleftrightarrow \quad g(x) = g_1(x) - g_2(x) = 0
```

### 3.2 Linear Decision Boundary (Hyperplane)

```math
g(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b = w_1x_1 + w_2x_2 + \dots + w_dx_d + b = 0
```

- In 2D it is a **line**, in 3D a **plane**, in $d$-D a **hyperplane** of dimension $d-1$.
- $\mathbf{w}$ is **normal (perpendicular)** to the hyperplane — it sets the orientation.
- $b$ shifts the hyperplane away from the origin. Distance of hyperplane from origin $= \lvert b\rvert / \lVert\mathbf{w}\rVert$.
- **Signed distance** of any point $\mathbf{x}$ from the hyperplane:

```math
r = \frac{g(\mathbf{x})}{\lVert \mathbf{w} \rVert} = \frac{\mathbf{w}^T\mathbf{x} + b}{\sqrt{w_1^2 + w_2^2 + \dots + w_d^2}}
```

  $r > 0$ → positive side (class +1), $r < 0$ → negative side (class −1).

**Proof that w is normal:** take two points $\mathbf{x}_A, \mathbf{x}_B$ on the hyperplane. Then $\mathbf{w}^T\mathbf{x}_A + b = 0$ and $\mathbf{w}^T\mathbf{x}_B + b = 0$. Subtracting: $\mathbf{w}^T(\mathbf{x}_A - \mathbf{x}_B) = 0$, so $\mathbf{w}$ is perpendicular to every vector lying in the hyperplane.

### 3.3 Properties of Decision Boundaries

| Property | Explanation |
|---|---|
| **Linear vs non-linear** | Linear models → straight boundaries; KNN, trees, kernel SVM, neural nets → curved/complex boundaries |
| **Complexity** | More flexible boundaries fit training data better but risk overfitting |
| **Margin** | Distance from boundary to nearest training point; larger margin → better generalization (SVM idea) |
| **Smoothness** | Smooth boundaries generalise better than jagged ones |
| **Linear separability** | Data is linearly separable if some hyperplane perfectly separates the classes. XOR is **not** linearly separable |
| **Axis-parallel** | Decision trees create boundaries parallel to feature axes (staircase-like) |
| **Piecewise linear** | 1-NN creates a Voronoi tessellation — piecewise linear boundaries |
| **Quadratic** | Gaussian Bayes classifier with unequal covariances gives quadratic boundaries (ellipses, parabolas, hyperbolas) |

### 3.4 Margin (SVM concept)

For a separating hyperplane scaled so that the closest points satisfy $\lvert\mathbf{w}^T\mathbf{x}+b\rvert = 1$:

```math
\text{Margin} = \frac{2}{\lVert \mathbf{w} \rVert}
```

Maximising the margin is equivalent to:

```math
\min_{\mathbf{w},b}\ \frac{1}{2}\lVert\mathbf{w}\rVert^2 \quad \text{subject to} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1,\ \ i = 1,\dots,m
```

### 📝 Solved Numerical 3.1 — Distance from a Hyperplane

Hyperplane: $3x_1 + 4x_2 - 10 = 0$. Classify the point $\mathbf{x} = (4, 3)$ and find its distance.

```math
g(\mathbf{x}) = 3(4) + 4(3) - 10 = 14 > 0 \Rightarrow \text{Class } +1
```

```math
r = \frac{14}{\sqrt{3^2+4^2}} = \frac{14}{5} = 2.8 \text{ units}
```

Distance of the hyperplane from the origin $= \lvert -10\rvert/5 = 2$.

### 📝 Solved Numerical 3.2 — Margin

If an SVM finds $\mathbf{w} = (3, 4)$, the margin $= 2/\lVert\mathbf{w}\rVert = 2/5 = 0.4$.

---

## 4. Linear Classifiers

A linear classifier predicts:

```math
\hat{y} = \text{sign}(\mathbf{w}^T\mathbf{x} + b) = \begin{cases} +1 & \text{if } \mathbf{w}^T\mathbf{x} + b \geq 0 \\ -1 & \text{otherwise} \end{cases}
```

### 4.1 The Perceptron (Rosenblatt, 1958)

**Model:** $\hat{y} = \text{step}(\mathbf{w}^T\mathbf{x} + b)$.

**Learning rule** (for each misclassified example):

```math
\mathbf{w} \leftarrow \mathbf{w} + \eta\,(y - \hat{y})\,\mathbf{x}, \qquad b \leftarrow b + \eta\,(y - \hat{y})
```

For labels in $\lbrace -1, +1 \rbrace$ the equivalent form is: if $y_i(\mathbf{w}^T\mathbf{x}_i + b) \leq 0$ then $\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i$, $b \leftarrow b + \eta y_i$.

**Perceptron Convergence Theorem:** If data is linearly separable with margin $\gamma$ and $\lVert\mathbf{x}_i\rVert \leq R$, the perceptron makes at most

```math
\left(\frac{R}{\gamma}\right)^2
```

mistakes. If data is **not** linearly separable (e.g., XOR), it never converges.

### 📝 Solved Numerical 4.1 — Perceptron for AND Gate

Train a perceptron for the AND gate. Initial $w_1 = 1, w_2 = 1, b = 0$, $\eta = 0.5$. Output $= 1$ if $z = w_1x_1 + w_2x_2 + b > 0$, else 0.

**Epoch 1**

| $x_1$ | $x_2$ | $z$ | $\hat{y}$ | $y$ | Error $e = y-\hat{y}$ | New $w_1$ | New $w_2$ | New $b$ |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| 0 | 1 | 1 | 1 | 0 | −1 | 1 | 0.5 | −0.5 |
| 1 | 0 | 0.5 | 1 | 0 | −1 | 0.5 | 0.5 | −1 |
| 1 | 1 | 0 | 0 | 1 | +1 | 1 | 1 | −0.5 |

Example of one update (row 2): $w_2 = 1 + 0.5(-1)(1) = 0.5$, $b = 0 + 0.5(-1) = -0.5$.

**Epoch 2**

| $x_1$ | $x_2$ | $z$ | $\hat{y}$ | $y$ | $e$ | $w_1$ | $w_2$ | $b$ |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | −0.5 | 0 | 0 | 0 | 1 | 1 | −0.5 |
| 0 | 1 | 0.5 | 1 | 0 | −1 | 1 | 0.5 | −1 |
| 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.5 | −1 |
| 1 | 1 | 0.5 | 1 | 1 | 0 | 1 | 0.5 | −1 |

**Epoch 3:** $z = -1, -0.5, 0, 0.5$ → outputs $0,0,0,1$ — all correct, no updates. **Converged.**

**Final model:** $w_1 = 1$, $w_2 = 0.5$, $b = -1$. Decision boundary: $x_1 + 0.5x_2 - 1 = 0$.

### 4.2 Logistic Regression

Despite its name, it is a **classification** model. It passes the linear score through the **sigmoid** function to get a probability:

```math
z = \mathbf{w}^T\mathbf{x} + b, \qquad \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad P(y=1 \mid \mathbf{x}) = \sigma(z)
```

**Properties of sigmoid:**

```math
\sigma(0) = 0.5, \quad \sigma(-z) = 1 - \sigma(z), \quad \frac{d\sigma}{dz} = \sigma(z)\,(1 - \sigma(z))
```

**Log-odds (logit) is linear:**

```math
\ln\frac{P(y=1\mid\mathbf{x})}{1 - P(y=1\mid\mathbf{x})} = \mathbf{w}^T\mathbf{x} + b
```

Decision rule: predict 1 if $\sigma(z) \geq 0.5 \Leftrightarrow z \geq 0$, so the boundary $\mathbf{w}^T\mathbf{x}+b=0$ is **linear**.

**Loss — Binary Cross-Entropy (Log Loss):**

```math
J(\mathbf{w}, b) = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i\ln\hat{p}_i + (1-y_i)\ln(1-\hat{p}_i)\right], \qquad \hat{p}_i = \sigma(\mathbf{w}^T\mathbf{x}_i + b)
```

**Gradients** (derived using $\sigma' = \sigma(1-\sigma)$):

```math
\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m}\sum_{i=1}^m(\hat{p}_i - y_i)\,\mathbf{x}_i, \qquad \frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^m(\hat{p}_i - y_i)
```

**Gradient descent update:**

```math
\mathbf{w} \leftarrow \mathbf{w} - \eta\,\frac{\partial J}{\partial \mathbf{w}}, \qquad b \leftarrow b - \eta\,\frac{\partial J}{\partial b}
```

### 📝 Solved Numerical 4.2 — Logistic Regression Prediction and One Update

Given $\mathbf{w} = (0.5, -0.25)$, $b = 0.2$, a student with $\mathbf{x} = (2, 4)$ (hours studied, hours of sleep lost) and true label $y = 1$ (passed). Learning rate $\eta = 0.1$.

**Step 1 — score:**

```math
z = 0.5(2) + (-0.25)(4) + 0.2 = 1 - 1 + 0.2 = 0.2
```

**Step 2 — probability:**

```math
\hat{p} = \sigma(0.2) = \frac{1}{1+e^{-0.2}} = \frac{1}{1 + 0.8187} = 0.5498
```

Prediction: $\hat{p} \geq 0.5$ → class 1 (pass).

**Step 3 — loss:**

```math
J = -\ln(0.5498) = 0.5982
```

**Step 4 — gradients:** $\hat{p} - y = 0.5498 - 1 = -0.4502$

```math
\frac{\partial J}{\partial w_1} = -0.4502 \times 2 = -0.9004, \quad \frac{\partial J}{\partial w_2} = -0.4502 \times 4 = -1.8008, \quad \frac{\partial J}{\partial b} = -0.4502
```

**Step 5 — update:**

```math
w_1 = 0.5 - 0.1(-0.9004) = 0.5900, \quad w_2 = -0.25 - 0.1(-1.8008) = -0.0699, \quad b = 0.2 - 0.1(-0.4502) = 0.2450
```

### 4.3 Other Linear Classifiers

| Classifier | Idea | Loss |
|---|---|---|
| Perceptron | Update on mistakes | $\max(0, -y\,\mathbf{w}^T\mathbf{x})$ |
| Logistic Regression | Probabilistic, sigmoid output | Log loss |
| Linear SVM | Maximum margin | Hinge: $\max(0, 1 - y\,\mathbf{w}^T\mathbf{x})$ |
| LDA (Fisher) | Maximise between-class / within-class variance | $J(\mathbf{w}) = \frac{(\mathbf{w}^T(\mu_1-\mu_2))^2}{\mathbf{w}^T S_W \mathbf{w}}$ |

**Fisher LDA solution:**

```math
\mathbf{w}^{\ast} \propto S_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2), \qquad S_W = \sum_{\mathbf{x}\in C_1}(\mathbf{x}-\boldsymbol{\mu}_1)(\mathbf{x}-\boldsymbol{\mu}_1)^T + \sum_{\mathbf{x}\in C_2}(\mathbf{x}-\boldsymbol{\mu}_2)(\mathbf{x}-\boldsymbol{\mu}_2)^T
```

**Limitation of linear classifiers:** they cannot solve non-linearly-separable problems like XOR unless features are transformed (e.g., add $x_1x_2$) or kernels are used.

---

## 5. Multi-class Classification Strategies

Many classifiers (Perceptron, SVM, Logistic Regression) are inherently binary. To handle $K > 2$ classes we decompose the problem.

### 5.1 One-vs-All (One-vs-Rest, OvA / OvR)

- Train **$K$ binary classifiers**. Classifier $k$ treats class $k$ as positive and all other classes as negative.
- Prediction: pick the class whose classifier gives the highest score/confidence:

```math
\hat{y} = \arg\max_{k \in \lbrace 1,\dots,K\rbrace} f_k(\mathbf{x})
```

**Pros:** only $K$ classifiers; simple. **Cons:** each classifier sees **imbalanced** data (1 class vs $K-1$ classes); scores from different classifiers may not be calibrated; ambiguous regions possible.

### 5.2 One-vs-One (OvO)

- Train one binary classifier for **every pair** of classes:

```math
\text{Number of classifiers} = \binom{K}{2} = \frac{K(K-1)}{2}
```

- Prediction: each classifier votes; the class with the **most votes** wins (majority voting).

**Pros:** each classifier trains on a smaller, balanced subset (only 2 classes) — good for algorithms that scale badly with data size (e.g., kernel SVM). **Cons:** number of classifiers grows quadratically; ties in voting possible.

### 5.3 Comparison

| Aspect | One-vs-All | One-vs-One |
|---|---|---|
| Number of classifiers | $K$ | $K(K-1)/2$ |
| Training data per classifier | All $m$ examples | Only examples of 2 classes (≈ $2m/K$) |
| Class imbalance | Yes | No (roughly balanced) |
| Prediction | Highest score | Majority vote |
| Used by default in | Logistic Regression (sklearn) | SVC (sklearn) |

### 5.4 Native Multi-class: Softmax Regression

```math
P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T\mathbf{x}}}, \qquad J = -\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K y_{ik}\ln\hat{p}_{ik}
```

### 📝 Solved Numerical 5.1 — Number of Classifiers

For $K = 4$ classes: OvA needs $4$ classifiers; OvO needs $\frac{4 \times 3}{2} = 6$ classifiers.
For $K = 10$ (digits): OvA needs $10$; OvO needs $\frac{10 \times 9}{2} = 45$.

### 📝 Solved Numerical 5.2 — OvA Prediction

Three OvA classifiers output confidence scores for a test point: $f_{\text{Cat}} = 0.62$, $f_{\text{Dog}} = 0.81$, $f_{\text{Bird}} = 0.15$.

```math
\hat{y} = \arg\max(0.62, 0.81, 0.15) = \text{Dog}
```

### 📝 Solved Numerical 5.3 — OvO Voting

Classes A, B, C, D (6 pairwise classifiers). Outputs for a test point:

| Classifier | A vs B | A vs C | A vs D | B vs C | B vs D | C vs D |
|---|---|---|---|---|---|---|
| Winner | A | C | A | B | D | C |

Votes: A = 2, B = 1, C = 2, D = 1. **Tie between A and C** → break the tie using the A-vs-C classifier (C won) or by summed confidence. **Prediction: C.**

### 📝 Solved Numerical 5.4 — Softmax

Scores for 3 classes: $z = (2.0, 1.0, 0.1)$.

```math
e^{2.0} = 7.389, \quad e^{1.0} = 2.718, \quad e^{0.1} = 1.105, \quad \text{Sum} = 11.212
```

```math
P = \left(\frac{7.389}{11.212}, \frac{2.718}{11.212}, \frac{1.105}{11.212}\right) = (0.659, 0.242, 0.099)
```

Prediction: class 1.

---

## 6. Bayes Theorem

### 6.1 Statement

```math
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}
```

In classification language (class $C_k$, features $\mathbf{x}$):

```math
\underbrace{P(C_k \mid \mathbf{x})}_{\text{posterior}} = \frac{\overbrace{P(\mathbf{x} \mid C_k)}^{\text{likelihood}}\ \overbrace{P(C_k)}^{\text{prior}}}{\underbrace{P(\mathbf{x})}_{\text{evidence}}}
```

**Evidence (total probability theorem):**

```math
P(\mathbf{x}) = \sum_{j=1}^{K} P(\mathbf{x} \mid C_j)\,P(C_j)
```

### 6.2 Derivation

From the definition of conditional probability:

```math
P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)
```

Dividing both sides by $P(B)$ gives Bayes' theorem.

### 6.3 MAP and ML Decisions

**Maximum A Posteriori (MAP):**

```math
\hat{y}_{MAP} = \arg\max_k P(C_k \mid \mathbf{x}) = \arg\max_k P(\mathbf{x} \mid C_k)\,P(C_k)
```

($P(\mathbf{x})$ is the same for all classes, so it can be dropped.)

**Maximum Likelihood (ML):** if all priors are equal,

```math
\hat{y}_{ML} = \arg\max_k P(\mathbf{x} \mid C_k)
```

### 📝 Solved Numerical 6.1 — Medical Test (classic)

A disease affects 1% of the population. A test detects it correctly 99% of the time (sensitivity), but gives a false positive 5% of the time. A person tests positive. What is the probability they actually have the disease?

**Given:** $P(D) = 0.01$, $P(\neg D) = 0.99$, $P(+ \mid D) = 0.99$, $P(+ \mid \neg D) = 0.05$.

**Evidence:**

```math
P(+) = P(+\mid D)P(D) + P(+\mid\neg D)P(\neg D) = 0.99(0.01) + 0.05(0.99) = 0.0099 + 0.0495 = 0.0594
```

**Posterior:**

```math
P(D \mid +) = \frac{0.0099}{0.0594} = 0.1667
```

**Only 16.7%!** The low prior (rare disease) dominates. This is the **base rate fallacy**.

### 📝 Solved Numerical 6.2 — Factory Machines

Machines M1, M2, M3 produce 50%, 30%, 20% of items with defect rates 2%, 3%, 4%. An item is defective. Which machine most likely made it?

```math
P(\text{Def}) = 0.5(0.02) + 0.3(0.03) + 0.2(0.04) = 0.010 + 0.009 + 0.008 = 0.027
```

```math
P(M1\mid\text{Def}) = \frac{0.010}{0.027} = 0.370, \quad P(M2\mid\text{Def}) = \frac{0.009}{0.027} = 0.333, \quad P(M3\mid\text{Def}) = \frac{0.008}{0.027} = 0.296
```

**Answer:** M1 (MAP decision).

---

## 7. Naïve Bayes Classifier

### 7.1 The Naïve Assumption

Computing $P(x_1, x_2, \dots, x_d \mid C_k)$ directly needs exponentially many parameters. Naïve Bayes assumes the features are **conditionally independent given the class**:

```math
P(x_1, x_2, \dots, x_d \mid C_k) = \prod_{j=1}^{d} P(x_j \mid C_k)
```

### 7.2 Classification Rule

```math
\hat{y} = \arg\max_{k}\ P(C_k)\prod_{j=1}^{d}P(x_j \mid C_k)
```

**Log form** (avoids numerical underflow from multiplying many small numbers):

```math
\hat{y} = \arg\max_{k}\left[\ln P(C_k) + \sum_{j=1}^{d}\ln P(x_j \mid C_k)\right]
```

### 7.3 Parameter Estimation

**Prior:**

```math
P(C_k) = \frac{N_k}{N} \qquad (N_k = \text{number of training examples in class } k)
```

**Categorical features:**

```math
P(x_j = v \mid C_k) = \frac{\text{count}(x_j = v,\ y = C_k)}{N_k}
```

### 7.4 Zero-Frequency Problem and Laplace Smoothing

If some value never occurs with a class in training, its probability is 0, and the whole product becomes 0. **Laplace (add-one) smoothing:**

```math
P(x_j = v \mid C_k) = \frac{\text{count}(x_j = v,\ y = C_k) + \alpha}{N_k + \alpha\,V_j}
```

where $V_j$ = number of distinct values of feature $j$ and $\alpha = 1$ for Laplace smoothing ($0 < \alpha < 1$ is Lidstone smoothing).

### 7.5 Variants of Naïve Bayes

**(a) Gaussian NB** — for continuous features:

```math
P(x_j \mid C_k) = \frac{1}{\sqrt{2\pi\sigma_{jk}^2}}\exp\left(-\frac{(x_j - \mu_{jk})^2}{2\sigma_{jk}^2}\right)
```

where $\mu_{jk}$ and $\sigma_{jk}^2$ are the mean and variance of feature $j$ within class $k$:

```math
\mu_{jk} = \frac{1}{N_k}\sum_{i:\,y_i = k} x_{ij}, \qquad \sigma_{jk}^2 = \frac{1}{N_k}\sum_{i:\,y_i=k}(x_{ij} - \mu_{jk})^2
```

**(b) Multinomial NB** — for word counts (text classification):

```math
P(w \mid C_k) = \frac{\text{count}(w, C_k) + 1}{\sum_{w'} \text{count}(w', C_k) + |V|}
```

($|V|$ = vocabulary size.)

**(c) Bernoulli NB** — for binary features (word present/absent):

```math
P(\mathbf{x} \mid C_k) = \prod_{j=1}^d p_{jk}^{x_j}(1 - p_{jk})^{1 - x_j}
```

### 7.6 Advantages and Disadvantages

| Advantages | Disadvantages |
|---|---|
| Very fast to train and predict | Independence assumption rarely true |
| Works well with small data and high dimensions (text) | Probability estimates are poorly calibrated |
| Handles multi-class naturally | Zero-frequency problem (needs smoothing) |
| Robust to irrelevant features | Correlated features are "double counted" |

### 📝 Solved Numerical 7.1 — Play Tennis (the most important exam problem)

**Dataset (14 days):**

| Day | Outlook | Temperature | Humidity | Wind | Play |
|---|---|---|---|---|---|
| 1 | Sunny | Hot | High | Weak | No |
| 2 | Sunny | Hot | High | Strong | No |
| 3 | Overcast | Hot | High | Weak | Yes |
| 4 | Rain | Mild | High | Weak | Yes |
| 5 | Rain | Cool | Normal | Weak | Yes |
| 6 | Rain | Cool | Normal | Strong | No |
| 7 | Overcast | Cool | Normal | Strong | Yes |
| 8 | Sunny | Mild | High | Weak | No |
| 9 | Sunny | Cool | Normal | Weak | Yes |
| 10 | Rain | Mild | Normal | Weak | Yes |
| 11 | Sunny | Mild | Normal | Strong | Yes |
| 12 | Overcast | Mild | High | Strong | Yes |
| 13 | Overcast | Hot | Normal | Weak | Yes |
| 14 | Rain | Mild | High | Strong | No |

**Classify:** $\mathbf{x}$ = (Outlook = Sunny, Temp = Cool, Humidity = High, Wind = Strong).

**Step 1 — Priors:** 9 Yes, 5 No.

```math
P(\text{Yes}) = \frac{9}{14} = 0.643, \qquad P(\text{No}) = \frac{5}{14} = 0.357
```

**Step 2 — Likelihoods (count from table):**

| Feature value | $P(\cdot \mid \text{Yes})$ | $P(\cdot \mid \text{No})$ |
|---|---|---|
| Outlook = Sunny | 2/9 | 3/5 |
| Temp = Cool | 3/9 | 1/5 |
| Humidity = High | 3/9 | 4/5 |
| Wind = Strong | 3/9 | 3/5 |

**Step 3 — Unnormalised posteriors:**

```math
P(\text{Yes})\prod P(x_j\mid\text{Yes}) = \frac{9}{14}\cdot\frac{2}{9}\cdot\frac{3}{9}\cdot\frac{3}{9}\cdot\frac{3}{9} = 0.643 \times 0.222 \times 0.333 \times 0.333 \times 0.333 = 0.00529
```

```math
P(\text{No})\prod P(x_j\mid\text{No}) = \frac{5}{14}\cdot\frac{3}{5}\cdot\frac{1}{5}\cdot\frac{4}{5}\cdot\frac{3}{5} = 0.357 \times 0.6 \times 0.2 \times 0.8 \times 0.6 = 0.02057
```

**Step 4 — Normalise:**

```math
P(\text{Yes}\mid\mathbf{x}) = \frac{0.00529}{0.00529 + 0.02057} = 0.205, \qquad P(\text{No}\mid\mathbf{x}) = \frac{0.02057}{0.02586} = 0.795
```

**Answer: Play = No** (79.5%).

### 📝 Solved Numerical 7.2 — Text Classification with Laplace Smoothing

**Training documents:**

| Doc | Text | Class |
|---|---|---|
| 1 | free money | Spam |
| 2 | free offer now | Spam |
| 3 | meeting now | Ham |
| 4 | project meeting money | Ham |

Classify: **"free money now"**.

**Vocabulary:** {free, money, offer, now, meeting, project} → $|V| = 6$.
Total words: Spam = 5, Ham = 5. Priors: $P(\text{Spam}) = P(\text{Ham}) = 2/4 = 0.5$.

**Word counts:**

| Word | Count in Spam | Count in Ham |
|---|---|---|
| free | 2 | 0 |
| money | 1 | 1 |
| now | 1 | 1 |

**Smoothed likelihoods** (denominator $5 + 6 = 11$ for both classes):

```math
P(\text{free}\mid S) = \frac{2+1}{11} = 0.2727, \quad P(\text{money}\mid S) = \frac{1+1}{11} = 0.1818, \quad P(\text{now}\mid S) = \frac{1+1}{11} = 0.1818
```

```math
P(\text{free}\mid H) = \frac{0+1}{11} = 0.0909, \quad P(\text{money}\mid H) = \frac{2}{11} = 0.1818, \quad P(\text{now}\mid H) = \frac{2}{11} = 0.1818
```

Without smoothing, $P(\text{free}\mid H) = 0$ would make the Ham score exactly 0.

**Scores:**

```math
\text{Spam: } 0.5 \times 0.2727 \times 0.1818 \times 0.1818 = 0.004508
```

```math
\text{Ham: } 0.5 \times 0.0909 \times 0.1818 \times 0.1818 = 0.001503
```

```math
P(\text{Spam}\mid\text{doc}) = \frac{0.004508}{0.004508 + 0.001503} = 0.75
```

**Answer: Spam.**

### 📝 Solved Numerical 7.3 — Gaussian Naïve Bayes

Classify a fruit with weight feature $x = 6$ (in 100 g). Class A: $\mu = 5.5, \sigma = 0.5$, prior 0.4. Class B: $\mu = 7, \sigma = 1$, prior 0.6.

```math
P(x=6\mid A) = \frac{1}{\sqrt{2\pi(0.25)}}\exp\left(-\frac{(6-5.5)^2}{2(0.25)}\right) = \frac{1}{1.2533}e^{-0.5} = 0.7979 \times 0.6065 = 0.4839
```

```math
P(x=6\mid B) = \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(6-7)^2}{2}\right) = 0.3989 \times 0.6065 = 0.2420
```

```math
\text{A: } 0.4 \times 0.4839 = 0.1936, \qquad \text{B: } 0.6 \times 0.2420 = 0.1452
```

```math
P(A\mid x) = \frac{0.1936}{0.1936 + 0.1452} = 0.571
```

**Answer: Class A.**

---

## 8. Bayesian Decision Theory

A fundamental statistical approach to classification that quantifies the trade-offs between decisions using **probabilities and costs**.

### 8.1 Setup

- Classes (states of nature): $\omega_1, \omega_2, \dots, \omega_c$
- Actions: $\alpha_1, \alpha_2, \dots, \alpha_a$ (usually $\alpha_i$ = "decide $\omega_i$")
- **Loss function** $\lambda(\alpha_i \mid \omega_j) = \lambda_{ij}$: cost of taking action $\alpha_i$ when the true class is $\omega_j$.

### 8.2 Conditional Risk

Expected loss of taking action $\alpha_i$ after observing $\mathbf{x}$:

```math
R(\alpha_i \mid \mathbf{x}) = \sum_{j=1}^{c}\lambda(\alpha_i \mid \omega_j)\,P(\omega_j \mid \mathbf{x})
```

### 8.3 Bayes Decision Rule

Choose the action with **minimum conditional risk**:

```math
\alpha^{\ast}(\mathbf{x}) = \arg\min_{i} R(\alpha_i \mid \mathbf{x})
```

**Overall risk:**

```math
R = \int R(\alpha(\mathbf{x}) \mid \mathbf{x})\,p(\mathbf{x})\,d\mathbf{x}
```

The Bayes rule minimises the overall risk; the minimum value is called the **Bayes risk** — the best achievable performance.

### 8.4 Two-Class Case

```math
R(\alpha_1\mid\mathbf{x}) = \lambda_{11}P(\omega_1\mid\mathbf{x}) + \lambda_{12}P(\omega_2\mid\mathbf{x})
```

```math
R(\alpha_2\mid\mathbf{x}) = \lambda_{21}P(\omega_1\mid\mathbf{x}) + \lambda_{22}P(\omega_2\mid\mathbf{x})
```

Decide $\omega_1$ if $R(\alpha_1\mid\mathbf{x}) < R(\alpha_2\mid\mathbf{x})$, which simplifies to:

```math
(\lambda_{21} - \lambda_{11})\,P(\omega_1\mid\mathbf{x}) > (\lambda_{12} - \lambda_{22})\,P(\omega_2\mid\mathbf{x})
```

**Likelihood Ratio Test** (using Bayes' theorem):

```math
\text{Decide } \omega_1 \text{ if } \quad \frac{p(\mathbf{x}\mid\omega_1)}{p(\mathbf{x}\mid\omega_2)} > \frac{(\lambda_{12}-\lambda_{22})}{(\lambda_{21}-\lambda_{11})}\cdot\frac{P(\omega_2)}{P(\omega_1)} = \theta
```

### 8.5 Minimum-Error-Rate Classification (0–1 Loss)

```math
\lambda(\alpha_i\mid\omega_j) = \begin{cases} 0 & i = j \\ 1 & i \neq j \end{cases}
```

Then:

```math
R(\alpha_i\mid\mathbf{x}) = \sum_{j\neq i}P(\omega_j\mid\mathbf{x}) = 1 - P(\omega_i\mid\mathbf{x})
```

Minimising risk = **maximising posterior** → the **MAP rule**. Probability of error:

```math
P(\text{error}\mid\mathbf{x}) = 1 - \max_i P(\omega_i\mid\mathbf{x})
```

### 8.6 Discriminant Functions

Classifier assigns $\mathbf{x}$ to $\omega_i$ if $g_i(\mathbf{x}) > g_j(\mathbf{x})$ for all $j\neq i$. Equivalent choices:

```math
g_i(\mathbf{x}) = P(\omega_i\mid\mathbf{x}) \quad\text{or}\quad g_i(\mathbf{x}) = p(\mathbf{x}\mid\omega_i)P(\omega_i) \quad\text{or}\quad g_i(\mathbf{x}) = \ln p(\mathbf{x}\mid\omega_i) + \ln P(\omega_i)
```

**For Gaussian class-conditional densities** $p(\mathbf{x}\mid\omega_i) = \mathcal{N}(\boldsymbol{\mu}_i, \Sigma_i)$:

```math
g_i(\mathbf{x}) = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T\Sigma_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i) - \frac{d}{2}\ln 2\pi - \frac{1}{2}\ln|\Sigma_i| + \ln P(\omega_i)
```

| Case | Covariance | Discriminant | Boundary |
|---|---|---|---|
| 1 | $\Sigma_i = \sigma^2 I$ | $g_i = -\frac{\lVert\mathbf{x}-\boldsymbol{\mu}_i\rVert^2}{2\sigma^2} + \ln P(\omega_i)$ | Linear (perpendicular bisector if equal priors) |
| 2 | $\Sigma_i = \Sigma$ (shared) | $g_i = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}_i) + \ln P(\omega_i)$ | Linear (not necessarily perpendicular) |
| 3 | $\Sigma_i$ arbitrary | Full formula | Quadratic (hyperquadrics) |

**Case 1 expanded** — it is linear in $\mathbf{x}$:

```math
g_i(\mathbf{x}) = \mathbf{w}_i^T\mathbf{x} + w_{i0}, \qquad \mathbf{w}_i = \frac{\boldsymbol{\mu}_i}{\sigma^2}, \qquad w_{i0} = -\frac{\boldsymbol{\mu}_i^T\boldsymbol{\mu}_i}{2\sigma^2} + \ln P(\omega_i)
```

**1-D boundary for two Gaussians with equal variance** (set $g_1 = g_2$):

```math
x^{\ast} = \frac{\mu_1 + \mu_2}{2} + \frac{\sigma^2}{\mu_1 - \mu_2}\ln\frac{P(\omega_2)}{P(\omega_1)}
```

With equal priors, $x^{\ast}$ is exactly the midpoint of the means.

### 📝 Solved Numerical 8.1 — Minimum Risk Decision

A patient's test gives posteriors $P(\omega_1\mid x) = 0.3$ (disease), $P(\omega_2\mid x) = 0.7$ (healthy). Loss matrix:

|  | True: Disease ($\omega_1$) | True: Healthy ($\omega_2$) |
|---|---|---|
| $\alpha_1$: Declare disease | $\lambda_{11} = 0$ | $\lambda_{12} = 1$ |
| $\alpha_2$: Declare healthy | $\lambda_{21} = 10$ | $\lambda_{22} = 0$ |

(Missing a disease is 10 times worse than a false alarm.)

```math
R(\alpha_1\mid x) = 0(0.3) + 1(0.7) = 0.7
```

```math
R(\alpha_2\mid x) = 10(0.3) + 0(0.7) = 3.0
```

$R(\alpha_1) < R(\alpha_2)$ → **declare disease**, even though the posterior for disease is only 0.3! Under 0–1 loss, MAP would have said "healthy". The asymmetric cost changes the decision.

**Threshold:** decide disease if $P(\omega_1\mid x)/P(\omega_2\mid x) > (\lambda_{12}-\lambda_{22})/(\lambda_{21}-\lambda_{11}) = 1/10$, i.e., whenever $P(\omega_1\mid x) > 1/11 = 0.0909$.

### 📝 Solved Numerical 8.2 — Gaussian Decision Boundary

Two classes, 1-D Gaussians: $\mu_1 = 2$, $\mu_2 = 6$, common $\sigma^2 = 1$, priors $P(\omega_1) = 0.8$, $P(\omega_2) = 0.2$.

```math
x^{\ast} = \frac{2+6}{2} + \frac{1}{2-6}\ln\frac{0.2}{0.8} = 4 + \frac{-1.3863}{-4} = 4 + 0.3466 = 4.347
```

Decide $\omega_1$ if $x < 4.347$. The boundary shifted from 4 toward $\mu_2$ because $\omega_1$ is more likely a-priori.

💡 **Exam tip:** In Bayesian decision theory questions, always write the conditional-risk formula first, then substitute. Remember that 0–1 loss reduces to MAP.

---

## 9. Formula Sheet

| Concept | Formula |
|---|---|
| Hyperplane | $\mathbf{w}^T\mathbf{x} + b = 0$ |
| Distance to hyperplane | $r = (\mathbf{w}^T\mathbf{x}+b)/\lVert\mathbf{w}\rVert$ |
| SVM margin | $2/\lVert\mathbf{w}\rVert$ |
| Perceptron update | $\mathbf{w} \leftarrow \mathbf{w} + \eta(y-\hat{y})\mathbf{x}$ |
| Perceptron mistake bound | $(R/\gamma)^2$ |
| Sigmoid | $\sigma(z) = 1/(1+e^{-z})$, $\sigma' = \sigma(1-\sigma)$ |
| Log loss | $-\frac{1}{m}\sum[y\ln\hat{p} + (1-y)\ln(1-\hat{p})]$ |
| Logistic gradient | $\frac{1}{m}\sum(\hat{p}_i - y_i)\mathbf{x}_i$ |
| Softmax | $e^{z_k}/\sum_j e^{z_j}$ |
| OvA / OvO classifiers | $K$ / $K(K-1)/2$ |
| Bayes theorem | $P(C\mid\mathbf{x}) = P(\mathbf{x}\mid C)P(C)/P(\mathbf{x})$ |
| Total probability | $P(\mathbf{x}) = \sum_j P(\mathbf{x}\mid C_j)P(C_j)$ |
| Naïve Bayes | $\arg\max_k P(C_k)\prod_j P(x_j\mid C_k)$ |
| Laplace smoothing | $(\text{count}+1)/(N_k + V)$ |
| Gaussian likelihood | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}$ |
| Conditional risk | $R(\alpha_i\mid\mathbf{x}) = \sum_j\lambda_{ij}P(\omega_j\mid\mathbf{x})$ |
| Likelihood ratio test | $\frac{p(\mathbf{x}\mid\omega_1)}{p(\mathbf{x}\mid\omega_2)} > \frac{\lambda_{12}-\lambda_{22}}{\lambda_{21}-\lambda_{11}}\cdot\frac{P(\omega_2)}{P(\omega_1)}$ |
| 1-D Gaussian boundary | $x^{\ast} = \frac{\mu_1+\mu_2}{2} + \frac{\sigma^2}{\mu_1-\mu_2}\ln\frac{P(\omega_2)}{P(\omega_1)}$ |
| Precision / Recall | $TP/(TP+FP)$ / $TP/(TP+FN)$ |
| F1 score | $2PR/(P+R)$ |

---

[⬅ Unit I](Unit-1-Foundations.md) | [Back to Index](README.md) | [Next: Unit III ➡](Unit-3-NonParametric-and-Ensembles.md)
