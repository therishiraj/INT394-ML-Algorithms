# Unit VI — Model Complexity & Optimization

[⬅ Back to Index](README.md)

## Contents

**Part A — Model Complexity**
1. [Why Measure Model Complexity?](#1-why-measure-model-complexity)
2. [VC Dimension](#2-vc-dimension)
3. [Rademacher Complexity](#3-rademacher-complexity)
4. [Bias–Variance Trade-off](#4-biasvariance-trade-off)
5. [Overfitting vs Underfitting](#5-overfitting-vs-underfitting)
6. [Regularization Techniques](#6-regularization-techniques)

**Part B — Optimization**

7. [Cross-Validation](#7-cross-validation)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Structural Risk Minimization (SRM)](#9-structural-risk-minimization-srm)
10. [Gradient Descent and Variants](#10-gradient-descent-and-variants)
11. [Convergence Analysis](#11-convergence-analysis)
12. [Formula Sheet](#12-formula-sheet)

---

# Part A — Model Complexity

## 1. Why Measure Model Complexity?

In Unit I, the PAC bound for a **finite** class depended on $\ln|\mathcal{H}|$. But most useful classes (lines, hyperplanes, neural networks) are **infinite**, so $\ln|\mathcal{H}| = \infty$ and that bound is useless. We need measures of the **effective richness** of a hypothesis class:

- **VC dimension** — a combinatorial measure (how many points can the class label in every possible way?).
- **Rademacher complexity** — a data-dependent measure (how well can the class fit random noise?).

Both lead to **generalization bounds** of the form:

```math
\text{True error} \leq \text{Training error} + \text{Complexity penalty}
```

---

## 2. VC Dimension

Named after **Vapnik and Chervonenkis** (1971).

### 2.1 Dichotomies and Shattering

For a set of $m$ points $C = \lbrace x_1, \dots, x_m\rbrace$, a **dichotomy** is one possible labelling of these points with $\lbrace -1, +1\rbrace$. There are $2^m$ possible dichotomies.

The **restriction** of $\mathcal{H}$ to $C$ is the set of labellings $\mathcal{H}$ can produce:

```math
\mathcal{H}_C = \lbrace (h(x_1), \dots, h(x_m)) : h \in \mathcal{H}\rbrace
```

**Shattering:** $\mathcal{H}$ **shatters** $C$ if it can realise **all** $2^m$ labellings:

```math
|\mathcal{H}_C| = 2^{|C|}
```

### 2.2 Definition of VC Dimension

```math
\text{VCdim}(\mathcal{H}) = \max\lbrace\,|C| : \mathcal{H} \text{ shatters } C\,\rbrace
```

**To prove $\text{VCdim}(\mathcal{H}) = d$ you must show two things:**
1. **There exists** a set of $d$ points that $\mathcal{H}$ shatters (one clever placement is enough).
2. **No** set of $d+1$ points can be shattered by $\mathcal{H}$ (must hold for every placement).

If $\mathcal{H}$ shatters arbitrarily large sets, $\text{VCdim}(\mathcal{H}) = \infty$.

### 2.3 Growth Function

```math
\Pi_{\mathcal{H}}(m) = \max_{C\subset\mathcal{X}:\,|C| = m}|\mathcal{H}_C|
```

- If $m \leq \text{VCdim}(\mathcal{H})$: $\Pi_{\mathcal{H}}(m) = 2^m$.
- **Sauer–Shelah Lemma:** if $\text{VCdim}(\mathcal{H}) = d < \infty$, then

```math
\Pi_{\mathcal{H}}(m) \leq \sum_{i=0}^{d}\binom{m}{i} \leq \left(\frac{em}{d}\right)^d \quad (\text{for } m \geq d)
```

So the growth function goes from **exponential** to **polynomial** once $m > d$. This is why finite VC dimension implies learnability.

### 2.4 Standard Examples

| Hypothesis class | VC dimension | Reason |
|---|---|---|
| Thresholds on the line: $h_a(x) = \mathbb{1}[x \geq a]$ | 1 | Can shatter 1 point; for 2 points $x_1 < x_2$, labelling $(1, 0)$ impossible |
| Intervals on the line: $\mathbb{1}[a \leq x \leq b]$ | 2 | For 3 points $x_1<x_2<x_3$, labelling $(1, 0, 1)$ impossible |
| Linear classifiers (half-planes) in $\mathbb{R}^2$ | 3 | 3 non-collinear points shattered; no 4 points (XOR pattern fails) |
| Linear classifiers (hyperplanes) in $\mathbb{R}^d$ | $d + 1$ | Number of parameters ($d$ weights + 1 bias) |
| Axis-aligned rectangles in $\mathbb{R}^2$ | 4 | 4 points in "diamond" shape shattered; for 5 points, the one inside the bounding box of the others can't be labelled 0 while the rest are 1 |
| Circles (discs) in $\mathbb{R}^2$ | 3 | |
| Finite class $\mathcal{H}$ | $\leq \log_2|\mathcal{H}|$ | Shattering $d$ points needs $2^d$ distinct hypotheses |
| $\lbrace \text{sign}(\sin(\omega x))\rbrace$ | $\infty$ | Only 1 parameter, yet infinite VC dim! |
| Neural network with $W$ weights (sign activation) | $O(W\log W)$ | |

> **Key insight:** VC dimension often equals the number of free parameters, but **not always** (the sine example has 1 parameter but infinite VC dimension).

### 2.5 Proof: VC dimension of half-planes in $\mathbb{R}^2$ is 3

**(a) 3 points can be shattered.** Take 3 non-collinear points (a triangle). There are $2^3 = 8$ labellings. All-positive / all-negative: put the line far away. Any labelling with one point different from the other two: a line can cut off that one vertex of the triangle. So all 8 are achievable. ✓

**(b) No 4 points can be shattered.**
- If the 4 points form a convex quadrilateral, label diagonally opposite corners the same: $(+, -, +, -)$ — the XOR pattern. The two $+$ points' segment and the two $-$ points' segment intersect, so no line separates them.
- If one point lies inside the triangle of the other three, label the inner point $-$ and the outer three $+$. Any half-plane containing the three outer points contains their convex hull, hence also the inner point. ✗

Therefore $\text{VCdim} = 3 = d + 1$ with $d = 2$. ∎

### 2.6 VC Generalization Bound (Vapnik)

With probability at least $1 - \delta$, for all $h \in \mathcal{H}$ with $\text{VCdim}(\mathcal{H}) = h_{vc}$ and $m$ training samples:

```math
L_{\mathcal{D}}(h) \leq L_S(h) + \sqrt{\frac{h_{vc}\left(\ln\frac{2m}{h_{vc}} + 1\right) + \ln\frac{4}{\delta}}{m}}
```

The square-root term is called the **VC confidence** (or capacity term). It grows with $h_{vc}$ and shrinks with $m$.

### 2.7 Fundamental Theorem of Statistical Learning

For binary classification with 0–1 loss, the following are **equivalent**:
1. $\mathcal{H}$ has the uniform convergence property.
2. ERM is a successful (agnostic) PAC learner for $\mathcal{H}$.
3. $\mathcal{H}$ is PAC learnable.
4. $\text{VCdim}(\mathcal{H})$ is **finite**.

**Quantitative version** — sample complexity with $d = \text{VCdim}(\mathcal{H})$ (constants $C_1, C_2$):

```math
\text{Realizable: } C_1\frac{d + \ln(1/\delta)}{\epsilon} \leq m_{\mathcal{H}}(\epsilon, \delta) \leq C_2\frac{d\ln(1/\epsilon) + \ln(1/\delta)}{\epsilon}
```

```math
\text{Agnostic: } C_1\frac{d + \ln(1/\delta)}{\epsilon^2} \leq m_{\mathcal{H}}(\epsilon, \delta) \leq C_2\frac{d + \ln(1/\delta)}{\epsilon^2}
```

**An explicit sufficient bound (Blumer et al., realizable):**

```math
m \geq \frac{1}{\epsilon}\left(4\log_2\frac{2}{\delta} + 8\,d\log_2\frac{13}{\epsilon}\right)
```

### 📝 Solved Numerical 2.1 — Sauer's Lemma

A class has VC dimension $d = 3$. Bound the number of labellings on $m = 10$ points.

```math
\Pi_{\mathcal{H}}(10) \leq \binom{10}{0} + \binom{10}{1} + \binom{10}{2} + \binom{10}{3} = 1 + 10 + 45 + 120 = 176
```

Compare: all possible labellings $= 2^{10} = 1024$. The looser bound $(em/d)^d = (10e/3)^3 = (9.061)^3 = 743.9$.

### 📝 Solved Numerical 2.2 — VC Generalization Bound

$h_{vc} = 10$, $m = 10{,}000$, $\delta = 0.05$, training error $= 0.02$.

```math
\ln\frac{2m}{h_{vc}} + 1 = \ln(2000) + 1 = 7.601 + 1 = 8.601
```

```math
h_{vc}(8.601) = 86.01, \qquad \ln\frac{4}{\delta} = \ln 80 = 4.382
```

```math
\text{VC confidence} = \sqrt{\frac{86.01 + 4.382}{10000}} = \sqrt{0.009039} = 0.0951
```

```math
L_{\mathcal{D}}(h) \leq 0.02 + 0.0951 = 0.115
```

With 95% confidence, test error ≤ 11.5%.

### 📝 Solved Numerical 2.3 — Sample Complexity from VC Dimension

Half-planes in $\mathbb{R}^2$ ($d = 3$), $\epsilon = 0.1$, $\delta = 0.05$ (realizable, Blumer bound):

```math
m \geq \frac{1}{0.1}\left(4\log_2 40 + 8(3)\log_2 130\right) = 10\left(4(5.322) + 24(7.022)\right) = 10(21.29 + 168.53) = 1898.2
```

**Answer:** $m = 1899$ examples suffice.

### 📝 Solved Numerical 2.4 — VC Dimension Reasoning

*Q: What is the VC dimension of a perceptron with 5 inputs?* Linear classifier in $\mathbb{R}^5$ → $d + 1 = 6$.

*Q: A finite class has 64 hypotheses. Upper bound on VC dimension?* $\log_2 64 = 6$.

---

## 3. Rademacher Complexity

### 3.1 Intuition

Rademacher complexity measures how well a hypothesis class can **correlate with random noise**. If a class can fit random ±1 labels well, it is very rich (and prone to overfitting).

### 3.2 Rademacher Variables

$\sigma_1, \dots, \sigma_m$ are i.i.d. with $P(\sigma_i = +1) = P(\sigma_i = -1) = \frac{1}{2}$.

### 3.3 Empirical Rademacher Complexity

For a fixed sample $S = (x_1, \dots, x_m)$:

```math
\hat{\mathcal{R}}_S(\mathcal{H}) = \mathbb{E}_{\boldsymbol{\sigma}}\left[\sup_{h\in\mathcal{H}}\frac{1}{m}\sum_{i=1}^m\sigma_i\,h(x_i)\right]
```

**Interpretation:** $\frac{1}{m}\sum_i\sigma_ih(x_i)$ is the correlation between $h$'s outputs and the random labels. The sup picks the best-correlated hypothesis; the expectation averages over all random labelings.

### 3.4 (Expected) Rademacher Complexity

```math
\mathcal{R}_m(\mathcal{H}) = \mathbb{E}_{S\sim\mathcal{D}^m}\left[\hat{\mathcal{R}}_S(\mathcal{H})\right]
```

### 3.5 Properties

- $0 \leq \hat{\mathcal{R}}_S(\mathcal{H}) \leq 1$ for $\pm 1$-valued $h$.
- If $\mathcal{H}$ shatters $S$, then $\hat{\mathcal{R}}_S(\mathcal{H}) = 1$ (can fit any noise perfectly).
- Single hypothesis: $\hat{\mathcal{R}}_S(\lbrace h\rbrace) = 0$ (since $\mathbb{E}[\sigma_i] = 0$).
- Monotone: $\mathcal{H}_1 \subseteq \mathcal{H}_2 \Rightarrow \hat{\mathcal{R}}_S(\mathcal{H}_1) \leq \hat{\mathcal{R}}_S(\mathcal{H}_2)$.
- Scaling: $\hat{\mathcal{R}}_S(c\mathcal{H}) = |c|\,\hat{\mathcal{R}}_S(\mathcal{H})$.
- Data-dependent — can give tighter bounds than VC dimension.

### 3.6 Key Bounds

**Massart's Lemma (finite class, $\pm 1$ outputs):**

```math
\hat{\mathcal{R}}_S(\mathcal{H}) \leq \sqrt{\frac{2\ln|\mathcal{H}|}{m}}
```

**Relation to VC dimension:**

```math
\mathcal{R}_m(\mathcal{H}) \leq \sqrt{\frac{2d\ln(em/d)}{m}}, \qquad d = \text{VCdim}(\mathcal{H})
```

**Linear class with bounded norm** $\mathcal{H} = \lbrace\mathbf{x}\mapsto\mathbf{w}^T\mathbf{x} : \lVert\mathbf{w}\rVert_2 \leq B\rbrace$, $\lVert\mathbf{x}\rVert_2 \leq R$:

```math
\hat{\mathcal{R}}_S(\mathcal{H}) \leq \frac{BR}{\sqrt{m}}
```

(This is why keeping weights small — regularization — controls complexity.)

### 3.7 Rademacher Generalization Bound

For a loss with values in $[0, 1]$, with probability at least $1 - \delta$, for all $h\in\mathcal{H}$:

```math
L_{\mathcal{D}}(h) \leq L_S(h) + 2\,\mathcal{R}_m(\ell\circ\mathcal{H}) + \sqrt{\frac{\ln(1/\delta)}{2m}}
```

Using the empirical Rademacher complexity instead:

```math
L_{\mathcal{D}}(h) \leq L_S(h) + 2\,\hat{\mathcal{R}}_S(\ell\circ\mathcal{H}) + 3\sqrt{\frac{\ln(2/\delta)}{2m}}
```

### 📝 Solved Numerical 3.1 — Computing Empirical Rademacher Complexity

$m = 2$ points, $\mathcal{H} = \lbrace h_1, h_2\rbrace$ with outputs $h_1 = (+1, +1)$, $h_2 = (+1, -1)$.

All $2^2 = 4$ equally likely $\boldsymbol{\sigma}$ vectors:

| $\boldsymbol{\sigma}$ | $\frac{1}{2}\boldsymbol{\sigma}\cdot h_1$ | $\frac{1}{2}\boldsymbol{\sigma}\cdot h_2$ | sup |
|---|---|---|---|
| $(+1, +1)$ | $\frac{1}{2}(1 + 1) = 1$ | $\frac{1}{2}(1 - 1) = 0$ | 1 |
| $(+1, -1)$ | $\frac{1}{2}(1 - 1) = 0$ | $\frac{1}{2}(1 + 1) = 1$ | 1 |
| $(-1, +1)$ | $\frac{1}{2}(-1 + 1) = 0$ | $\frac{1}{2}(-1 - 1) = -1$ | 0 |
| $(-1, -1)$ | $\frac{1}{2}(-1 - 1) = -1$ | $\frac{1}{2}(-1 + 1) = 0$ | 0 |

```math
\hat{\mathcal{R}}_S(\mathcal{H}) = \frac{1 + 1 + 0 + 0}{4} = 0.5
```

**Massart check:** $\sqrt{2\ln 2/2} = \sqrt{0.693} = 0.833 \geq 0.5$ ✓.

(If we added $h_3 = (-1, +1)$ and $h_4 = (-1, -1)$, the class would shatter $S$ and $\hat{\mathcal{R}}_S = 1$.)

### 📝 Solved Numerical 3.2 — Rademacher Bound

$L_S(h) = 0.10$, $\mathcal{R}_m = 0.05$, $m = 1000$, $\delta = 0.05$.

```math
\sqrt{\frac{\ln(1/0.05)}{2 \times 1000}} = \sqrt{\frac{2.996}{2000}} = \sqrt{0.001498} = 0.0387
```

```math
L_{\mathcal{D}}(h) \leq 0.10 + 2(0.05) + 0.0387 = 0.2387
```

### 3.8 VC Dimension vs Rademacher Complexity

| Aspect | VC Dimension | Rademacher Complexity |
|---|---|---|
| Type | Combinatorial (a single integer) | Real number, expectation |
| Data-dependent? | No (worst-case over all datasets) | Yes (depends on the distribution/sample) |
| Applies to | Binary classification | Any real-valued class (regression too) |
| Tightness | Often loose | Usually tighter |
| Computation | Analytical proof | Can be estimated by sampling $\boldsymbol{\sigma}$ |

---

## 4. Bias–Variance Trade-off

### 4.1 Setup

True relationship: $y = f(x) + \varepsilon$, where $\mathbb{E}[\varepsilon] = 0$, $\text{Var}(\varepsilon) = \sigma^2$. We train $\hat{f}$ on a random training set $D$. $\hat{f}$ is random (depends on $D$).

**Definitions (at a point $x$):**

```math
\text{Bias}[\hat{f}(x)] = \mathbb{E}_D[\hat{f}(x)] - f(x)
```

```math
\text{Var}[\hat{f}(x)] = \mathbb{E}_D\left[\left(\hat{f}(x) - \mathbb{E}_D[\hat{f}(x)]\right)^2\right]
```

### 4.2 Decomposition (Derivation)

Let $\bar{f}(x) = \mathbb{E}_D[\hat{f}(x)]$. Write $y - \hat{f} = (f - \bar{f}) + (\bar{f} - \hat{f}) + \varepsilon$. Square and take expectations — all cross terms vanish because $\mathbb{E}[\varepsilon] = 0$, $\varepsilon$ is independent of $D$, and $\mathbb{E}_D[\bar{f} - \hat{f}] = 0$:

```math
\begin{aligned}
\mathbb{E}\left[(y - \hat{f}(x))^2\right] &= \mathbb{E}\left[(f - \bar{f})^2\right] + \mathbb{E}\left[(\bar{f} - \hat{f})^2\right] + \mathbb{E}[\varepsilon^2]\\
&= \underbrace{\left(\bar{f}(x) - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_D\left[(\hat{f}(x) - \bar{f}(x))^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible error}}
\end{aligned}
```

```math
\boxed{\text{Expected Test Error} = \text{Bias}^2 + \text{Variance} + \sigma^2}
```

| Term | Meaning | Caused by |
|---|---|---|
| **Bias²** | Error from wrong assumptions; model systematically misses the true pattern | Too simple a model (underfitting) |
| **Variance** | Sensitivity to fluctuations in the training set | Too complex a model (overfitting) |
| **Irreducible error $\sigma^2$** | Noise in the data | Cannot be reduced by any model |

### 4.3 The Trade-off

As model complexity increases: **bias ↓, variance ↑**. Total error is U-shaped; the optimum lies in between.

| Model complexity → | Very simple | Simple | **Optimal** | Complex | Very complex |
|---|---|---|---|---|---|
| Bias² | Very high | High | Moderate | Low | Very low |
| Variance | Very low | Low | Moderate | High | Very high |
| Training error | High | Medium | Low | Very low | ≈ 0 |
| Test error (Bias² + Var + σ²) | High | Medium | **Minimum** | Medium | High |
| Regime | Underfitting | | Sweet spot | | Overfitting |

| Model | Bias | Variance |
|---|---|---|
| Linear regression on non-linear data | High | Low |
| High-degree polynomial | Low | High |
| KNN with $k = 1$ | Low | High |
| KNN with large $k$ | High | Low |
| Deep unpruned decision tree | Low | High |
| Decision stump | High | Low |
| Bagging / Random Forest | ≈ same | ↓ Reduced |
| Boosting | ↓ Reduced | may ↑ |

**For KNN regression** (analytical form):

```math
\text{Err}(x_0) = \sigma^2 + \left[f(x_0) - \frac{1}{k}\sum_{l=1}^kf(x_{(l)})\right]^2 + \frac{\sigma^2}{k}
```

Variance $\sigma^2/k$ falls with $k$; bias usually rises with $k$.

### 📝 Solved Numerical 4.1 — Bias–Variance Decomposition

True $f(x_0) = 3.5$, noise variance $\sigma^2 = 0.25$. A model trained on 4 different training sets predicts at $x_0$: $2.8, 3.2, 3.0, 3.4$.

**Mean prediction:**

```math
\bar{f}(x_0) = \frac{2.8 + 3.2 + 3.0 + 3.4}{4} = \frac{12.4}{4} = 3.1
```

**Bias²:**

```math
\text{Bias}^2 = (3.1 - 3.5)^2 = (-0.4)^2 = 0.16
```

**Variance:**

```math
\text{Var} = \frac{(2.8 - 3.1)^2 + (3.2 - 3.1)^2 + (3.0 - 3.1)^2 + (3.4 - 3.1)^2}{4} = \frac{0.09 + 0.01 + 0.01 + 0.09}{4} = 0.05
```

**Expected error:**

```math
\text{Error} = 0.16 + 0.05 + 0.25 = 0.46
```

Bias dominates the reducible error (0.16 vs 0.05) → the model is **underfitting**; a more flexible model would help.

---

## 5. Overfitting vs Underfitting

| Aspect | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| Training error | High | Low | Very low |
| Test/validation error | High | Low | High |
| Gap (test − train) | Small | Small | **Large** |
| Bias / Variance | High bias | Balanced | High variance |
| Model | Too simple | Right complexity | Too complex |
| Example | Straight line on curved data | Quadratic on quadratic data | Degree-15 polynomial on 10 points |

### 5.1 Diagnosing with Learning Curves

Plot training and validation error vs **training-set size**:
- **Underfitting:** both curves plateau at a **high** error, close together. → More data will NOT help; increase complexity.
- **Overfitting:** training error low, validation error much higher, with a gap that narrows slowly. → More data helps; or regularize.

Plot vs **model complexity (or training epochs)**: training error keeps decreasing; validation error decreases then **increases** — the turning point is where overfitting begins.

### 5.2 Remedies

| Underfitting | Overfitting |
|---|---|
| Increase model complexity (more features, higher degree, deeper tree) | Simplify the model / fewer features |
| Add feature engineering / interactions | Get more training data / data augmentation |
| Reduce regularization ($\lambda\downarrow$) | Increase regularization ($\lambda\uparrow$) |
| Train longer | Early stopping |
| Use a more powerful algorithm (ensembles, boosting) | Pruning, dropout, bagging, cross-validation |

### 📝 Solved Numerical 5.1 — Diagnose

| Model | Train error | Validation error | Diagnosis |
|---|---|---|---|
| A | 25% | 27% | Underfitting (both high, small gap) |
| B | 2% | 20% | Overfitting (gap = 18%) |
| C | 6% | 8% | Good fit |

---

## 6. Regularization Techniques

**Regularization** adds a penalty on model complexity to the loss:

```math
J_{\text{reg}}(\mathbf{w}) = \underbrace{J(\mathbf{w})}_{\text{data loss}} + \lambda\,\underbrace{\Omega(\mathbf{w})}_{\text{penalty}}
```

$\lambda \geq 0$ controls the trade-off: $\lambda = 0$ → no regularization; $\lambda\rightarrow\infty$ → weights forced to 0 (underfitting).

### 6.1 L2 Regularization — Ridge Regression (Weight Decay)

```math
J(\mathbf{w}) = \sum_{i=1}^n(y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda\lVert\mathbf{w}\rVert_2^2 = \lVert\mathbf{y} - X\mathbf{w}\rVert^2 + \lambda\sum_{j=1}^dw_j^2
```

**Closed form:**

```math
\mathbf{w}_{\text{ridge}} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}
```

(Adding $\lambda I$ also makes the matrix invertible — solves multicollinearity.)

**Gradient update (weight decay):**

```math
\mathbf{w} \leftarrow \mathbf{w} - \eta\left(\nabla J_{\text{data}} + 2\lambda\mathbf{w}\right) = (1 - 2\eta\lambda)\mathbf{w} - \eta\nabla J_{\text{data}}
```

Effect: shrinks all weights **towards** zero but rarely **exactly** zero. Bayesian view: Gaussian prior on weights (MAP estimate).

### 6.2 L1 Regularization — Lasso Regression

```math
J(\mathbf{w}) = \sum_{i=1}^n(y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda\lVert\mathbf{w}\rVert_1 = \lVert\mathbf{y} - X\mathbf{w}\rVert^2 + \lambda\sum_{j=1}^d|w_j|
```

- No closed form in general (non-differentiable at 0); solved by coordinate descent / subgradient.
- Produces **sparse** solutions — some weights become exactly 0 → automatic **feature selection**.
- Bayesian view: Laplace prior.
- **Why sparse?** The L1 constraint region $\sum|w_j| \leq t$ is a diamond with corners on the axes; the elliptical loss contours usually touch it at a corner, where some $w_j = 0$. The L2 region is a circle (no corners).

**Soft-thresholding operator** (1-D / orthonormal solution):

```math
S(z, \lambda) = \text{sign}(z)\max(|z| - \lambda,\ 0)
```

### 6.3 Elastic Net

```math
J(\mathbf{w}) = \lVert\mathbf{y} - X\mathbf{w}\rVert^2 + \lambda_1\lVert\mathbf{w}\rVert_1 + \lambda_2\lVert\mathbf{w}\rVert_2^2
```

Combines sparsity (L1) with stability for correlated features (L2).

### 6.4 Comparison

| Aspect | Ridge (L2) | Lasso (L1) | Elastic Net |
|---|---|---|---|
| Penalty | $\lambda\sum w_j^2$ | $\lambda\sum\lvert w_j\rvert$ | Both |
| Sparsity | No | Yes | Yes |
| Closed form | Yes | No | No |
| Correlated features | Shares weight among them | Picks one arbitrarily | Groups them |
| Constraint shape | Circle / sphere | Diamond | Rounded diamond |

### 📝 Solved Numerical 6.1 — Ridge vs Lasso in 1-D

Use centred data from Unit IV: $S_{xy} = \sum x_iy_i = 6$, $S_{xx} = \sum x_i^2 = 10$ (no intercept on centred data). Objectives:

```math
\text{Ridge: } \frac{1}{2}\sum(y_i - wx_i)^2 + \frac{\lambda}{2}w^2 \Rightarrow w = \frac{S_{xy}}{S_{xx} + \lambda}
```

```math
\text{Lasso: } \frac{1}{2}\sum(y_i - wx_i)^2 + \lambda|w| \Rightarrow w = \frac{S(S_{xy}, \lambda)}{S_{xx}} = \frac{\text{sign}(S_{xy})\max(|S_{xy}| - \lambda, 0)}{S_{xx}}
```

**Derivation (ridge):** $\frac{d}{dw} = -\sum x_i(y_i - wx_i) + \lambda w = -S_{xy} + wS_{xx} + \lambda w = 0$.

| $\lambda$ | OLS | Ridge $w$ | Lasso $w$ |
|---|---|---|---|
| 0 | 0.6 | $6/10 = 0.600$ | $6/10 = 0.600$ |
| 2 | 0.6 | $6/12 = 0.500$ | $(6 - 2)/10 = 0.400$ |
| 5 | 0.6 | $6/15 = 0.400$ | $(6 - 5)/10 = 0.100$ |
| 7 | 0.6 | $6/17 = 0.353$ | $\max(6 - 7, 0)/10 = $ **0** |
| 10 | 0.6 | $6/20 = 0.300$ | **0** |

Ridge shrinks smoothly but never reaches 0; Lasso hits **exactly 0** once $\lambda \geq |S_{xy}| = 6$.

### 6.5 Other Regularization Techniques

**Early stopping:** monitor validation error during training; stop when it stops improving for $p$ epochs (patience). Acts like L2 regularization for GD on linear models.

**Dropout (neural networks):** during training, each neuron is kept with probability $p$ (dropped with $1 - p$):

```math
\tilde{\mathbf{h}} = \mathbf{m}\odot\mathbf{h}, \quad m_j \sim \text{Bernoulli}(p)
```

At test time use all neurons and scale activations by $p$ (or, equivalently, **inverted dropout**: scale by $1/p$ during training). Prevents co-adaptation; approximates an ensemble of $2^n$ thinned networks.

*Example:* activation $h = 4$, keep probability $p = 0.8$. Inverted dropout during training: if kept, output $4/0.8 = 5$; if dropped, 0. Expected value $= 0.8(5) = 4$ ✓ — matches the test-time output of 4.

**Data augmentation:** create modified copies (rotations, flips, crops, noise) to enlarge the dataset.

**Batch normalization, noise injection, weight constraints (max-norm), pruning (trees), ensembling.**

---

# Part B — Optimization

## 7. Cross-Validation

Used to (a) estimate how well a model generalises and (b) select models/hyperparameters without touching the test set.

### 7.1 Hold-out Validation

Split data into train (e.g., 70%) / validation (15%) / test (15%). Simple, but the estimate depends heavily on which points land in which split (high variance), and it wastes data.

### 7.2 k-Fold Cross-Validation

1. Randomly split data into $k$ equal folds $F_1, \dots, F_k$.
2. For $i = 1$ to $k$: train on all folds except $F_i$; compute error $E_i$ on $F_i$.
3. CV estimate:

```math
CV_{(k)} = \frac{1}{k}\sum_{i=1}^kE_i, \qquad SE = \frac{s}{\sqrt{k}}, \quad s = \sqrt{\frac{1}{k-1}\sum_{i=1}^k(E_i - CV_{(k)})^2}
```

Every point is used for validation exactly once and for training $k - 1$ times. Common: $k = 5$ or $10$.

### 7.3 Variants

| Method | Description | Pros / Cons |
|---|---|---|
| **LOOCV** | $k = n$; leave one point out each time | Nearly unbiased; very expensive ($n$ fits); high variance |
| **Stratified k-fold** | Each fold keeps class proportions | Essential for imbalanced classification |
| **Repeated k-fold** | Repeat k-fold with different shuffles and average | More stable estimate |
| **Time-series CV** (forward chaining) | Train on past, validate on future: train [1], test [2]; train [1,2], test [3]; … | Respects temporal order (no future leakage) |
| **Group k-fold** | Same group (e.g., same patient) never in both train and validation | Prevents leakage |
| **Nested CV** | Outer loop estimates performance; inner loop tunes hyperparameters | Unbiased estimate when tuning |

**LOOCV shortcut for linear regression** ($h_{ii}$ = leverage, diagonal of hat matrix $H = X(X^TX)^{-1}X^T$):

```math
CV_{(n)} = \frac{1}{n}\sum_{i=1}^n\left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2
```

**Bias–variance of k:** small $k$ → each model sees less data → pessimistic (biased) estimate; large $k$ → less bias but more variance and cost.

**1-SE rule:** choose the simplest model whose CV error is within one standard error of the minimum.

### 📝 Solved Numerical 7.1 — 5-Fold CV

Fold errors: $0.12, 0.10, 0.15, 0.11, 0.12$.

```math
CV_{(5)} = \frac{0.12 + 0.10 + 0.15 + 0.11 + 0.12}{5} = \frac{0.60}{5} = 0.12
```

```math
s = \sqrt{\frac{0 + 0.0004 + 0.0009 + 0.0001 + 0}{4}} = \sqrt{0.00035} = 0.0187, \qquad SE = \frac{0.0187}{\sqrt{5}} = 0.0084
```

Report: error $= 0.12 \pm 0.008$.

### 📝 Solved Numerical 7.2 — Data per Fold

1000 samples, 10-fold CV: each fold has 100 samples; each model trains on 900 and validates on 100; 10 models are trained. LOOCV would train 1000 models on 999 samples each.

---

## 8. Hyperparameter Tuning

**Parameters** (weights $\mathbf{w}$) are learned from data. **Hyperparameters** are set **before** training and control the learning process: learning rate $\eta$, $\lambda$, $k$ in KNN, tree depth, number of trees, $C$ and $\gamma$ in SVM, batch size, number of layers.

### 8.1 Methods

| Method | How it works | Pros | Cons |
|---|---|---|---|
| **Manual** | Trial and error with intuition | Uses expertise | Slow, not reproducible |
| **Grid Search** | Try every combination on a predefined grid | Exhaustive, simple, parallel | Exponential cost in number of hyperparameters |
| **Random Search** | Sample combinations randomly from distributions | More efficient when few hyperparameters matter | May miss the optimum |
| **Bayesian Optimization** | Build a surrogate model (Gaussian Process / TPE) of score vs hyperparameters; choose next point via an acquisition function (e.g., Expected Improvement) | Few evaluations needed | Sequential, more complex |
| **Successive Halving / Hyperband** | Start many configs with small budget; keep the best half, double the budget | Efficient early stopping of bad configs | Needs a budget notion |
| **Evolutionary / Genetic** | Population of configs evolves | Handles complex spaces | Expensive |

**Why random search beats grid search (Bergstra & Bengio, 2012):** if only 1 of 2 hyperparameters matters, a 3×3 grid tests only 3 distinct values of the important one, while 9 random trials test 9 distinct values.

**Expected Improvement acquisition:**

```math
EI(\mathbf{x}) = \mathbb{E}\left[\max(f(\mathbf{x}) - f(\mathbf{x}^+), 0)\right]
```

where $f(\mathbf{x}^+)$ is the best score found so far.

**Learning rate / regularization strength** are usually searched on a **log scale**: $\lbrace 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\rbrace$.

### 📝 Solved Numerical 8.1 — Cost of Grid Search

SVM grid: $C \in \lbrace 0.1, 1, 10\rbrace$, $\gamma \in \lbrace 0.001, 0.01, 0.1, 1\rbrace$, kernel $\in \lbrace\text{linear, rbf}\rbrace$, with 5-fold CV.

```math
\text{Combinations} = 3 \times 4 \times 2 = 24, \qquad \text{Model fits} = 24 \times 5 = 120
```

(+1 final refit on all training data = 121.) Adding one more hyperparameter with 5 values → $24 \times 5 \times 5 = 600$ fits — the "curse of dimensionality" of grid search.

**Probability argument for random search:** if the top 5% of the space is "good", the chance that $n$ random trials all miss it is $0.95^n$. For $n = 60$: $0.95^{60} = 0.046$, so 60 random trials find a top-5% configuration with ≈ 95% probability.

---

## 9. Structural Risk Minimization (SRM)

Proposed by **Vapnik**. ERM minimises only training error, which can overfit with a rich class. SRM balances **training error** and **complexity**.

### 9.1 Idea

1. Define a **nested sequence** of hypothesis classes with increasing complexity:

```math
\mathcal{H}_1 \subset \mathcal{H}_2 \subset \mathcal{H}_3 \subset \dots, \qquad h_1 \leq h_2 \leq h_3 \leq \dots \quad (h_i = \text{VCdim}(\mathcal{H}_i))
```

   Example: polynomials of degree 1, 2, 3, …; or linear classifiers with $\lVert\mathbf{w}\rVert \leq B_1 < B_2 < \dots$

2. In each class, find the ERM solution $\hat{h}_i$.
3. Choose the class (and its $\hat{h}_i$) that minimises the **guaranteed risk** (upper bound):

```math
\hat{h}_{SRM} = \arg\min_{i}\left[L_S(\hat{h}_i) + \sqrt{\frac{h_i\left(\ln\frac{2m}{h_i} + 1\right) + \ln\frac{4}{\delta}}{m}}\right]
```

**General SRM (with weights $w(n)$ over classes, $\sum_n w(n) \leq 1$):**

```math
L_{\mathcal{D}}(h) \leq L_S(h) + \min_{n:\,h\in\mathcal{H}_n}\epsilon_n\left(m, w(n)\,\delta\right)
```

### 9.2 ERM vs SRM

| ERM | SRM |
|---|---|
| Minimises empirical risk only | Minimises empirical risk + complexity penalty |
| Fixed hypothesis class | Chooses among nested classes |
| Can overfit with rich classes | Automatically trades off fit vs complexity |
| — | Theoretical basis for regularization and SVM (maximising margin = minimising $\lVert\mathbf{w}\rVert$ = smaller capacity class) |

**Regularization is SRM in practice:** minimising $L_S(\mathbf{w}) + \lambda\lVert\mathbf{w}\rVert^2$ corresponds to choosing among classes $\lbrace\mathbf{w} : \lVert\mathbf{w}\rVert \leq B\rbrace$ of increasing $B$.

### 📝 Solved Numerical 9.1 — Model Selection with SRM

$m = 1000$, $\delta = 0.05$. Nested classes with VC dimensions and training errors:

| Class | $h_i$ | Training error | VC confidence | **Guaranteed risk** |
|---|---|---|---|---|
| $\mathcal{H}_1$ | 1 | 0.30 | 0.114 | 0.414 |
| $\mathcal{H}_2$ | 2 | 0.18 | 0.142 | 0.322 |
| $\mathcal{H}_3$ | 3 | 0.10 | 0.164 | **0.264** ← minimum |
| $\mathcal{H}_4$ | 5 | 0.08 | 0.198 | 0.278 |
| $\mathcal{H}_5$ | 10 | 0.06 | 0.260 | 0.320 |
| $\mathcal{H}_6$ | 20 | 0.05 | 0.341 | 0.391 |

**Sample calculation for $\mathcal{H}_3$ ($h = 3$):**

```math
\sqrt{\frac{3\left(\ln\frac{2000}{3} + 1\right) + \ln 80}{1000}} = \sqrt{\frac{3(6.502 + 1) + 4.382}{1000}} = \sqrt{\frac{26.888}{1000}} = \sqrt{0.02689} = 0.164
```

**ERM** would choose $\mathcal{H}_6$ (lowest training error 0.05); **SRM** chooses $\mathcal{H}_3$ — the best balance.

---

## 10. Gradient Descent and Variants

### 10.1 Gradient Descent (Batch GD)

To minimise $J(\boldsymbol{\theta})$, move opposite to the gradient:

```math
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\,\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}_t)
```

Batch GD uses **all** $n$ examples: $\nabla J = \frac{1}{n}\sum_{i=1}^n\nabla\ell_i(\boldsymbol{\theta})$.

**Why negative gradient?** First-order Taylor: $J(\boldsymbol{\theta} + \Delta) \approx J(\boldsymbol{\theta}) + \nabla J^T\Delta$. With $\Delta = -\eta\nabla J$: $J$ changes by $-\eta\lVert\nabla J\rVert^2 \leq 0$ — steepest decrease.

**Learning rate $\eta$:** too small → very slow; too large → oscillation / divergence.

### 10.2 Stochastic and Mini-batch GD

**SGD** (one random example per step):

```math
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\,\nabla\ell_{i}(\boldsymbol{\theta}_t), \quad i \sim \text{Uniform}\lbrace 1, \dots, n\rbrace
```

**Mini-batch GD** (batch $B$ of size $b$, typically 32–256):

```math
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{b}\sum_{i\in B}\nabla\ell_i(\boldsymbol{\theta}_t)
```

| Aspect | Batch GD | SGD | Mini-batch GD |
|---|---|---|---|
| Examples per update | All $n$ | 1 | $b$ |
| Cost per update | High | Very low | Moderate |
| Gradient noise | None | High | Moderate |
| Convergence path | Smooth | Very noisy (can escape shallow local minima) | Fairly smooth |
| Hardware efficiency | Memory heavy | Poor vectorisation | Best (GPU-friendly) |
| Updates per epoch | 1 | $n$ | $n/b$ |

The stochastic gradient is **unbiased**: $\mathbb{E}_i[\nabla\ell_i] = \nabla J$.

### 10.3 Momentum

Accumulates an exponentially decaying average of past gradients — accelerates along consistent directions and damps oscillations (like a ball rolling downhill):

```math
\mathbf{v}_{t} = \beta\mathbf{v}_{t-1} + \eta\nabla J(\boldsymbol{\theta}_{t-1}), \qquad \boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \mathbf{v}_t
```

Typical $\beta = 0.9$. (Alternative form: $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + (1 - \beta)\nabla J$, $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\mathbf{v}_t$.) With constant gradient $g$, velocity approaches $\eta g/(1 - \beta)$ — an effective $10\times$ larger step for $\beta = 0.9$.

**Nesterov Accelerated Gradient (NAG)** — "look ahead" before computing the gradient:

```math
\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \eta\nabla J(\boldsymbol{\theta}_{t-1} - \beta\mathbf{v}_{t-1}), \qquad \boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \mathbf{v}_t
```

### 10.4 Adaptive Learning Rates

**AdaGrad** — per-parameter learning rate; accumulates all past squared gradients:

```math
G_t = G_{t-1} + g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon}\,g_t
```

Good for sparse features, but $G_t$ keeps growing → learning rate shrinks to ~0 and learning stops.

**RMSprop** (Hinton) — fixes AdaGrad by using an **exponential moving average** of squared gradients:

```math
E[g^2]_t = \rho\,E[g^2]_{t-1} + (1 - \rho)\,g_t^2
```

```math
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}}\,g_t
```

Typical $\rho = 0.9$, $\eta = 0.001$, $\epsilon = 10^{-8}$. Parameters with large gradients get smaller steps and vice versa → handles different curvature per direction.

**Adam** (Adaptive Moment Estimation) = Momentum + RMSprop:

```math
m_t = \beta_1m_{t-1} + (1 - \beta_1)g_t \qquad \text{(1st moment)}
```

```math
v_t = \beta_2v_{t-1} + (1 - \beta_2)g_t^2 \qquad \text{(2nd moment)}
```

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \qquad \text{(bias correction)}
```

```math
\theta_{t+1} = \theta_t - \frac{\eta\,\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.001$, $\epsilon = 10^{-8}$.

### 10.5 Summary of Optimizers

| Optimizer | Key idea | Update |
|---|---|---|
| GD / SGD | Follow negative gradient | $\theta \leftarrow \theta - \eta g$ |
| Momentum | Accumulate velocity | $v \leftarrow \beta v + \eta g$; $\theta \leftarrow \theta - v$ |
| NAG | Gradient at look-ahead point | $g$ evaluated at $\theta - \beta v$ |
| AdaGrad | Divide by root of sum of squared grads | $\theta \leftarrow \theta - \eta g/\sqrt{G}$ |
| RMSprop | Divide by root of EMA of squared grads | $\theta \leftarrow \theta - \eta g/\sqrt{E[g^2]}$ |
| Adam | Momentum + RMSprop + bias correction | $\theta \leftarrow \theta - \eta\hat{m}/\sqrt{\hat{v}}$ |

### 📝 Solved Numerical 10.1 — Plain Gradient Descent

Minimise $J(\theta) = \theta^2$ ($\nabla J = 2\theta$), $\theta_0 = 5$, $\eta = 0.1$.

```math
\theta_{t+1} = \theta_t - 0.1(2\theta_t) = 0.8\,\theta_t
```

| $t$ | $\theta_t$ | $g = 2\theta_t$ | $J(\theta_t)$ |
|---|---|---|---|
| 0 | 5.000 | 10.00 | 25.00 |
| 1 | 4.000 | 8.00 | 16.00 |
| 2 | 3.200 | 6.40 | 10.24 |
| 3 | 2.560 | 5.12 | 6.55 |

In closed form $\theta_t = 5(0.8)^t$ → converges to 0.

### 📝 Solved Numerical 10.2 — Momentum

Same problem, $\eta = 0.1$, $\beta = 0.9$, $v_0 = 0$.

| $t$ | $g_t = 2\theta_{t-1}$ | $v_t = 0.9v_{t-1} + 0.1g_t$ | $\theta_t = \theta_{t-1} - v_t$ |
|---|---|---|---|
| 1 | $2(5) = 10$ | $0 + 1.0 = 1.00$ | $5 - 1 = 4.00$ |
| 2 | $2(4) = 8$ | $0.9 + 0.8 = 1.70$ | $4 - 1.7 = 2.30$ |
| 3 | $2(2.3) = 4.6$ | $1.53 + 0.46 = 1.99$ | $2.3 - 1.99 = 0.31$ |

After 3 steps momentum reaches $\theta = 0.31$ vs $2.56$ for plain GD. (It will overshoot past 0 next and oscillate with decaying amplitude.)

### 📝 Solved Numerical 10.3 — RMSprop

Same problem, $\eta = 0.1$, $\rho = 0.9$, $E_0 = 0$, $\epsilon \approx 0$.

**Step 1:** $g_1 = 10$

```math
E_1 = 0.9(0) + 0.1(10^2) = 10, \qquad \Delta = \frac{0.1 \times 10}{\sqrt{10}} = \frac{1}{3.162} = 0.316, \qquad \theta_1 = 5 - 0.316 = 4.684
```

**Step 2:** $g_2 = 2(4.684) = 9.368$

```math
E_2 = 0.9(10) + 0.1(9.368^2) = 9 + 8.775 = 17.775, \qquad \Delta = \frac{0.1 \times 9.368}{\sqrt{17.775}} = \frac{0.9368}{4.216} = 0.222
```

```math
\theta_2 = 4.684 - 0.222 = 4.462
```

**Step 3:** $g_3 = 8.923$, $E_3 = 0.9(17.775) + 0.1(79.62) = 23.960$, $\Delta = 0.8923/4.895 = 0.182$, $\theta_3 = 4.279$.

Note: the step size is ≈ $\eta$ regardless of the gradient's magnitude (normalised steps).

### 📝 Solved Numerical 10.4 — Adam (first step)

Same problem, $\eta = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$. $g_1 = 10$.

```math
m_1 = 0.1(10) = 1, \qquad v_1 = 0.001(100) = 0.1
```

```math
\hat{m}_1 = \frac{1}{1 - 0.9} = 10, \qquad \hat{v}_1 = \frac{0.1}{1 - 0.999} = 100
```

```math
\theta_1 = 5 - \frac{0.1 \times 10}{\sqrt{100}} = 5 - 0.1 = 4.9
```

Bias correction makes the first step exactly $\eta = 0.1$ in the gradient's direction. Next steps: $\theta_2 = 4.800$, $\theta_3 = 4.700$.

### 📝 Solved Numerical 10.5 — Mini-batch Count

$n = 50{,}000$ samples, batch size $b = 128$: updates per epoch $= \lceil 50000/128\rceil = 391$. For 20 epochs: $7820$ updates. Batch GD would do only 20 updates; SGD would do $1{,}000{,}000$.

---

## 11. Convergence Analysis

### 11.1 Key Definitions

**Convex function:**

```math
f(\lambda\mathbf{x} + (1 - \lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1 - \lambda)f(\mathbf{y}), \quad \lambda\in[0, 1]
```

Equivalently $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x})$; for twice-differentiable $f$: Hessian $\nabla^2f \succeq 0$. Every local minimum is global.

**L-smooth (Lipschitz-continuous gradient):**

```math
\lVert\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\rVert \leq L\lVert\mathbf{x} - \mathbf{y}\rVert \quad\Rightarrow\quad f(\mathbf{y}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) + \frac{L}{2}\lVert\mathbf{y} - \mathbf{x}\rVert^2
```

**μ-strongly convex:**

```math
f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}\lVert\mathbf{y} - \mathbf{x}\rVert^2
```

**Condition number:** $\kappa = L/\mu \geq 1$ (for a quadratic, ratio of largest to smallest Hessian eigenvalue). Large $\kappa$ = elongated, "ill-conditioned" valley → slow GD.

### 11.2 Descent Lemma

For L-smooth $f$ and $\eta \leq 1/L$, one GD step satisfies:

```math
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \eta\left(1 - \frac{L\eta}{2}\right)\lVert\nabla f(\boldsymbol{\theta}_t)\rVert^2 \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert\nabla f(\boldsymbol{\theta}_t)\rVert^2
```

So the objective strictly decreases until the gradient is zero. The best step is $\eta = 1/L$.

### 11.3 Convergence Rates of Gradient Descent

| Setting | Step size | Rate | Iterations for $\epsilon$-accuracy |
|---|---|---|---|
| Convex, L-smooth | $\eta = 1/L$ | $f(\theta_T) - f^{\ast} \leq \dfrac{L\lVert\theta_0 - \theta^{\ast}\rVert^2}{2T}$ — **sublinear** $O(1/T)$ | $O(1/\epsilon)$ |
| Strongly convex, L-smooth | $\eta = 1/L$ | $\lVert\theta_T - \theta^{\ast}\rVert^2 \leq \left(1 - \dfrac{\mu}{L}\right)^T\lVert\theta_0 - \theta^{\ast}\rVert^2$ — **linear** (geometric) | $O(\kappa\log(1/\epsilon))$ |
| Non-convex, L-smooth | $\eta = 1/L$ | $\min_{t<T}\lVert\nabla f(\theta_t)\rVert^2 \leq \dfrac{2L(f(\theta_0) - f^{\ast})}{T}$ | $O(1/\epsilon)$ to a stationary point |
| SGD, convex | $\eta_t \propto 1/\sqrt{t}$ | $O(1/\sqrt{T})$ | $O(1/\epsilon^2)$ |
| SGD, strongly convex | $\eta_t \propto 1/(\mu t)$ | $O(1/T)$ | $O(1/\epsilon)$ |
| Nesterov (convex) | — | $O(1/T^2)$ — optimal for first-order methods | $O(1/\sqrt{\epsilon})$ |

**Contraction factor for quadratics** with optimal fixed step $\eta = 2/(L + \mu)$:

```math
\text{GD: } \rho = \frac{\kappa - 1}{\kappa + 1} \qquad\qquad \text{Heavy-ball momentum: } \rho = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}
```

Iterations needed to reduce error by factor $\epsilon$: $T = \ln(1/\epsilon)/\ln(1/\rho)$.

### 11.4 SGD Convergence Conditions

Because of gradient noise, SGD with a **constant** step only converges to a neighbourhood of the optimum. For exact convergence, the step sizes must satisfy the **Robbins–Monro conditions**:

```math
\sum_{t=1}^\infty\eta_t = \infty, \qquad \sum_{t=1}^\infty\eta_t^2 < \infty \qquad (\text{e.g., } \eta_t = \eta_0/t)
```

Common **learning-rate schedules:** step decay, exponential decay $\eta_t = \eta_0e^{-kt}$, $1/t$ decay, cosine annealing, warm-up.

### 11.5 Stability Condition for Quadratics

For $J(\theta) = \frac{a}{2}\theta^2$ ($a > 0$, so $L = a$): $\theta_{t+1} = \theta_t - \eta a\theta_t = (1 - \eta a)\theta_t$, hence $\theta_t = (1 - \eta a)^t\theta_0$.

```math
\text{Converges} \iff |1 - \eta a| < 1 \iff 0 < \eta < \frac{2}{a} = \frac{2}{L}
```

| Range of $\eta$ | Behaviour |
|---|---|
| $0 < \eta < 1/a$ | Monotonic convergence |
| $\eta = 1/a$ | Converges in **one** step |
| $1/a < \eta < 2/a$ | Oscillating convergence |
| $\eta = 2/a$ | Oscillates forever |
| $\eta > 2/a$ | **Diverges** |

### 📝 Solved Numerical 11.1 — Learning-rate Stability

$J(\theta) = \theta^2$ → $a = 2$, $L = 2$. Stable if $0 < \eta < 1$. $\theta_0 = 5$.

| $\eta$ | Factor $1 - 2\eta$ | $\theta_1$ | $\theta_2$ | $\theta_3$ | Behaviour |
|---|---|---|---|---|---|
| 0.1 | 0.8 | 4 | 3.2 | 2.56 | Slow monotonic convergence |
| 0.5 | 0 | 0 | 0 | 0 | One-step convergence |
| 0.9 | −0.8 | −4 | 3.2 | −2.56 | Oscillating convergence |
| 1.0 | −1 | −5 | 5 | −5 | Oscillates forever |
| 1.1 | −1.2 | −6 | 7.2 | −8.64 | **Diverges** |

### 📝 Solved Numerical 11.2 — Effect of Condition Number

Quadratic with $L = 100$, $\mu = 1$ → $\kappa = 100$. How many iterations to reduce error by $10^{-6}$?

**GD:**

```math
\rho = \frac{100 - 1}{100 + 1} = 0.9802, \qquad T = \frac{\ln(10^6)}{-\ln(0.9802)} = \frac{13.816}{0.0200} \approx 691
```

**Momentum:**

```math
\rho = \frac{\sqrt{100} - 1}{\sqrt{100} + 1} = \frac{9}{11} = 0.8182, \qquad T = \frac{13.816}{-\ln(0.8182)} = \frac{13.816}{0.2007} \approx 69
```

Momentum is about **10× faster** ($\approx\sqrt{\kappa}$ speed-up) on ill-conditioned problems.

### 📝 Solved Numerical 11.3 — Sublinear Rate Bound

Convex, L-smooth $f$ with $L = 4$, $\lVert\theta_0 - \theta^{\ast}\rVert = 3$, $\eta = 1/L = 0.25$. After $T = 100$ iterations:

```math
f(\theta_{100}) - f^{\ast} \leq \frac{4 \times 3^2}{2 \times 100} = \frac{36}{200} = 0.18
```

To guarantee $\leq 0.01$: $T \geq \frac{36}{2 \times 0.01} = 1800$ iterations.

💡 **Exam tip:** For convergence questions, identify $L$ (largest curvature) first; the safe step size is $\eta \leq 1/L$ and the divergence threshold is $\eta \geq 2/L$.

---

## 12. Formula Sheet

| Concept | Formula |
|---|---|
| Shattering | $\lvert\mathcal{H}_C\rvert = 2^{\lvert C\rvert}$ |
| VC dimension | $\max\lbrace\lvert C\rvert : \mathcal{H}\text{ shatters } C\rbrace$ |
| VC of hyperplanes in $\mathbb{R}^d$ | $d + 1$ |
| Sauer's lemma | $\Pi_{\mathcal{H}}(m) \leq \sum_{i=0}^d\binom{m}{i} \leq (em/d)^d$ |
| VC bound | $L_{\mathcal{D}} \leq L_S + \sqrt{\frac{h(\ln(2m/h)+1) + \ln(4/\delta)}{m}}$ |
| Blumer sample complexity | $m \geq \frac{1}{\epsilon}(4\log_2\frac{2}{\delta} + 8d\log_2\frac{13}{\epsilon})$ |
| Empirical Rademacher | $\hat{\mathcal{R}}_S = \mathbb{E}_\sigma[\sup_h\frac{1}{m}\sum\sigma_ih(x_i)]$ |
| Massart's lemma | $\hat{\mathcal{R}}_S \leq \sqrt{2\ln\lvert\mathcal{H}\rvert/m}$ |
| Linear class | $\hat{\mathcal{R}}_S \leq BR/\sqrt{m}$ |
| Rademacher bound | $L_{\mathcal{D}} \leq L_S + 2\mathcal{R}_m + \sqrt{\ln(1/\delta)/2m}$ |
| Bias | $\mathbb{E}[\hat{f}] - f$ |
| Variance | $\mathbb{E}[(\hat{f} - \mathbb{E}\hat{f})^2]$ |
| Decomposition | $\text{Err} = \text{Bias}^2 + \text{Var} + \sigma^2$ |
| KNN variance | $\sigma^2/k$ |
| Regularized loss | $J + \lambda\Omega(\mathbf{w})$ |
| Ridge | $\mathbf{w} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$ |
| Weight decay | $\mathbf{w} \leftarrow (1 - 2\eta\lambda)\mathbf{w} - \eta\nabla J$ |
| Lasso penalty | $\lambda\sum\lvert w_j\rvert$ |
| Soft-thresholding | $\text{sign}(z)\max(\lvert z\rvert - \lambda, 0)$ |
| Elastic net | $\lambda_1\lVert\mathbf{w}\rVert_1 + \lambda_2\lVert\mathbf{w}\rVert_2^2$ |
| k-fold CV | $CV_{(k)} = \frac{1}{k}\sum E_i$ |
| LOOCV (linear) | $\frac{1}{n}\sum\left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2$ |
| SRM | $\arg\min_i[L_S(\hat{h}_i) + \text{VC confidence}(h_i)]$ |
| GD | $\theta \leftarrow \theta - \eta\nabla J$ |
| Momentum | $v_t = \beta v_{t-1} + \eta g_t$; $\theta_t = \theta_{t-1} - v_t$ |
| NAG | gradient at $\theta - \beta v$ |
| AdaGrad | $G_t = G_{t-1} + g_t^2$; $\theta \leftarrow \theta - \eta g/\sqrt{G_t + \epsilon}$ |
| RMSprop | $E_t = \rho E_{t-1} + (1-\rho)g_t^2$; $\theta \leftarrow \theta - \eta g/\sqrt{E_t + \epsilon}$ |
| Adam | $\hat{m} = m/(1-\beta_1^t)$, $\hat{v} = v/(1-\beta_2^t)$, $\theta \leftarrow \theta - \eta\hat{m}/(\sqrt{\hat{v}}+\epsilon)$ |
| Descent lemma | $f(\theta_{t+1}) \leq f(\theta_t) - \frac{\eta}{2}\lVert\nabla f\rVert^2$ ($\eta \leq 1/L$) |
| Convex rate | $f(\theta_T) - f^{\ast} \leq L\lVert\theta_0 - \theta^{\ast}\rVert^2/2T$ |
| Strongly convex rate | $(1 - \mu/L)^T$ |
| Condition number | $\kappa = L/\mu$ |
| GD vs momentum factor | $\frac{\kappa - 1}{\kappa + 1}$ vs $\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$ |
| Stability (quadratic) | $0 < \eta < 2/L$ |
| Robbins–Monro | $\sum\eta_t = \infty$, $\sum\eta_t^2 < \infty$ |

---

[⬅ Unit V](Unit-5-Reinforcement-Learning.md) | [Back to Index](README.md)
