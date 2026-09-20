# Unit IV — Regression & Clustering

[⬅ Back to Index](README.md)

## Contents

**Part A — Regression**
1. [Classification vs Regression](#1-classification-vs-regression)
2. [Simple Linear Regression](#2-simple-linear-regression)
3. [Multiple Linear Regression](#3-multiple-linear-regression)
4. [Regression Evaluation Metrics](#4-regression-evaluation-metrics)
5. [Linear vs Non-linear Regression](#5-linear-vs-non-linear-regression)
6. [Loss Functions for Regression](#6-loss-functions-for-regression)
7. [Non-parametric Regression](#7-non-parametric-regression)

**Part B — Clustering**

8. [Classification vs Clustering](#8-classification-vs-clustering)
9. [Similarity and Distance Measures](#9-similarity-and-distance-measures)
10. [Partition-based Clustering](#10-partition-based-clustering)
11. [Hierarchical Clustering](#11-hierarchical-clustering)
12. [Cluster Validation and Evaluation](#12-cluster-validation-and-evaluation)
13. [Formula Sheet](#13-formula-sheet)

---

# Part A — Regression

## 1. Classification vs Regression

| Aspect | Classification | Regression |
|---|---|---|
| Output type | Discrete / categorical label | Continuous real value |
| Question answered | "Which class?" | "How much? How many?" |
| Example | Will it rain tomorrow? (Yes/No) | How many mm of rain tomorrow? |
| Output of model | Class label or class probability | Real number |
| Decision boundary | Separates classes | Best-fit line/curve through data |
| Loss functions | Cross-entropy, hinge, 0–1 loss | MSE, MAE, Huber |
| Evaluation metrics | Accuracy, precision, recall, F1, AUC | MSE, RMSE, MAE, $R^2$ |
| Algorithms | Logistic regression, SVM, Naïve Bayes, KNN classifier | Linear regression, SVR, KNN regressor, regression trees |

> **Note:** Logistic *regression* is a **classification** algorithm — it regresses the log-odds but outputs a class.

---

## 2. Simple Linear Regression

### 2.1 Model

```math
y = \beta_0 + \beta_1 x + \varepsilon
```

- $\beta_0$ = intercept, $\beta_1$ = slope, $\varepsilon$ = random error (assumed $\mathcal{N}(0, \sigma^2)$).
- Prediction: $\hat{y} = \beta_0 + \beta_1 x$. Residual: $e_i = y_i - \hat{y}_i$.

### 2.2 Assumptions (LINE)

1. **L**inearity — relationship between $x$ and $y$ is linear.
2. **I**ndependence — errors are independent.
3. **N**ormality — errors are normally distributed.
4. **E**qual variance (homoscedasticity) — constant error variance.

### 2.3 Ordinary Least Squares (OLS) — Derivation

Minimise the sum of squared errors:

```math
J(\beta_0, \beta_1) = \sum_{i=1}^n\left(y_i - \beta_0 - \beta_1 x_i\right)^2
```

Set partial derivatives to zero:

```math
\frac{\partial J}{\partial\beta_0} = -2\sum_{i=1}^n(y_i - \beta_0 - \beta_1x_i) = 0 \quad\Rightarrow\quad \beta_0 = \bar{y} - \beta_1\bar{x}
```

```math
\frac{\partial J}{\partial\beta_1} = -2\sum_{i=1}^n x_i(y_i - \beta_0 - \beta_1x_i) = 0
```

Substituting $\beta_0$ and simplifying:

```math
\beta_1 = \frac{\sum_{i=1}^n(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n(x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}} = \frac{n\sum x_iy_i - \sum x_i\sum y_i}{n\sum x_i^2 - \left(\sum x_i\right)^2}
```

```math
\beta_0 = \bar{y} - \beta_1\bar{x}
```

Also: $\beta_1 = r\,\dfrac{s_y}{s_x}$, where $r$ is the Pearson correlation coefficient:

```math
r = \frac{S_{xy}}{\sqrt{S_{xx}\,S_{yy}}}
```

### 📝 Solved Numerical 2.1 — Fit a Regression Line

| $x$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $y$ | 2 | 4 | 5 | 4 | 5 |

**Step 1 — means:** $\bar{x} = 15/5 = 3$, $\bar{y} = 20/5 = 4$.

**Step 2 — table:**

| $x_i$ | $y_i$ | $x_i-\bar{x}$ | $y_i-\bar{y}$ | $(x_i-\bar{x})(y_i-\bar{y})$ | $(x_i-\bar{x})^2$ | $(y_i-\bar{y})^2$ |
|---|---|---|---|---|---|---|
| 1 | 2 | −2 | −2 | 4 | 4 | 4 |
| 2 | 4 | −1 | 0 | 0 | 1 | 0 |
| 3 | 5 | 0 | 1 | 0 | 0 | 1 |
| 4 | 4 | 1 | 0 | 0 | 1 | 0 |
| 5 | 5 | 2 | 1 | 2 | 4 | 1 |
| | | | **Sum** | $S_{xy} = 6$ | $S_{xx} = 10$ | $S_{yy} = 6$ |

**Step 3 — coefficients:**

```math
\beta_1 = \frac{6}{10} = 0.6, \qquad \beta_0 = 4 - 0.6(3) = 2.2
```

**Regression line:** $\hat{y} = 2.2 + 0.6x$.

**Step 4 — predict** $x = 6$: $\hat{y} = 2.2 + 3.6 = 5.8$.

**Step 5 — correlation:**

```math
r = \frac{6}{\sqrt{10 \times 6}} = \frac{6}{7.746} = 0.775
```

(Continued in Numerical 4.1 for error metrics.)

### 2.4 Gradient Descent for Linear Regression

Cost (MSE): $J = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$ with $\hat{y}_i = wx_i + b$.

```math
\frac{\partial J}{\partial w} = -\frac{2}{n}\sum_{i=1}^n x_i(y_i - \hat{y}_i), \qquad \frac{\partial J}{\partial b} = -\frac{2}{n}\sum_{i=1}^n(y_i - \hat{y}_i)
```

```math
w \leftarrow w - \eta\frac{\partial J}{\partial w}, \qquad b \leftarrow b - \eta\frac{\partial J}{\partial b}
```

### 📝 Solved Numerical 2.2 — One Gradient Descent Step

Same data, start $w = 0$, $b = 0$, $\eta = 0.01$. All predictions are 0, so residuals $= y = (2, 4, 5, 4, 5)$.

```math
\sum x_i y_i = 2 + 8 + 15 + 16 + 25 = 66, \qquad \sum y_i = 20
```

```math
\frac{\partial J}{\partial w} = -\frac{2}{5}(66) = -26.4, \qquad \frac{\partial J}{\partial b} = -\frac{2}{5}(20) = -8
```

```math
w = 0 - 0.01(-26.4) = 0.264, \qquad b = 0 - 0.01(-8) = 0.08
```

Repeating many steps converges to $w = 0.6$, $b = 2.2$.

---

## 3. Multiple Linear Regression

### 3.1 Model

```math
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_dx_d + \varepsilon
```

In matrix form with design matrix $X$ ($n \times (d+1)$, first column all 1's):

```math
\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}
```

### 3.2 Normal Equation

Minimise $J(\boldsymbol{\beta}) = (\mathbf{y} - X\boldsymbol{\beta})^T(\mathbf{y} - X\boldsymbol{\beta})$:

```math
\nabla_{\boldsymbol{\beta}}J = -2X^T\mathbf{y} + 2X^TX\boldsymbol{\beta} = 0
```

```math
\boldsymbol{\beta} = (X^TX)^{-1}X^T\mathbf{y}
```

**Complexity:** $O(d^3)$ for the inverse — slow for very many features; then use gradient descent. If $X^TX$ is singular (multicollinearity), use the pseudo-inverse or ridge regression.

### 📝 Solved Numerical 3.1 — Normal Equation

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 1 | 1 | 6 |
| 2 | 1 | 8 |
| 2 | 2 | 9 |
| 3 | 2 | 11 |

```math
X = \begin{bmatrix}1&1&1\\1&2&1\\1&2&2\\1&3&2\end{bmatrix}, \qquad \mathbf{y} = \begin{bmatrix}6\\8\\9\\11\end{bmatrix}
```

```math
X^TX = \begin{bmatrix}4&8&6\\8&18&13\\6&13&10\end{bmatrix}, \qquad X^T\mathbf{y} = \begin{bmatrix}34\\73\\54\end{bmatrix}
```

```math
(X^TX)^{-1} = \begin{bmatrix}2.75&-0.5&-1\\-0.5&1&-1\\-1&-1&2\end{bmatrix}
```

```math
\boldsymbol{\beta} = \begin{bmatrix}2.75(34) - 0.5(73) - 1(54)\\ -0.5(34) + 1(73) - 1(54)\\ -1(34) - 1(73) + 2(54)\end{bmatrix} = \begin{bmatrix}93.5 - 36.5 - 54\\ -17 + 73 - 54\\ -34 - 73 + 108\end{bmatrix} = \begin{bmatrix}3\\2\\1\end{bmatrix}
```

**Model:** $\hat{y} = 3 + 2x_1 + x_2$ (fits all 4 points exactly).

---

## 4. Regression Evaluation Metrics

```math
\text{MSE} = \frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2 \qquad \text{RMSE} = \sqrt{\text{MSE}} \qquad \text{MAE} = \frac{1}{n}\sum_{i=1}^n|y_i - \hat{y}_i|
```

```math
\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^n\left|\frac{y_i - \hat{y}_i}{y_i}\right|
```

**Sums of squares:**

```math
SST = \sum(y_i - \bar{y})^2, \qquad SSR = \sum(\hat{y}_i - \bar{y})^2, \qquad SSE = \sum(y_i - \hat{y}_i)^2, \qquad SST = SSR + SSE
```

**Coefficient of determination:**

```math
R^2 = 1 - \frac{SSE}{SST} = \frac{SSR}{SST}
```

$R^2$ = fraction of variance in $y$ explained by the model (0 to 1 on training data for OLS). For simple linear regression, $R^2 = r^2$.

**Adjusted $R^2$** (penalises adding useless features; $p$ = number of predictors):

```math
R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
```

### 📝 Solved Numerical 4.1 — Metrics for the Fitted Line

Using $\hat{y} = 2.2 + 0.6x$:

| $x$ | $y$ | $\hat{y}$ | $e = y - \hat{y}$ | $e^2$ | $\lvert e\rvert$ |
|---|---|---|---|---|---|
| 1 | 2 | 2.8 | −0.8 | 0.64 | 0.8 |
| 2 | 4 | 3.4 | 0.6 | 0.36 | 0.6 |
| 3 | 5 | 4.0 | 1.0 | 1.00 | 1.0 |
| 4 | 4 | 4.6 | −0.6 | 0.36 | 0.6 |
| 5 | 5 | 5.2 | −0.2 | 0.04 | 0.2 |
| | | | **Sum** | **2.40** | **3.2** |

```math
SSE = 2.4, \quad \text{MSE} = \frac{2.4}{5} = 0.48, \quad \text{RMSE} = \sqrt{0.48} = 0.693, \quad \text{MAE} = \frac{3.2}{5} = 0.64
```

```math
R^2 = 1 - \frac{2.4}{6} = 0.6 \qquad (\text{check: } r^2 = 0.775^2 = 0.6\ ✓)
```

```math
R^2_{adj} = 1 - \frac{(1 - 0.6)(5 - 1)}{5 - 1 - 1} = 1 - \frac{1.6}{3} = 0.467
```

---

## 5. Linear vs Non-linear Regression

### 5.1 What does "linear" mean?

A model is **linear** if it is linear in its **parameters** $\boldsymbol{\beta}$ — not necessarily in $x$.

| Model | Linear in parameters? | Type |
|---|---|---|
| $y = \beta_0 + \beta_1x$ | Yes | Linear |
| $y = \beta_0 + \beta_1x + \beta_2x^2$ | Yes | Linear (polynomial) |
| $y = \beta_0 + \beta_1\ln x$ | Yes | Linear |
| $y = \beta_0 e^{\beta_1 x}$ | No | Non-linear (but linearisable) |
| $y = \frac{\beta_0}{1 + e^{-\beta_1(x-\beta_2)}}$ | No | Non-linear (logistic growth) |

### 5.2 Comparison

| Aspect | Linear Regression | Non-linear Regression |
|---|---|---|
| Relationship | Straight line / hyperplane | Curve |
| Solution | Closed form (normal equation) | Iterative (Gauss–Newton, Levenberg–Marquardt, GD) |
| Interpretability | High | Lower |
| Risk | Underfitting complex data | Overfitting, local minima, needs good initial guesses |
| Examples | Salary vs experience | Population growth, drug dose–response |

### 5.3 Polynomial Regression

```math
\hat{y} = \beta_0 + \beta_1x + \beta_2x^2 + \dots + \beta_px^p
```

Create features $(x, x^2, \dots, x^p)$ and apply ordinary linear regression. High degree $p$ → overfitting (wiggly curve).

### 5.4 Linearisation by Transformation

**Exponential model** $y = ae^{bx}$ → take natural log:

```math
\ln y = \ln a + bx \quad\Rightarrow\quad Y = A + bx \quad (Y = \ln y,\ A = \ln a)
```

**Power model** $y = ax^b$ → $\ln y = \ln a + b\ln x$.

### 📝 Solved Numerical 5.1 — Exponential Fit via Log Transform

| $x$ | 0 | 1 | 2 |
|---|---|---|---|
| $y$ | 2 | 5.4 | 14.8 |
| $Y = \ln y$ | 0.693 | 1.686 | 2.695 |

$\bar{x} = 1$, $\bar{Y} = (0.693 + 1.686 + 2.695)/3 = 1.691$.

```math
b = \frac{(-1)(0.693 - 1.691) + 0 + (1)(2.695 - 1.691)}{1 + 0 + 1} = \frac{0.998 + 1.004}{2} = 1.001
```

```math
A = 1.691 - 1.001(1) = 0.690, \qquad a = e^{0.690} = 1.99
```

**Model:** $y \approx 2e^{x}$.

---

## 6. Loss Functions for Regression

Let residual $r = y - \hat{y}$.

### 6.1 Mean Squared Error (L2 loss)

```math
L_{MSE} = \frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2, \qquad \frac{\partial}{\partial\hat{y}}(y - \hat{y})^2 = -2(y - \hat{y})
```

- Smooth and differentiable everywhere; convex.
- Penalises large errors heavily → **sensitive to outliers**.
- Minimiser of $\sum(y_i - c)^2$ is the **mean**.

### 6.2 Mean Absolute Error (L1 loss)

```math
L_{MAE} = \frac{1}{n}\sum_{i=1}^n|y_i - \hat{y}_i|, \qquad \frac{\partial}{\partial\hat{y}}|y - \hat{y}| = -\text{sign}(y - \hat{y})
```

- **Robust to outliers**; not differentiable at 0.
- Minimiser of $\sum|y_i - c|$ is the **median**.

### 6.3 Huber Loss (best of both)

```math
L_\delta(r) = \begin{cases}\frac{1}{2}r^2 & \text{if } |r| \leq \delta\\ \delta\left(|r| - \frac{1}{2}\delta\right) & \text{if } |r| > \delta\end{cases}
```

Quadratic for small errors, linear for large errors. $\delta$ is a hyperparameter.

### 6.4 Log-Cosh Loss

```math
L = \sum_{i=1}^n\ln\left(\cosh(\hat{y}_i - y_i)\right)
```

≈ $r^2/2$ for small $r$, ≈ $|r| - \ln 2$ for large $r$; twice differentiable.

### 6.5 Quantile (Pinball) Loss

For quantile $\tau\in(0,1)$:

```math
L_\tau(r) = \begin{cases}\tau\,r & \text{if } r \geq 0\\ (\tau - 1)\,r & \text{if } r < 0\end{cases}
```

Used to predict intervals ($\tau = 0.5$ gives MAE/2 → median regression).

### 6.6 Comparison

| Loss | Outlier robustness | Differentiable | Optimal constant |
|---|---|---|---|
| MSE | Poor | Yes | Mean |
| MAE | Good | Not at 0 | Median |
| Huber | Good | Yes (once) | Between mean & median |
| Log-cosh | Good | Yes (twice) | — |
| Quantile | Good | Not at 0 | $\tau$-th quantile |

### 📝 Solved Numerical 6.1 — Compare Losses with an Outlier

Residuals: $r = (1, -2, 0.5, 6)$ (the 6 is an outlier).

**MSE:**

```math
\frac{1 + 4 + 0.25 + 36}{4} = \frac{41.25}{4} = 10.3125, \qquad \text{RMSE} = 3.211
```

**MAE:**

```math
\frac{1 + 2 + 0.5 + 6}{4} = \frac{9.5}{4} = 2.375
```

**Huber ($\delta = 1.5$):**

| $r$ | $\lvert r\rvert \leq 1.5$? | Loss |
|---|---|---|
| 1 | Yes | $0.5(1)^2 = 0.5$ |
| −2 | No | $1.5(2 - 0.75) = 1.875$ |
| 0.5 | Yes | $0.5(0.25) = 0.125$ |
| 6 | No | $1.5(6 - 0.75) = 7.875$ |

```math
\text{Huber} = \frac{0.5 + 1.875 + 0.125 + 7.875}{4} = \frac{10.375}{4} = 2.594
```

**Quantile ($\tau = 0.9$):** $0.9(1) = 0.9$; $(-0.1)(-2) = 0.2$; $0.9(0.5) = 0.45$; $0.9(6) = 5.4$ → mean $= 6.95/4 = 1.7375$.

The outlier contributes 36/41.25 = **87% of the MSE** but only 6/9.5 = 63% of the MAE — MSE is dominated by outliers.

---

## 7. Non-parametric Regression

No fixed functional form; the model complexity grows with the data.

### 7.1 KNN Regression

```math
\hat{y}(x_0) = \frac{1}{k}\sum_{i\in N_k(x_0)}y_i
```

(See Unit III, Numerical 2.2.)

### 7.2 Kernel Regression (Nadaraya–Watson Estimator)

Weighted average of all training targets, weights from a kernel that decreases with distance:

```math
\hat{y}(x_0) = \frac{\sum_{i=1}^n K\left(\frac{x_0 - x_i}{h}\right)y_i}{\sum_{i=1}^n K\left(\frac{x_0 - x_i}{h}\right)}
```

**Gaussian kernel:**

```math
K(u) = \exp\left(-\frac{u^2}{2}\right) \quad\Rightarrow\quad K\left(\frac{x_0 - x_i}{h}\right) = \exp\left(-\frac{(x_0 - x_i)^2}{2h^2}\right)
```

**Epanechnikov kernel:** $K(u) = \frac{3}{4}(1 - u^2)$ for $|u| \leq 1$, else 0.

**Bandwidth $h$:** small $h$ → wiggly (high variance); large $h$ → overly smooth (high bias).

### 📝 Solved Numerical 7.1 — Nadaraya–Watson

Data $x = (1,2,3,4,5)$, $y = (2,4,5,4,5)$. Predict at $x_0 = 2.5$ with Gaussian kernel, $h = 1$.

| $x_i$ | $(x_0 - x_i)^2$ | $K_i = e^{-(x_0-x_i)^2/2}$ | $K_iy_i$ |
|---|---|---|---|
| 1 | 2.25 | 0.3247 | 0.6494 |
| 2 | 0.25 | 0.8825 | 3.5300 |
| 3 | 0.25 | 0.8825 | 4.4125 |
| 4 | 2.25 | 0.3247 | 1.2988 |
| 5 | 6.25 | 0.0439 | 0.2195 |
| **Sum** | | **2.4583** | **10.1102** |

```math
\hat{y}(2.5) = \frac{10.1102}{2.4583} = 4.113
```

(Linear regression would predict $2.2 + 0.6(2.5) = 3.7$.)

### 7.3 Other Non-parametric Methods

- **Regression Trees (CART):** piecewise-constant predictions (Unit III, §7).
- **LOESS / LOWESS (Locally Weighted Regression):** at each query point, fit a weighted linear regression using nearby points:

```math
\min_{\beta_0,\beta_1}\sum_{i=1}^n K\left(\frac{x_0 - x_i}{h}\right)\left(y_i - \beta_0 - \beta_1x_i\right)^2
```

- **Splines:** piecewise polynomials joined smoothly at knots.
- **Support Vector Regression (with RBF kernel), Gaussian Process Regression.**

---

# Part B — Clustering

## 8. Classification vs Clustering

| Aspect | Classification | Clustering |
|---|---|---|
| Learning type | Supervised | Unsupervised |
| Labels | Known, predefined classes | No labels; groups discovered |
| Goal | Predict class of new data | Group similar data together |
| Number of groups | Fixed by the data | Often chosen by user ($k$) or discovered |
| Evaluation | Accuracy, F1 against true labels | Internal indices (silhouette, SSE); external if labels exist |
| Examples | Spam filter, disease diagnosis | Customer segmentation, document grouping, image segmentation |
| Algorithms | Decision tree, SVM, Naïve Bayes | K-Means, Hierarchical, DBSCAN |

**Goal of clustering:** high **intra-cluster similarity** (compact clusters) and low **inter-cluster similarity** (well-separated clusters).

---

## 9. Similarity and Distance Measures

### 9.1 Properties of a Distance Metric

1. **Non-negativity:** $d(x, y) \geq 0$
2. **Identity:** $d(x, y) = 0 \iff x = y$
3. **Symmetry:** $d(x, y) = d(y, x)$
4. **Triangle inequality:** $d(x, z) \leq d(x, y) + d(y, z)$

### 9.2 Numeric Data

```math
\text{Euclidean: } d(\mathbf{x},\mathbf{y}) = \sqrt{\sum_{j=1}^d(x_j - y_j)^2}
```

```math
\text{Manhattan: } d(\mathbf{x},\mathbf{y}) = \sum_{j=1}^d|x_j - y_j|
```

```math
\text{Minkowski: } d(\mathbf{x},\mathbf{y}) = \left(\sum_{j=1}^d|x_j - y_j|^p\right)^{1/p}
```

```math
\text{Chebyshev: } d(\mathbf{x},\mathbf{y}) = \max_j|x_j - y_j|
```

**Mahalanobis** (accounts for correlation and scale; $\Sigma$ = covariance matrix):

```math
d_M(\mathbf{x},\mathbf{y}) = \sqrt{(\mathbf{x}-\mathbf{y})^T\Sigma^{-1}(\mathbf{x}-\mathbf{y})}
```

### 9.3 Similarity Measures

**Cosine similarity** (text/documents — ignores magnitude):

```math
\cos\theta = \frac{\mathbf{x}\cdot\mathbf{y}}{\lVert\mathbf{x}\rVert\,\lVert\mathbf{y}\rVert} = \frac{\sum_j x_jy_j}{\sqrt{\sum_j x_j^2}\sqrt{\sum_j y_j^2}}, \qquad d_{\cos} = 1 - \cos\theta
```

**Pearson correlation:**

```math
\rho(\mathbf{x},\mathbf{y}) = \frac{\sum_j(x_j - \bar{x})(y_j - \bar{y})}{\sqrt{\sum_j(x_j - \bar{x})^2}\sqrt{\sum_j(y_j - \bar{y})^2}}
```

### 9.4 Binary / Set Data

For two binary vectors, let $M_{11}$ = both 1, $M_{00}$ = both 0, $M_{10}$, $M_{01}$ = mismatches.

```math
\text{Simple Matching Coefficient: } SMC = \frac{M_{11} + M_{00}}{M_{11} + M_{00} + M_{10} + M_{01}}
```

```math
\text{Jaccard: } J(A, B) = \frac{|A\cap B|}{|A\cup B|} = \frac{M_{11}}{M_{11} + M_{10} + M_{01}}, \qquad d_J = 1 - J
```

**Hamming distance:** number of positions in which two strings/vectors differ.

### 📝 Solved Numerical 9.1 — All Distances

$\mathbf{A} = (1, 2, 3)$, $\mathbf{B} = (4, 6, 8)$. Differences: $(3, 4, 5)$.

```math
\text{Euclidean} = \sqrt{9 + 16 + 25} = \sqrt{50} = 7.071
```

```math
\text{Manhattan} = 3 + 4 + 5 = 12
```

```math
\text{Minkowski } (p=3) = (27 + 64 + 125)^{1/3} = 216^{1/3} = 6
```

```math
\text{Chebyshev} = \max(3, 4, 5) = 5
```

Notice: Manhattan (12) ≥ Euclidean (7.07) ≥ Minkowski p=3 (6) ≥ Chebyshev (5).

```math
\cos\theta = \frac{1(4) + 2(6) + 3(8)}{\sqrt{14}\sqrt{116}} = \frac{40}{3.742 \times 10.770} = \frac{40}{40.30} = 0.9926
```

Cosine distance $= 0.0074$ → almost the same direction.

### 📝 Solved Numerical 9.2 — Jaccard, SMC, Hamming

$\mathbf{p} = (1, 0, 1, 1, 0, 0, 1)$, $\mathbf{q} = (1, 0, 0, 1, 0, 1, 1)$.

Position-wise: $M_{11} = 3$ (positions 1, 4, 7), $M_{00} = 2$ (positions 2, 5), $M_{10} = 1$ (position 3), $M_{01} = 1$ (position 6).

```math
SMC = \frac{3 + 2}{7} = 0.714, \qquad J = \frac{3}{3 + 1 + 1} = 0.6, \qquad \text{Hamming} = 1 + 1 = 2
```

### 📝 Solved Numerical 9.3 — Mahalanobis vs Euclidean

$\Sigma = \begin{bmatrix}4 & 0\\0 & 1\end{bmatrix}$, point $\mathbf{x} = (2, 1)$, mean $\boldsymbol{\mu} = (0, 0)$.

```math
d_M = \sqrt{\begin{bmatrix}2 & 1\end{bmatrix}\begin{bmatrix}1/4 & 0\\0 & 1\end{bmatrix}\begin{bmatrix}2\\1\end{bmatrix}} = \sqrt{\frac{4}{4} + 1} = \sqrt{2} = 1.414
```

Euclidean $= \sqrt{5} = 2.236$. Feature 1 has larger variance, so its deviation counts less in Mahalanobis distance.

---

## 10. Partition-based Clustering

Divide $n$ objects into $k$ non-overlapping clusters, optimising an objective.

### 10.1 K-Means Algorithm

**Objective — Within-Cluster Sum of Squares (WCSS / SSE / inertia):**

```math
J = \sum_{j=1}^{k}\sum_{\mathbf{x}_i\in C_j}\lVert\mathbf{x}_i - \boldsymbol{\mu}_j\rVert^2
```

**Algorithm (Lloyd's):**
1. Choose $k$; initialise $k$ centroids (randomly or with K-Means++).
2. **Assignment step:** assign each point to the nearest centroid:

```math
C_j = \lbrace\mathbf{x}_i : \lVert\mathbf{x}_i - \boldsymbol{\mu}_j\rVert \leq \lVert\mathbf{x}_i - \boldsymbol{\mu}_l\rVert\ \ \forall l\rbrace
```

3. **Update step:** recompute each centroid as the mean of its points:

```math
\boldsymbol{\mu}_j = \frac{1}{|C_j|}\sum_{\mathbf{x}_i\in C_j}\mathbf{x}_i
```

4. Repeat 2–3 until assignments no longer change (convergence).

**Properties:**
- Each step never increases $J$ → guaranteed to converge (to a **local** minimum).
- Time complexity: $O(nkdI)$ ($I$ = iterations).
- Assumes spherical, similar-size clusters; sensitive to initialisation, outliers, and scale; needs $k$ in advance.

### 10.2 K-Means++ Initialisation

1. Pick the first centroid uniformly at random from the data.
2. For each point, compute $D(\mathbf{x})$ = distance to nearest chosen centroid.
3. Choose the next centroid with probability proportional to $D(\mathbf{x})^2$:

```math
P(\mathbf{x}_i) = \frac{D(\mathbf{x}_i)^2}{\sum_l D(\mathbf{x}_l)^2}
```

4. Repeat until $k$ centroids are chosen. Spreads out initial centroids → faster, better convergence.

### 10.3 Choosing k — Elbow Method

Plot WCSS vs $k$. WCSS always decreases with $k$; choose the $k$ at the "elbow" where the decrease sharply slows down. (Silhouette score can also be used — pick $k$ with the highest average silhouette.)

### 10.4 K-Medoids (PAM — Partitioning Around Medoids)

- Cluster centre must be an **actual data point** (medoid).
- Minimises the sum of dissimilarities (any distance, e.g., Manhattan):

```math
J = \sum_{j=1}^k\sum_{\mathbf{x}_i\in C_j}d(\mathbf{x}_i, \mathbf{m}_j)
```

- **More robust to outliers** than K-Means; more expensive: $O(k(n-k)^2)$ per iteration.

### 📝 Solved Numerical 10.1 — K-Means (k = 2)

| Point | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| $x$ | 1 | 1.5 | 2.9 | 5 | 3.5 | 4.5 | 3.5 |
| $y$ | 1 | 2 | 4 | 7 | 5 | 5 | 4.5 |

Initial centroids: $\boldsymbol{\mu}_1 = A = (1, 1)$, $\boldsymbol{\mu}_2 = D = (5, 7)$.

**Iteration 1 — Assignment:**

| Point | $d$ to $\mu_1 (1,1)$ | $d$ to $\mu_2 (5,7)$ | Cluster |
|---|---|---|---|
| A (1, 1) | 0 | 7.211 | 1 |
| B (1.5, 2) | 1.118 | 6.103 | 1 |
| C (2.9, 4) | 3.551 | 3.662 | 1 |
| D (5, 7) | 7.211 | 0 | 2 |
| E (3.5, 5) | 4.717 | 2.500 | 2 |
| F (4.5, 5) | 5.315 | 2.062 | 2 |
| G (3.5, 4.5) | 4.301 | 2.915 | 2 |

Example: $d(C, \mu_1) = \sqrt{(2.9-1)^2 + (4-1)^2} = \sqrt{3.61 + 9} = 3.551$.

**Update:**

```math
\boldsymbol{\mu}_1 = \left(\frac{1 + 1.5 + 2.9}{3}, \frac{1 + 2 + 4}{3}\right) = (1.8,\ 2.333)
```

```math
\boldsymbol{\mu}_2 = \left(\frac{5 + 3.5 + 4.5 + 3.5}{4}, \frac{7 + 5 + 5 + 4.5}{4}\right) = (4.125,\ 5.375)
```

**Iteration 2 — Assignment:**

| Point | $d$ to $\mu_1 (1.8, 2.333)$ | $d$ to $\mu_2 (4.125, 5.375)$ | Cluster |
|---|---|---|---|
| A | 1.555 | 5.376 | 1 |
| B | 0.448 | 4.276 | 1 |
| **C** | **1.997** | **1.842** | **2** ← moved! |
| D | 5.658 | 1.846 | 2 |
| E | 3.162 | 0.729 | 2 |
| F | 3.795 | 0.530 | 2 |
| G | 2.754 | 1.075 | 2 |

**Update:**

```math
\boldsymbol{\mu}_1 = \left(\frac{1 + 1.5}{2}, \frac{1 + 2}{2}\right) = (1.25,\ 1.5)
```

```math
\boldsymbol{\mu}_2 = \left(\frac{2.9 + 5 + 3.5 + 4.5 + 3.5}{5}, \frac{4 + 7 + 5 + 5 + 4.5}{5}\right) = (3.88,\ 5.1)
```

**Iteration 3:** re-assigning gives the same clusters → **converged**.

**Final clusters:** $C_1 = \lbrace A, B\rbrace$, $C_2 = \lbrace C, D, E, F, G\rbrace$. WCSS drops from 11.98 (after iteration 1) to **8.71** (final).

---

## 11. Hierarchical Clustering

Builds a **tree of clusters (dendrogram)**. No need to specify $k$ in advance — cut the dendrogram at the desired level.

### 11.1 Types

- **Agglomerative (bottom-up, AGNES):** start with each point as its own cluster; repeatedly merge the two closest clusters until one cluster remains.
- **Divisive (top-down, DIANA):** start with all points in one cluster; recursively split.

### 11.2 Agglomerative Algorithm

1. Compute the $n \times n$ distance matrix.
2. Treat each point as a cluster.
3. Repeat: merge the two closest clusters; update the distance matrix using a **linkage** criterion.
4. Stop when one cluster remains. Complexity $O(n^2\log n)$ to $O(n^3)$; memory $O(n^2)$.

### 11.3 Linkage Criteria

```math
\text{Single (MIN): } d(C_i, C_j) = \min_{\mathbf{x}\in C_i,\,\mathbf{y}\in C_j}d(\mathbf{x},\mathbf{y})
```

```math
\text{Complete (MAX): } d(C_i, C_j) = \max_{\mathbf{x}\in C_i,\,\mathbf{y}\in C_j}d(\mathbf{x},\mathbf{y})
```

```math
\text{Average (UPGMA): } d(C_i, C_j) = \frac{1}{|C_i||C_j|}\sum_{\mathbf{x}\in C_i}\sum_{\mathbf{y}\in C_j}d(\mathbf{x},\mathbf{y})
```

```math
\text{Centroid: } d(C_i, C_j) = \lVert\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\rVert
```

```math
\text{Ward: } \Delta(C_i, C_j) = \frac{|C_i||C_j|}{|C_i| + |C_j|}\lVert\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\rVert^2 \quad \text{(increase in total WCSS after merging)}
```

**Lance–Williams update formula** (general form for updating distances after merging $C_i$ and $C_j$ into $C_{ij}$):

```math
d(C_k, C_{ij}) = \alpha_i\,d(C_k, C_i) + \alpha_j\,d(C_k, C_j) + \beta\,d(C_i, C_j) + \gamma\,|d(C_k, C_i) - d(C_k, C_j)|
```

(Single: $\alpha_i = \alpha_j = 1/2, \beta = 0, \gamma = -1/2$; Complete: $\gamma = +1/2$.)

| Linkage | Behaviour |
|---|---|
| Single | Can find elongated shapes; suffers from **chaining** effect; sensitive to noise |
| Complete | Compact, similar-diameter clusters; sensitive to outliers |
| Average | Compromise between single and complete |
| Ward | Minimises variance; similar to K-Means; spherical clusters |

### 📝 Solved Numerical 11.1 — Agglomerative Clustering (Single & Complete)

Points on a line: A = 1, B = 2, C = 4.5, D = 8, E = 9.5.

**Distance matrix:**

|  | A | B | C | D | E |
|---|---|---|---|---|---|
| **A** | 0 | | | | |
| **B** | 1 | 0 | | | |
| **C** | 3.5 | 2.5 | 0 | | |
| **D** | 7 | 6 | 3.5 | 0 | |
| **E** | 8.5 | 7.5 | 5 | 1.5 | 0 |

#### Single Linkage

**Step 1:** smallest distance = 1 (A, B) → merge **{A,B}** at height 1.

|  | AB | C | D | E |
|---|---|---|---|---|
| **AB** | 0 | | | |
| **C** | min(3.5, 2.5) = 2.5 | 0 | | |
| **D** | min(7, 6) = 6 | 3.5 | 0 | |
| **E** | min(8.5, 7.5) = 7.5 | 5 | 1.5 | 0 |

**Step 2:** smallest = 1.5 (D, E) → merge **{D,E}** at height 1.5.

|  | AB | C | DE |
|---|---|---|---|
| **AB** | 0 | | |
| **C** | 2.5 | 0 | |
| **DE** | min(6, 7.5) = 6 | min(3.5, 5) = 3.5 | 0 |

**Step 3:** smallest = 2.5 → merge **{A,B,C}** at height 2.5.

**Step 4:** $d(ABC, DE) = \min(6, 3.5) = 3.5$ → merge all at height 3.5.

#### Complete Linkage

**Step 1:** merge {A,B} at 1. Updated: $d(AB, C) = \max(3.5, 2.5) = 3.5$, $d(AB, D) = 7$, $d(AB, E) = 8.5$.

**Step 2:** merge {D,E} at 1.5. Updated: $d(AB, DE) = \max(7, 8.5) = 8.5$, $d(C, DE) = \max(3.5, 5) = 5$.

**Step 3:** smallest among {3.5, 8.5, 5} = 3.5 → merge {A,B,C} at 3.5.

**Step 4:** $d(ABC, DE) = \max(8.5, 5) = 8.5$ → merge all at 8.5.

#### Average Linkage (for comparison)

After {A,B} (1) and {D,E} (1.5): $d(AB, C) = (3.5 + 2.5)/2 = 3.0$; $d(C, DE) = (3.5 + 5)/2 = 4.25$; $d(AB, DE) = (7 + 8.5 + 6 + 7.5)/4 = 7.25$. Merge {A,B,C} at 3.0; final merge at $(7 + 8.5 + 6 + 7.5 + 3.5 + 5)/6 = 6.25$.

**Merge heights summary:**

| Merge | Single | Complete | Average |
|---|---|---|---|
| {A,B} | 1 | 1 | 1 |
| {D,E} | 1.5 | 1.5 | 1.5 |
| {A,B,C} | 2.5 | 3.5 | 3.0 |
| All | 3.5 | 8.5 | 6.25 |

**Dendrogram (single linkage):**

```
Height
 3.5 |                +--------+--------+
     |                |                 |
 2.5 |           +----+----+            |
     |           |         |            |
 1.5 |           |         |       +----+----+
     |           |         |       |         |
 1.0 |        +--+--+      |       |         |
     |        |     |      |       |         |
     +--------A-----B------C-------D---------E-----
```

Cutting at height 3 gives 2 clusters: {A, B, C} and {D, E}.

---

## 12. Cluster Validation and Evaluation

### 12.1 Types of Validation

| Type | Uses | Examples |
|---|---|---|
| **Internal** | Only the data and clustering (no labels) | SSE, Silhouette, Davies–Bouldin, Dunn, Calinski–Harabasz |
| **External** | Compare with ground-truth labels | Purity, Rand Index, ARI, NMI, F-measure |
| **Relative** | Compare different clusterings/values of $k$ | Elbow method, silhouette vs $k$ |

### 12.2 Internal Measures

**Cohesion (WCSS) and Separation (BCSS):**

```math
WCSS = \sum_{j=1}^k\sum_{\mathbf{x}\in C_j}\lVert\mathbf{x} - \boldsymbol{\mu}_j\rVert^2, \qquad BCSS = \sum_{j=1}^k|C_j|\,\lVert\boldsymbol{\mu}_j - \boldsymbol{\mu}\rVert^2
```

$TSS = WCSS + BCSS$ ($\boldsymbol{\mu}$ = overall mean).

**Silhouette Coefficient** (for point $i$):

```math
a(i) = \frac{1}{|C_I| - 1}\sum_{j\in C_I,\,j\neq i}d(i,j) \qquad \text{(mean distance to own cluster)}
```

```math
b(i) = \min_{J\neq I}\frac{1}{|C_J|}\sum_{j\in C_J}d(i,j) \qquad \text{(mean distance to nearest other cluster)}
```

```math
s(i) = \frac{b(i) - a(i)}{\max\lbrace a(i), b(i)\rbrace}, \qquad -1 \leq s(i) \leq 1
```

- $s \approx 1$: well clustered; $s \approx 0$: on the boundary; $s < 0$: probably in the wrong cluster.
- Overall score = mean of $s(i)$ over all points.

**Davies–Bouldin Index** (lower is better):

```math
DB = \frac{1}{k}\sum_{i=1}^k\max_{j\neq i}\left(\frac{S_i + S_j}{d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)}\right), \qquad S_i = \frac{1}{|C_i|}\sum_{\mathbf{x}\in C_i}\lVert\mathbf{x} - \boldsymbol{\mu}_i\rVert
```

**Dunn Index** (higher is better):

```math
D = \frac{\min_{i\neq j}\delta(C_i, C_j)}{\max_l\Delta(C_l)}
```

$\delta$ = minimum distance between points of different clusters; $\Delta$ = diameter (maximum intra-cluster distance).

**Calinski–Harabasz Index** (higher is better):

```math
CH = \frac{BCSS/(k - 1)}{WCSS/(n - k)}
```

### 📝 Solved Numerical 12.1 — Silhouette, DB, Dunn, CH

1-D data: $C_1 = \lbrace 1, 2, 3\rbrace$, $C_2 = \lbrace 8, 9\rbrace$.

**Silhouette for each point:**

| Point | $a(i)$ | $b(i)$ | $s(i) = (b-a)/\max(a,b)$ |
|---|---|---|---|
| 1 | $(1 + 2)/2 = 1.5$ | $(7 + 8)/2 = 7.5$ | $6/7.5 = 0.800$ |
| 2 | $(1 + 1)/2 = 1.0$ | $(6 + 7)/2 = 6.5$ | $5.5/6.5 = 0.846$ |
| 3 | $(2 + 1)/2 = 1.5$ | $(5 + 6)/2 = 5.5$ | $4/5.5 = 0.727$ |
| 8 | $1/1 = 1.0$ | $(7 + 6 + 5)/3 = 6.0$ | $5/6 = 0.833$ |
| 9 | $1.0$ | $(8 + 7 + 6)/3 = 7.0$ | $6/7 = 0.857$ |

```math
\bar{s} = \frac{0.800 + 0.846 + 0.727 + 0.833 + 0.857}{5} = 0.813
```

Excellent clustering.

**Davies–Bouldin:** $\mu_1 = 2$, $\mu_2 = 8.5$. $S_1 = (1 + 0 + 1)/3 = 0.667$, $S_2 = (0.5 + 0.5)/2 = 0.5$.

```math
DB = \frac{0.667 + 0.5}{|2 - 8.5|} = \frac{1.167}{6.5} = 0.180
```

**Dunn:** min inter-cluster distance $= |3 - 8| = 5$; max diameter $= \max(|3 - 1|, |9 - 8|) = 2$ → $D = 5/2 = 2.5$.

**Calinski–Harabasz:** overall mean $= 23/5 = 4.6$.

```math
BCSS = 3(2 - 4.6)^2 + 2(8.5 - 4.6)^2 = 3(6.76) + 2(15.21) = 50.7
```

```math
WCSS = (1 + 0 + 1) + (0.25 + 0.25) = 2.5, \qquad CH = \frac{50.7/1}{2.5/3} = 60.84
```

### 12.3 External Measures

**Purity:**

```math
\text{Purity} = \frac{1}{n}\sum_{j=1}^k\max_i|C_j\cap T_i|
```

($T_i$ = true class $i$.) Purity = 1 is perfect, but it is trivially 1 if every point is its own cluster.

**Rand Index:** consider all $\binom{n}{2}$ pairs of points.
- $a$ = pairs in the same class **and** same cluster (TP)
- $b$ = pairs in different classes **and** different clusters (TN)
- $c$ = same class, different clusters (FN); $d$ = different classes, same cluster (FP)

```math
RI = \frac{a + b}{a + b + c + d} = \frac{a + b}{\binom{n}{2}}
```

**Adjusted Rand Index** (corrects for chance; 0 = random, 1 = perfect). With contingency table entries $n_{ij}$, row sums $a_i$, column sums $b_j$:

```math
ARI = \frac{\sum_{ij}\binom{n_{ij}}{2} - \left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]/\binom{n}{2}}{\frac{1}{2}\left[\sum_i\binom{a_i}{2} + \sum_j\binom{b_j}{2}\right] - \left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]/\binom{n}{2}}
```

**Normalized Mutual Information:**

```math
NMI(Y, C) = \frac{2\,I(Y; C)}{H(Y) + H(C)}, \qquad I(Y;C) = \sum_{i,j}P(i,j)\log\frac{P(i,j)}{P(i)P(j)}
```

**F-measure:** for class $i$ and cluster $j$, precision $= n_{ij}/|C_j|$, recall $= n_{ij}/|T_i|$, $F = 2PR/(P+R)$.

### 📝 Solved Numerical 12.2 — Purity, Rand Index, ARI

6 points. True labels: $(0, 0, 0, 1, 1, 1)$. Cluster labels: $(0, 0, 1, 1, 1, 1)$.

**Contingency table:**

|  | Cluster 0 | Cluster 1 | Row sum |
|---|---|---|---|
| Class 0 | 2 | 1 | 3 |
| Class 1 | 0 | 3 | 3 |
| Col sum | 2 | 4 | 6 |

**Purity:**

```math
\text{Purity} = \frac{\max(2, 0) + \max(1, 3)}{6} = \frac{2 + 3}{6} = 0.833
```

**Rand Index:** total pairs $\binom{6}{2} = 15$.

- $a$ (same class, same cluster): $\binom{2}{2} + \binom{1}{2} + \binom{0}{2} + \binom{3}{2} = 1 + 0 + 0 + 3 = 4$
- Same-class pairs: $\binom{3}{2} + \binom{3}{2} = 6$ → $c = 6 - 4 = 2$
- Same-cluster pairs: $\binom{2}{2} + \binom{4}{2} = 1 + 6 = 7$ → $d = 7 - 4 = 3$
- $b = 15 - 4 - 2 - 3 = 6$

```math
RI = \frac{4 + 6}{15} = 0.667
```

**ARI:**

```math
\text{Expected} = \frac{6 \times 7}{15} = 2.8, \qquad \text{Max} = \frac{6 + 7}{2} = 6.5
```

```math
ARI = \frac{4 - 2.8}{6.5 - 2.8} = \frac{1.2}{3.7} = 0.324
```

RI (0.667) looks decent, but ARI (0.324) reveals the clustering is only moderately better than random.

💡 **Exam tip:** For K-Means numericals, always show the distance table for every iteration and clearly state the convergence check. For hierarchical, redraw the updated distance matrix after each merge.

---

## 13. Formula Sheet

| Concept | Formula |
|---|---|
| OLS slope | $\beta_1 = S_{xy}/S_{xx} = \frac{\sum(x-\bar{x})(y-\bar{y})}{\sum(x-\bar{x})^2}$ |
| OLS intercept | $\beta_0 = \bar{y} - \beta_1\bar{x}$ |
| Correlation | $r = S_{xy}/\sqrt{S_{xx}S_{yy}}$ |
| Normal equation | $\boldsymbol{\beta} = (X^TX)^{-1}X^T\mathbf{y}$ |
| GD for LR | $\partial J/\partial w = -\frac{2}{n}\sum x_i(y_i - \hat{y}_i)$ |
| MSE / RMSE / MAE | $\frac{1}{n}\sum e^2$ / $\sqrt{\text{MSE}}$ / $\frac{1}{n}\sum\lvert e\rvert$ |
| $R^2$ | $1 - SSE/SST$ |
| Adjusted $R^2$ | $1 - \frac{(1-R^2)(n-1)}{n-p-1}$ |
| Huber | $\frac{1}{2}r^2$ if $\lvert r\rvert\leq\delta$, else $\delta(\lvert r\rvert - \delta/2)$ |
| Log-cosh | $\ln\cosh(r)$ |
| Quantile | $\tau r$ if $r\geq 0$, $(\tau-1)r$ otherwise |
| Nadaraya–Watson | $\hat{y} = \sum K_iy_i/\sum K_i$ |
| Gaussian kernel | $\exp(-(x_0-x_i)^2/2h^2)$ |
| Euclidean / Manhattan | $\sqrt{\sum(x_j-y_j)^2}$ / $\sum\lvert x_j - y_j\rvert$ |
| Minkowski | $(\sum\lvert x_j-y_j\rvert^p)^{1/p}$ |
| Mahalanobis | $\sqrt{(\mathbf{x}-\mathbf{y})^T\Sigma^{-1}(\mathbf{x}-\mathbf{y})}$ |
| Cosine similarity | $\mathbf{x}\cdot\mathbf{y}/(\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert)$ |
| Jaccard | $\lvert A\cap B\rvert/\lvert A\cup B\rvert$ |
| SMC | $(M_{11}+M_{00})/\text{total}$ |
| K-Means objective | $\sum_j\sum_{\mathbf{x}\in C_j}\lVert\mathbf{x}-\boldsymbol{\mu}_j\rVert^2$ |
| Centroid update | $\boldsymbol{\mu}_j = \frac{1}{\lvert C_j\rvert}\sum_{\mathbf{x}\in C_j}\mathbf{x}$ |
| K-Means++ | $P(\mathbf{x}) \propto D(\mathbf{x})^2$ |
| Single / Complete linkage | min / max pairwise distance |
| Ward | $\frac{\lvert C_i\rvert\lvert C_j\rvert}{\lvert C_i\rvert+\lvert C_j\rvert}\lVert\boldsymbol{\mu}_i-\boldsymbol{\mu}_j\rVert^2$ |
| Silhouette | $s = (b-a)/\max(a,b)$ |
| Davies–Bouldin | $\frac{1}{k}\sum_i\max_{j\neq i}\frac{S_i+S_j}{d(\mu_i,\mu_j)}$ |
| Dunn | min inter-cluster distance / max diameter |
| Calinski–Harabasz | $\frac{BCSS/(k-1)}{WCSS/(n-k)}$ |
| Purity | $\frac{1}{n}\sum_j\max_i\lvert C_j\cap T_i\rvert$ |
| Rand Index | $(a+b)/\binom{n}{2}$ |
| NMI | $2I(Y;C)/(H(Y)+H(C))$ |

---

[⬅ Unit III](Unit-3-NonParametric-and-Ensembles.md) | [Back to Index](README.md) | [Next: Unit V ➡](Unit-5-Reinforcement-Learning.md)
