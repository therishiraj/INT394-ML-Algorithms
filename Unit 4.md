# Unit IV — Regression & Clustering

[⬅ Back to Index](README.md)

**Part A – Regression:** linear regression → error metrics → linear vs non-linear → loss functions → non-parametric regression
**Part B – Clustering:** distance measures → K-Means → hierarchical clustering → how to evaluate clusters

---

# Part A — Regression

## 1. Classification vs Regression

> 💡 **Classification** answers "**which category?**" · **Regression** answers "**how much?**"

| | Classification | Regression |
|---|---|---|
| Output | Category (Yes/No, Cat/Dog) | Number (price, temperature) |
| Example | Will it rain? | How many mm of rain? |
| Loss | Cross-entropy | MSE, MAE |
| Metrics | Accuracy, F1 | MSE, RMSE, $R^2$ |
| Algorithms | Logistic Regression, SVM, Naïve Bayes | Linear Regression, Regression Trees |

> ⚠️ **Logistic Regression is a classification algorithm**, despite its name.

---

## 2. Simple Linear Regression

> 💡 **In simple words:** Draw the **best straight line** through the data points.

```math
\hat{y} = \beta_0 + \beta_1 x
```

- $\beta_1$ = slope (how much $y$ changes when $x$ increases by 1)
- $\beta_0$ = intercept (value of $y$ when $x = 0$)

**"Best" line = the one with the smallest total squared error** (Ordinary Least Squares):

```math
\text{minimise } \sum_{i=1}^n (y_i - \hat{y}_i)^2
```

**Solution:**

```math
\beta_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} \qquad \beta_0 = \bar{y} - \beta_1\bar{x}
```

($\bar{x}$, $\bar{y}$ = averages of $x$ and $y$.)

<details>
<summary>📘 Optional: Where do these formulas come from?</summary>

Set the derivatives of $J = \sum(y_i - \beta_0 - \beta_1x_i)^2$ to zero:

```math
\frac{\partial J}{\partial\beta_0} = -2\sum(y_i - \beta_0 - \beta_1x_i) = 0 \Rightarrow \beta_0 = \bar{y} - \beta_1\bar{x}
```

```math
\frac{\partial J}{\partial\beta_1} = -2\sum x_i(y_i - \beta_0 - \beta_1x_i) = 0
```

Substituting $\beta_0$ into the second equation and simplifying gives the formula for $\beta_1$.

</details>

### 📝 Example 2.1 — Fit a Line

| $x$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $y$ | 2 | 4 | 5 | 4 | 5 |

**Step 1:** $\bar{x} = 3$, $\bar{y} = 4$

**Step 2:**

| $x$ | $y$ | $x - \bar{x}$ | $y - \bar{y}$ | product | $(x - \bar{x})^2$ |
|---|---|---|---|---|---|
| 1 | 2 | −2 | −2 | 4 | 4 |
| 2 | 4 | −1 | 0 | 0 | 1 |
| 3 | 5 | 0 | 1 | 0 | 0 |
| 4 | 4 | 1 | 0 | 0 | 1 |
| 5 | 5 | 2 | 1 | 2 | 4 |
| | | | **Sum** | **6** | **10** |

**Step 3:**

```math
\beta_1 = \frac{6}{10} = 0.6 \qquad \beta_0 = 4 - 0.6 \times 3 = 2.2
```

**Line:** $\hat{y} = 2.2 + 0.6x$. For $x = 6$: $\hat{y} = 5.8$.

### 2.1 Gradient Descent Version

Instead of the formula, we can start from $w = 0, b = 0$ and take small steps downhill:

```math
w \leftarrow w - \eta\frac{\partial J}{\partial w}, \qquad \frac{\partial J}{\partial w} = -\frac{2}{n}\sum x_i(y_i - \hat{y}_i), \qquad \frac{\partial J}{\partial b} = -\frac{2}{n}\sum(y_i - \hat{y}_i)
```

**One step** on the same data ($w = b = 0$, $\eta = 0.01$): $\sum x_iy_i = 66$, $\sum y_i = 20$

```math
\frac{\partial J}{\partial w} = -\frac{2}{5}(66) = -26.4 \Rightarrow w = 0.264 \qquad \frac{\partial J}{\partial b} = -\frac{2}{5}(20) = -8 \Rightarrow b = 0.08
```

Many steps later it reaches $w = 0.6$, $b = 2.2$.

---

## 3. Multiple Linear Regression

More than one input:

```math
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_dx_d
```

**Normal equation** (matrix form; $X$ has a first column of 1's):

```math
\boldsymbol{\beta} = (X^TX)^{-1}X^T\mathbf{y}
```

### 📝 Example 3.1

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 1 | 1 | 6 |
| 2 | 1 | 8 |
| 2 | 2 | 9 |
| 3 | 2 | 11 |

```math
X^TX = \begin{bmatrix}4&8&6\\8&18&13\\6&13&10\end{bmatrix}, \quad X^T\mathbf{y} = \begin{bmatrix}34\\73\\54\end{bmatrix}, \quad (X^TX)^{-1} = \begin{bmatrix}2.75&-0.5&-1\\-0.5&1&-1\\-1&-1&2\end{bmatrix}
```

```math
\boldsymbol{\beta} = (X^TX)^{-1}X^T\mathbf{y} = \begin{bmatrix}3\\2\\1\end{bmatrix} \Rightarrow \hat{y} = 3 + 2x_1 + x_2
```

---

## 4. Measuring Regression Error

```math
\text{MSE} = \frac{1}{n}\sum(y_i - \hat{y}_i)^2 \qquad \text{RMSE} = \sqrt{\text{MSE}} \qquad \text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|
```

**$R^2$ — how much of the variation the model explains** (1 = perfect, 0 = no better than the average):

```math
R^2 = 1 - \frac{SSE}{SST} \qquad SSE = \sum(y_i - \hat{y}_i)^2 \qquad SST = \sum(y_i - \bar{y})^2
```

**Adjusted $R^2$** (penalises adding useless features; $p$ = number of features):

```math
R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
```

### 📝 Example 4.1 — Errors for $\hat{y} = 2.2 + 0.6x$

| $x$ | $y$ | $\hat{y}$ | error | error² |
|---|---|---|---|---|
| 1 | 2 | 2.8 | −0.8 | 0.64 |
| 2 | 4 | 3.4 | 0.6 | 0.36 |
| 3 | 5 | 4.0 | 1.0 | 1.00 |
| 4 | 4 | 4.6 | −0.6 | 0.36 |
| 5 | 5 | 5.2 | −0.2 | 0.04 |
| | | | **Sum** | **2.40** |

```math
\text{MSE} = \frac{2.4}{5} = 0.48 \quad \text{RMSE} = 0.69 \quad \text{MAE} = \frac{0.8 + 0.6 + 1 + 0.6 + 0.2}{5} = 0.64
```

```math
SST = 4 + 0 + 1 + 0 + 1 = 6 \qquad R^2 = 1 - \frac{2.4}{6} = 0.6 \qquad R^2_{adj} = 1 - \frac{0.4 \times 4}{3} = 0.47
```

The line explains 60% of the variation in $y$.

---

## 5. Linear vs Non-linear Regression

> 💡 "Linear" means **linear in the parameters** ($\beta$'s), not necessarily a straight line in $x$.

| Model | Type |
|---|---|
| $y = \beta_0 + \beta_1x$ | Linear |
| $y = \beta_0 + \beta_1x + \beta_2x^2$ | Still **linear** (polynomial regression) |
| $y = ae^{bx}$ | Non-linear |

| | Linear | Non-linear |
|---|---|---|
| Shape | Straight line / plane | Curve |
| Solving | Direct formula | Iterative methods |
| Risk | Underfitting curved data | Overfitting |
| Example | Salary vs experience | Population growth |

**Trick:** some non-linear models become linear after taking logs:

```math
y = ae^{bx} \quad\xrightarrow{\ \ln\ }\quad \ln y = \ln a + bx
```

### 📝 Example 5.1 — Exponential Fit

| $x$ | 0 | 1 | 2 |
|---|---|---|---|
| $y$ | 2 | 5.4 | 14.8 |
| $\ln y$ | 0.693 | 1.686 | 2.695 |

Fit a line to $(x, \ln y)$: slope $b = \frac{(-1)(-0.998) + (1)(1.004)}{2} = 1.0$, intercept $= 1.691 - 1.0 = 0.69$, so $a = e^{0.69} = 2$.

**Model:** $y = 2e^{x}$.

---

## 6. Loss Functions for Regression

Let error $r = y - \hat{y}$.

| Loss | Formula | Behaviour |
|---|---|---|
| **MSE** | $\frac{1}{n}\sum r^2$ | Punishes big errors heavily → **sensitive to outliers** |
| **MAE** | $\frac{1}{n}\sum\lvert r\rvert$ | Treats all errors equally → **robust to outliers** |
| **Huber** | see below | Mix of both |
| **Log-cosh** | $\sum\ln(\cosh r)$ | Smooth version similar to Huber |

**Huber loss** — squared for small errors, linear for large ones:

```math
L_\delta(r) = \begin{cases}\frac{1}{2}r^2 & \text{if } |r| \leq \delta\\[1mm] \delta\left(|r| - \frac{\delta}{2}\right) & \text{if } |r| > \delta\end{cases}
```

### 📝 Example 6.1 — Effect of an Outlier

Errors: $1, -2, 0.5, 6$ (6 is an outlier).

```math
\text{MSE} = \frac{1 + 4 + 0.25 + 36}{4} = 10.31 \qquad \text{MAE} = \frac{1 + 2 + 0.5 + 6}{4} = 2.375
```

**Huber ($\delta = 1.5$):** $0.5(1)^2 = 0.5$; $1.5(2 - 0.75) = 1.875$; $0.5(0.5)^2 = 0.125$; $1.5(6 - 0.75) = 7.875$

```math
\text{Huber} = \frac{0.5 + 1.875 + 0.125 + 7.875}{4} = 2.59
```

The outlier makes up 87% of the MSE — MSE is heavily affected by outliers.

---

## 7. Non-parametric Regression

> 💡 No fixed equation — the prediction is built directly from nearby data points.

**KNN Regression:** average of the $k$ nearest $y$ values (see Unit III).

**Kernel Regression (Nadaraya–Watson):** weighted average of **all** points; closer points get bigger weights.

```math
\hat{y}(x_0) = \frac{\sum_i K_i\,y_i}{\sum_i K_i} \qquad K_i = \exp\left(-\frac{(x_0 - x_i)^2}{2h^2}\right)
```

($h$ = bandwidth: small → wiggly; large → too smooth.)

**Regression trees** (Unit III) and **LOESS** (local line fitting) are also non-parametric.

### 📝 Example 7.1 — Kernel Regression

Data $x = (1..5)$, $y = (2, 4, 5, 4, 5)$. Predict at $x_0 = 2.5$, $h = 1$.

| $x_i$ | $(2.5 - x_i)^2$ | $K_i$ | $K_iy_i$ |
|---|---|---|---|
| 1 | 2.25 | 0.325 | 0.649 |
| 2 | 0.25 | 0.883 | 3.530 |
| 3 | 0.25 | 0.883 | 4.413 |
| 4 | 2.25 | 0.325 | 1.299 |
| 5 | 6.25 | 0.044 | 0.220 |
| **Sum** | | **2.458** | **10.110** |

```math
\hat{y}(2.5) = \frac{10.110}{2.458} = 4.11
```

---

# Part B — Clustering

## 8. Classification vs Clustering

> 💡 **Classification** = sorting into **known** groups (labels given). **Clustering** = **discovering** groups (no labels).

| | Classification | Clustering |
|---|---|---|
| Learning | Supervised | Unsupervised |
| Labels | Given | Not given |
| Goal | Predict label of new data | Group similar data |
| Example | Spam filter | Customer segmentation |
| Algorithms | Decision tree, SVM | K-Means, Hierarchical |

**Good clustering:** points in the same cluster are **close**, and different clusters are **far apart**.

---

## 9. Similarity and Distance Measures

| Measure | Formula | Used for |
|---|---|---|
| Euclidean | $\sqrt{\sum(x_j - y_j)^2}$ | General numeric data |
| Manhattan | $\sum\lvert x_j - y_j\rvert$ | Grid-like data |
| Minkowski | $\left(\sum\lvert x_j - y_j\rvert^p\right)^{1/p}$ | General form |
| Chebyshev | $\max_j\lvert x_j - y_j\rvert$ | Largest single difference |
| Cosine similarity | $\frac{\mathbf{x}\cdot\mathbf{y}}{\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert}$ | Text / documents |
| Jaccard | $\frac{\lvert A\cap B\rvert}{\lvert A\cup B\rvert}$ | Sets, binary data |
| Hamming | No. of positions that differ | Strings, binary vectors |

### 📝 Example 9.1 — All Distances

$\mathbf{A} = (1, 2, 3)$, $\mathbf{B} = (4, 6, 8)$. Differences: $(3, 4, 5)$.

```math
\text{Euclidean} = \sqrt{9 + 16 + 25} = 7.07 \qquad \text{Manhattan} = 3 + 4 + 5 = 12 \qquad \text{Chebyshev} = 5
```

```math
\text{Minkowski } (p = 3) = (27 + 64 + 125)^{1/3} = 216^{1/3} = 6
```

```math
\cos\theta = \frac{1 \times 4 + 2 \times 6 + 3 \times 8}{\sqrt{14}\sqrt{116}} = \frac{40}{40.3} = 0.99 \quad \text{(almost the same direction)}
```

### 📝 Example 9.2 — Jaccard and Hamming

$\mathbf{p} = 1011001$, $\mathbf{q} = 1001011$. Both-1 positions: 3; mismatches: 2.

```math
\text{Jaccard} = \frac{3}{3 + 2} = 0.6 \qquad \text{Hamming} = 2
```

---

## 10. Partition-based Clustering: K-Means

> 💡 **In simple words:** Pick $k$ centre points. Assign every point to its nearest centre. Move each centre to the middle of its points. Repeat until nothing changes.

### 10.1 Algorithm
1. Choose $k$ and pick $k$ starting centroids.
2. **Assign** each point to the nearest centroid.
3. **Update** each centroid = mean of its points:

```math
\boldsymbol{\mu}_j = \frac{1}{|C_j|}\sum_{\mathbf{x}\in C_j}\mathbf{x}
```

4. Repeat steps 2–3 until the clusters stop changing.

**What K-Means minimises** — Within-Cluster Sum of Squares (WCSS):

```math
J = \sum_{j=1}^{k}\sum_{\mathbf{x}\in C_j}\lVert\mathbf{x} - \boldsymbol{\mu}_j\rVert^2
```

**Choosing $k$ — Elbow method:** plot WCSS vs $k$ and pick the "elbow" where the curve stops dropping sharply.

| Pros | Cons |
|---|---|
| Simple, fast | Must choose $k$ in advance |
| Works well for round clusters | Sensitive to starting centroids and outliers |

**Variants:** **K-Means++** (smarter starting centroids, spread apart) · **K-Medoids** (centre must be an actual data point → more robust to outliers).

### 📝 Example 10.1 — K-Means with k = 2

| Point | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| $(x, y)$ | (1,1) | (1.5,2) | (2.9,4) | (5,7) | (3.5,5) | (4.5,5) | (3.5,4.5) |

Start: $\mu_1 = A(1, 1)$, $\mu_2 = D(5, 7)$.

**Iteration 1 — assign:**

| Point | Dist to $\mu_1$ | Dist to $\mu_2$ | Cluster |
|---|---|---|---|
| A | 0 | 7.21 | 1 |
| B | 1.12 | 6.10 | 1 |
| C | 3.55 | 3.66 | 1 |
| D | 7.21 | 0 | 2 |
| E | 4.72 | 2.50 | 2 |
| F | 5.32 | 2.06 | 2 |
| G | 4.30 | 2.92 | 2 |

(e.g. $d(C, \mu_1) = \sqrt{1.9^2 + 3^2} = \sqrt{12.61} = 3.55$)

**Update:**

```math
\mu_1 = \left(\frac{1 + 1.5 + 2.9}{3}, \frac{1 + 2 + 4}{3}\right) = (1.8, 2.33) \qquad \mu_2 = \left(\frac{5 + 3.5 + 4.5 + 3.5}{4}, \frac{7 + 5 + 5 + 4.5}{4}\right) = (4.13, 5.38)
```

**Iteration 2 — assign again:** point C is now closer to $\mu_2$ (1.84) than to $\mu_1$ (2.00) → **C moves to cluster 2**. All others stay.

**Update:**

```math
\mu_1 = (1.25, 1.5) \qquad \mu_2 = (3.88, 5.1)
```

**Iteration 3:** no point changes cluster → **stop**.

**Final:** Cluster 1 = {A, B}, Cluster 2 = {C, D, E, F, G}.

---

## 11. Hierarchical Clustering

> 💡 **In simple words:** Start with every point as its own cluster, then keep **merging the two closest clusters** until only one is left. The merge history is drawn as a tree called a **dendrogram**.

- **Agglomerative** = bottom-up (merge). **Divisive** = top-down (split).
- No need to choose $k$ beforehand — just **cut the dendrogram** at the height you want.

### 11.1 Linkage — "distance between two clusters"

| Linkage | Distance between clusters | Behaviour |
|---|---|---|
| **Single** | Closest pair of points (min) | Long, chain-like clusters |
| **Complete** | Farthest pair of points (max) | Compact, round clusters |
| **Average** | Average over all pairs | In between |
| **Ward** | Increase in WCSS after merging | Similar to K-Means |

### 📝 Example 11.1 — Single and Complete Linkage

Points on a line: A = 1, B = 2, C = 4.5, D = 8, E = 9.5.

**Distance matrix:**

| | A | B | C | D | E |
|---|---|---|---|---|---|
| **A** | 0 | | | | |
| **B** | 1 | 0 | | | |
| **C** | 3.5 | 2.5 | 0 | | |
| **D** | 7 | 6 | 3.5 | 0 | |
| **E** | 8.5 | 7.5 | 5 | 1.5 | 0 |

**Single linkage (use the minimum):**
1. Smallest = 1 → merge **{A,B}** at height 1.
2. Next smallest = 1.5 → merge **{D,E}** at height 1.5.
3. $d(AB, C) = \min(3.5, 2.5) = 2.5$; $d(C, DE) = \min(3.5, 5) = 3.5$ → merge **{A,B,C}** at 2.5.
4. $d(ABC, DE) = \min(6, 3.5, \dots) = 3.5$ → merge all at 3.5.

**Complete linkage (use the maximum):**
1. {A,B} at 1; {D,E} at 1.5.
2. $d(AB, C) = \max(3.5, 2.5) = 3.5$; $d(C, DE) = \max(3.5, 5) = 5$ → merge **{A,B,C}** at 3.5.
3. $d(ABC, DE) = \max(\text{all pairs}) = 8.5$ → merge all at 8.5.

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

Cutting at height 3 → 2 clusters: {A, B, C} and {D, E}.

---

## 12. Cluster Validation and Evaluation

> 💡 How do we know if the clusters are good?
> - **Internal** measures: use only the data (are clusters tight and far apart?).
> - **External** measures: compare with true labels, if we have them.

### 12.1 Silhouette Score (internal) — most important

For each point $i$:
- $a$ = average distance to points in **its own** cluster (want **small**)
- $b$ = average distance to points in the **nearest other** cluster (want **large**)

```math
s(i) = \frac{b - a}{\max(a, b)}
```

- $s \approx 1$ → well clustered · $s \approx 0$ → on the border · $s < 0$ → probably in the wrong cluster.

### 📝 Example 12.1 — Silhouette

Clusters: $C_1 = \lbrace 1, 2, 3\rbrace$, $C_2 = \lbrace 8, 9\rbrace$.

| Point | $a$ (own cluster) | $b$ (other cluster) | $s$ |
|---|---|---|---|
| 1 | $(1 + 2)/2 = 1.5$ | $(7 + 8)/2 = 7.5$ | $6/7.5 = 0.80$ |
| 2 | $(1 + 1)/2 = 1.0$ | $(6 + 7)/2 = 6.5$ | $5.5/6.5 = 0.85$ |
| 3 | $(2 + 1)/2 = 1.5$ | $(5 + 6)/2 = 5.5$ | $4/5.5 = 0.73$ |
| 8 | 1 | $(7 + 6 + 5)/3 = 6$ | $5/6 = 0.83$ |
| 9 | 1 | $(8 + 7 + 6)/3 = 7$ | $6/7 = 0.86$ |

Average $= 0.81$ → very good clustering.

### 12.2 Other Internal Measures

| Measure | Idea | Better when |
|---|---|---|
| WCSS / SSE | Total squared distance to centroids | Lower |
| Davies–Bouldin | Average similarity of each cluster with its most similar one | Lower |
| Dunn index | $\frac{\text{smallest gap between clusters}}{\text{largest cluster width}}$ | Higher |

For Example 12.1: smallest gap = $8 - 3 = 5$, largest width = $3 - 1 = 2$ → Dunn $= 5/2 = 2.5$.

### 12.3 External Measures

**Purity** — for each cluster, count its most common true class:

```math
\text{Purity} = \frac{1}{n}\sum_{\text{clusters}}\max(\text{count of one class in the cluster})
```

**Rand Index** — look at every pair of points; count pairs where clustering and true labels **agree** (both "same group" or both "different group"):

```math
RI = \frac{\text{agreeing pairs}}{\text{total pairs}} = \frac{a + b}{\binom{n}{2}}
```

### 📝 Example 12.2 — Purity and Rand Index

True labels: $(0, 0, 0, 1, 1, 1)$. Clusters: $(0, 0, 1, 1, 1, 1)$.

| | Cluster 0 | Cluster 1 |
|---|---|---|
| Class 0 | 2 | 1 |
| Class 1 | 0 | 3 |

```math
\text{Purity} = \frac{2 + 3}{6} = 0.83
```

**Rand Index:** total pairs $= \binom{6}{2} = 15$.
- Same class & same cluster ($a$): pair (1,2) in cluster 0 + 3 pairs among points 4,5,6 → $a = 4$
- Different class & different cluster ($b$): $b = 6$

```math
RI = \frac{4 + 6}{15} = 0.67
```

---

## ✅ Quick Revision

| Concept | Formula |
|---|---|
| Slope | $\beta_1 = \frac{\sum(x-\bar{x})(y-\bar{y})}{\sum(x-\bar{x})^2}$ |
| Intercept | $\beta_0 = \bar{y} - \beta_1\bar{x}$ |
| Normal equation | $\boldsymbol{\beta} = (X^TX)^{-1}X^T\mathbf{y}$ |
| MSE / RMSE / MAE | $\frac{1}{n}\sum r^2$ / $\sqrt{\text{MSE}}$ / $\frac{1}{n}\sum\lvert r\rvert$ |
| $R^2$ | $1 - SSE/SST$ |
| Adjusted $R^2$ | $1 - \frac{(1-R^2)(n-1)}{n-p-1}$ |
| Huber | $\frac{1}{2}r^2$ if small, $\delta(\lvert r\rvert - \delta/2)$ if large |
| Kernel regression | $\frac{\sum K_iy_i}{\sum K_i}$ |
| Euclidean / Manhattan | $\sqrt{\sum(x-y)^2}$ / $\sum\lvert x-y\rvert$ |
| Cosine similarity | $\frac{\mathbf{x}\cdot\mathbf{y}}{\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert}$ |
| Jaccard | $\frac{\lvert A\cap B\rvert}{\lvert A\cup B\rvert}$ |
| Centroid | mean of points in the cluster |
| WCSS | $\sum_j\sum_{\mathbf{x}\in C_j}\lVert\mathbf{x}-\boldsymbol{\mu}_j\rVert^2$ |
| Single / Complete linkage | min / max distance between clusters |
| Silhouette | $\frac{b-a}{\max(a,b)}$ |
| Purity | $\frac{1}{n}\sum\max(\text{class count})$ |
| Rand Index | $\frac{a+b}{\binom{n}{2}}$ |

**Likely exam questions:** Fit a regression line + find $R^2$ · MSE vs MAE vs Huber · K-Means iterations · Single vs complete linkage with dendrogram · Silhouette calculation · Classification vs clustering.

---

[⬅ Unit III](Unit-3-NonParametric-and-Ensembles.md) | [Back to Index](README.md) | [Next: Unit V ➡](Unit-5-Reinforcement-Learning.md)
