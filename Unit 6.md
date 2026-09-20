# Unit VI — Model Complexity & Optimization

[⬅ Back to Index](README.md)

**Part A – Model Complexity:** VC dimension → Rademacher complexity → bias–variance → overfitting vs underfitting → regularization
**Part B – Optimization:** cross-validation → hyperparameter tuning → Structural Risk Minimization → gradient descent & variants → convergence

---

# Part A — Model Complexity

> 💡 **Big idea of this unit:** A more powerful (complex) model can fit the training data better, but it may **memorise noise** and do badly on new data. We need ways to **measure** complexity and **control** it.

```math
\text{Test error} \leq \text{Training error} + \text{Complexity penalty}
```

## 1. VC Dimension

> 💡 **In simple words:** VC dimension = the **largest number of points** a model type can label in **every possible way**. Higher VC dimension = more powerful (more flexible) model.

### 1.1 Shattering

$m$ points with two classes can be labelled in $2^m$ ways. A model class **shatters** the points if it can produce **all $2^m$ labellings**.

**VC dimension** = size of the largest set of points that can be shattered.

To show $\text{VC} = d$:
1. Find **some** set of $d$ points that can be shattered ✅
2. Show that **no** set of $d + 1$ points can be shattered ❌

### 1.2 Common Examples

| Model | VC dimension |
|---|---|
| Threshold on a line ($x \geq a$) | 1 |
| Interval on a line ($a \leq x \leq b$) | 2 |
| Straight line in 2D | **3** |
| Hyperplane in $d$ dimensions | $d + 1$ |
| Axis-aligned rectangle in 2D | 4 |
| Finite class of $N$ models | at most $\log_2 N$ |

> Rule of thumb: VC dimension ≈ number of parameters (but not always).

### 1.3 Why a Line in 2D has VC dimension 3

- **3 points (triangle):** all $2^3 = 8$ labellings can be separated by a line ✅
- **4 points:** label opposite corners the same (+ − + −, the **XOR** pattern) → no straight line can separate them ❌

So VC dimension = 3.

### 1.4 VC Generalization Bound

With probability at least $1 - \delta$:

```math
\text{Test error} \leq \text{Training error} + \sqrt{\frac{h\left(\ln\frac{2m}{h} + 1\right) + \ln\frac{4}{\delta}}{m}}
```

- $h$ = VC dimension, $m$ = number of training examples.
- **Bigger $h$** → bigger penalty. **More data $m$** → smaller penalty.

### 📝 Example 1.1 — VC Bound

$h = 10$, $m = 10{,}000$, $\delta = 0.05$, training error = 0.02.

```math
h\left(\ln\frac{2m}{h} + 1\right) = 10(\ln 2000 + 1) = 10(7.601 + 1) = 86.01 \qquad \ln\frac{4}{0.05} = \ln 80 = 4.38
```

```math
\text{Penalty} = \sqrt{\frac{86.01 + 4.38}{10000}} = \sqrt{0.00904} = 0.095
```

```math
\text{Test error} \leq 0.02 + 0.095 = 0.115
```

With 95% confidence, test error is at most 11.5%.

### 📝 Example 1.2 — Quick Questions

- VC dimension of a perceptron with 5 inputs? → $5 + 1 = 6$
- Upper bound on VC dimension of a class with 64 models? → $\log_2 64 = 6$

<details>
<summary>📘 Optional: Sauer's Lemma</summary>

If VC dimension is $d$, the number of different labellings on $m$ points is at most:

```math
\sum_{i=0}^{d}\binom{m}{i}
```

Example: $d = 3$, $m = 10$ → $1 + 10 + 45 + 120 = 176$ labellings, compared with $2^{10} = 1024$ possible. So a finite VC dimension limits how many patterns the model can fit.

</details>

---

## 2. Rademacher Complexity

> 💡 **In simple words:** Give the data **random labels** (+1/−1, like coin flips). If a model class can fit these random labels well, it is very flexible — and likely to overfit. Rademacher complexity measures **how well a model class can fit pure noise**.

**Rademacher variables:** $\sigma_i = +1$ or $-1$, each with probability ½.

**Empirical Rademacher complexity:**

```math
\hat{\mathcal{R}}(\mathcal{H}) = \mathbb{E}_\sigma\left[\max_{h\in\mathcal{H}}\frac{1}{m}\sum_{i=1}^m\sigma_i\,h(x_i)\right]
```

Reading it: for each random labelling, find the model that **agrees** with it the most; average over all random labellings.

- Value between 0 and 1. **Higher = more complex.**
- If the class can fit every labelling (shatters the data) → value = 1.
- A single fixed model → value = 0.

**Generalization bound:**

```math
\text{Test error} \leq \text{Training error} + 2\,\mathcal{R}_m(\mathcal{H}) + \sqrt{\frac{\ln(1/\delta)}{2m}}
```

| | VC Dimension | Rademacher Complexity |
|---|---|---|
| Type | A whole number | A real number (average) |
| Depends on the data? | No (worst case) | Yes |
| Bounds | Often loose | Usually tighter |

### 📝 Example 2.1 — Compute Rademacher Complexity

2 points; two models: $h_1 = (+1, +1)$, $h_2 = (+1, -1)$. List all 4 random labellings:

| $\sigma$ | Agreement with $h_1$: $\frac{1}{2}\sigma\cdot h_1$ | With $h_2$ | Max |
|---|---|---|---|
| (+1, +1) | 1 | 0 | 1 |
| (+1, −1) | 0 | 1 | 1 |
| (−1, +1) | 0 | −1 | 0 |
| (−1, −1) | −1 | 0 | 0 |

```math
\hat{\mathcal{R}} = \frac{1 + 1 + 0 + 0}{4} = 0.5
```

---

## 3. Bias–Variance Trade-off

> 💡 **In simple words:**
> - **Bias** = error from wrong assumptions (model too simple, **misses the pattern**).
> - **Variance** = error from being too sensitive to the training data (model too complex, **changes a lot** with different data).
> - **Noise** = randomness in the data that no model can remove.

🎯 **Dartboard analogy:** high bias = darts grouped together but away from the centre; high variance = darts scattered all over.

```math
\text{Expected Test Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}(\sigma^2)
```

```math
\text{Bias} = \bar{f}(x) - f(x) \qquad \text{Variance} = \mathbb{E}\left[(\hat{f}(x) - \bar{f}(x))^2\right]
```

($f$ = true function, $\hat{f}$ = model's prediction, $\bar{f}$ = average prediction over different training sets.)

**The trade-off:** as complexity increases, bias ↓ but variance ↑. The best model is in the middle.

| Model complexity → | Simple | Just right | Complex |
|---|---|---|---|
| Bias | High | Medium | Low |
| Variance | Low | Medium | High |
| Test error | High | **Lowest** | High |
| Problem | Underfitting | ✅ | Overfitting |

| Model | Bias | Variance |
|---|---|---|
| Straight line on curved data | High | Low |
| Very high-degree polynomial | Low | High |
| KNN with $k = 1$ | Low | High |
| KNN with large $k$ | High | Low |

### 📝 Example 3.1 — Bias–Variance Calculation

True value $f(x_0) = 3.5$, noise $\sigma^2 = 0.25$. A model trained on 4 different datasets predicts: 2.8, 3.2, 3.0, 3.4.

**Average prediction:** $\bar{f} = \frac{2.8 + 3.2 + 3.0 + 3.4}{4} = 3.1$

```math
\text{Bias}^2 = (3.1 - 3.5)^2 = 0.16
```

```math
\text{Variance} = \frac{(-0.3)^2 + (0.1)^2 + (-0.1)^2 + (0.3)^2}{4} = \frac{0.2}{4} = 0.05
```

```math
\text{Total error} = 0.16 + 0.05 + 0.25 = 0.46
```

Bias is bigger than variance → the model is **underfitting**; try a more complex model.

---

## 4. Overfitting vs Underfitting

| | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| Training error | High | Low | Very low |
| Test error | High | Low | High |
| Gap between them | Small | Small | **Large** |
| Cause | Too simple | Right | Too complex |
| Like a student who… | Didn't study | Understood | Memorised answers |

**Fixes:**

| Underfitting | Overfitting |
|---|---|
| Use a more complex model | Use a simpler model |
| Add more / better features | Get more data |
| Reduce regularization | Increase regularization |
| Train longer | Early stopping, pruning, dropout |

### 📝 Example 4.1 — Diagnose

| Model | Train error | Test error | Diagnosis |
|---|---|---|---|
| A | 25% | 27% | Underfitting |
| B | 2% | 20% | Overfitting |
| C | 6% | 8% | Good fit |

---

## 5. Regularization Techniques

> 💡 **In simple words:** Add a **penalty for large weights** to the loss. This stops the model from becoming too complex.

```math
\text{New loss} = \text{Original loss} + \lambda \times \text{Penalty}
```

$\lambda$ controls the strength: $\lambda = 0$ → no regularization; very large $\lambda$ → weights squashed towards 0 (underfitting).

### 5.1 Ridge (L2)

```math
J = \sum(y_i - \hat{y}_i)^2 + \lambda\sum_j w_j^2 \qquad \mathbf{w} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}
```

Shrinks all weights **towards** zero, but they rarely become **exactly** zero.

### 5.2 Lasso (L1)

```math
J = \sum(y_i - \hat{y}_i)^2 + \lambda\sum_j |w_j|
```

Can make some weights **exactly zero** → automatically removes useless features (**feature selection**).

### 5.3 Elastic Net = L1 + L2

```math
J = \sum(y_i - \hat{y}_i)^2 + \lambda_1\sum|w_j| + \lambda_2\sum w_j^2
```

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty | $\lambda\sum w^2$ | $\lambda\sum\lvert w\rvert$ |
| Makes weights exactly 0? | No | Yes |
| Feature selection | No | Yes |

### 📝 Example 5.1 — Ridge vs Lasso (1 feature)

From Unit IV data: $\sum xy = 6$, $\sum x^2 = 10$ (after centring), so the normal answer is $w = 0.6$.

```math
\text{Ridge: } w = \frac{\sum xy}{\sum x^2 + \lambda} \qquad \text{Lasso: } w = \frac{\max(\sum xy - \lambda,\ 0)}{\sum x^2}
```

| $\lambda$ | Ridge | Lasso |
|---|---|---|
| 0 | 6/10 = 0.60 | 6/10 = 0.60 |
| 2 | 6/12 = 0.50 | 4/10 = 0.40 |
| 5 | 6/15 = 0.40 | 1/10 = 0.10 |
| 7 | 6/17 = 0.35 | **0** |

Ridge shrinks slowly; Lasso hits **exactly 0**.

### 5.4 Other Techniques

- **Early stopping:** stop training when validation error starts rising.
- **Dropout** (neural networks): randomly switch off neurons during training so the network can't depend on any single one.
- **Data augmentation:** create more training data (rotate/flip images, add noise).
- **Pruning:** cut branches of decision trees (Unit III).

---

# Part B — Optimization

## 6. Cross-Validation

> 💡 **In simple words:** Test the model on data it **has not seen**, several times, and average the results — gives a more reliable estimate than a single test.

### 6.1 k-Fold Cross-Validation
1. Split the data into $k$ equal parts (folds).
2. Train on $k - 1$ folds, test on the remaining one.
3. Repeat $k$ times so every fold is tested once.
4. Average the errors:

```math
CV = \frac{1}{k}\sum_{i=1}^{k} E_i
```

| Type | Description |
|---|---|
| **Hold-out** | One train/test split (e.g. 80/20) — simple but less reliable |
| **k-Fold** | $k$ = 5 or 10 is common |
| **Leave-One-Out (LOOCV)** | $k = n$; very accurate but slow |
| **Stratified k-Fold** | Each fold keeps the same class proportions — use for imbalanced data |

### 📝 Example 6.1 — 5-Fold CV

Fold errors: 0.12, 0.10, 0.15, 0.11, 0.12.

```math
CV = \frac{0.12 + 0.10 + 0.15 + 0.11 + 0.12}{5} = \frac{0.60}{5} = 0.12
```

With 1000 samples and 10-fold CV: each model trains on 900 and tests on 100; 10 models are trained.

---

## 7. Hyperparameter Tuning

**Parameters** (like weights) are learned by the model. **Hyperparameters** are set **by you before training** — e.g. learning rate, $k$ in KNN, tree depth, $\lambda$.

| Method | How | Pros / Cons |
|---|---|---|
| **Grid Search** | Try every combination | Thorough, but slow |
| **Random Search** | Try random combinations | Faster, often just as good |
| **Bayesian Optimization** | Learns which region looks promising and tries there next | Fewest trials, more complex |

Always use cross-validation to score each combination.

### 📝 Example 7.1 — Cost of Grid Search

$C \in \lbrace 0.1, 1, 10\rbrace$, $\gamma \in \lbrace 0.001, 0.01, 0.1, 1\rbrace$, kernel ∈ {linear, rbf}, with 5-fold CV:

```math
\text{Combinations} = 3 \times 4 \times 2 = 24 \qquad \text{Models trained} = 24 \times 5 = 120
```

---

## 8. Structural Risk Minimization (SRM)

> 💡 **In simple words:** ERM picks the model with the lowest **training error** (can overfit). SRM picks the model with the lowest **training error + complexity penalty**. It chooses the **simplest model that fits well enough**.

**Steps:**
1. Make a sequence of model classes from simple to complex: $\mathcal{H}_1 \subset \mathcal{H}_2 \subset \mathcal{H}_3 \dots$ (e.g. polynomials of degree 1, 2, 3, …)
2. Find the best model in each class.
3. Pick the one with the lowest **training error + VC penalty**:

```math
\text{Choose } \arg\min_i\left[\text{Training error}_i + \sqrt{\frac{h_i\left(\ln\frac{2m}{h_i} + 1\right) + \ln\frac{4}{\delta}}{m}}\right]
```

Regularization is SRM in practice — the $\lambda$ penalty plays the role of the complexity term.

### 📝 Example 8.1 — Model Selection with SRM

$m = 1000$, $\delta = 0.05$:

| Class | VC dim $h$ | Train error | Penalty | **Total** |
|---|---|---|---|---|
| $\mathcal{H}_1$ | 1 | 0.30 | 0.114 | 0.414 |
| $\mathcal{H}_2$ | 2 | 0.18 | 0.142 | 0.322 |
| $\mathcal{H}_3$ | 3 | 0.10 | 0.164 | **0.264** ✅ |
| $\mathcal{H}_4$ | 5 | 0.08 | 0.198 | 0.278 |
| $\mathcal{H}_5$ | 10 | 0.06 | 0.260 | 0.320 |

Penalty for $h = 3$: $\sqrt{\frac{3(\ln 666.7 + 1) + \ln 80}{1000}} = \sqrt{\frac{3(7.502) + 4.382}{1000}} = \sqrt{0.0269} = 0.164$

- **ERM** would choose $\mathcal{H}_5$ (lowest training error).
- **SRM** chooses $\mathcal{H}_3$ (best balance). ✅

---

## 9. Gradient Descent and Variants

> 💡 **In simple words:** To find the lowest point of a valley while blindfolded, feel the slope under your feet and take a small step **downhill**. Repeat.

### 9.1 Gradient Descent

```math
\theta \leftarrow \theta - \eta\,\nabla J(\theta)
```

- $\nabla J$ = gradient (slope) · $\eta$ = learning rate (step size).
- $\eta$ too small → very slow. $\eta$ too large → jumps around or **diverges**.

| Type | Data used per step | Notes |
|---|---|---|
| **Batch GD** | All examples | Smooth but slow for big data |
| **Stochastic GD (SGD)** | 1 random example | Fast but noisy |
| **Mini-batch GD** | Small batch (32–256) | Best of both — used in practice |

### 9.2 Momentum

> 💡 Like a ball rolling downhill — it **builds up speed** in a consistent direction and doesn't zig-zag as much.

```math
v_t = \beta v_{t-1} + \eta\,\nabla J(\theta) \qquad \theta \leftarrow \theta - v_t
```

($\beta$ ≈ 0.9 = how much past speed is kept.)

### 9.3 RMSprop

> 💡 Gives **each parameter its own step size**: parameters with big gradients take smaller steps, and vice versa.

```math
E_t = \rho\,E_{t-1} + (1 - \rho)\,g_t^2 \qquad \theta \leftarrow \theta - \frac{\eta}{\sqrt{E_t + \epsilon}}\,g_t
```

($g_t$ = current gradient, $\rho \approx 0.9$, $\epsilon$ = tiny number to avoid dividing by 0.)

### 9.4 Adam (Momentum + RMSprop) — for reference

```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2
```

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \qquad \theta \leftarrow \theta - \frac{\eta\,\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

The most popular optimizer today (defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$).

### 📝 Example 9.1 — Compare Optimizers

Minimise $J(\theta) = \theta^2$ (gradient $= 2\theta$). Start $\theta = 5$, $\eta = 0.1$.

**Plain GD:** $\theta \leftarrow \theta - 0.1(2\theta) = 0.8\,\theta$

| Step | $\theta$ |
|---|---|
| 0 | 5 |
| 1 | 4 |
| 2 | 3.2 |
| 3 | 2.56 |

**Momentum** ($\beta = 0.9$, $v = 0$):

| Step | Gradient | $v = 0.9v + 0.1g$ | $\theta = \theta - v$ |
|---|---|---|---|
| 1 | 10 | 1.0 | 4.0 |
| 2 | 8 | 0.9 + 0.8 = 1.7 | 2.3 |
| 3 | 4.6 | 1.53 + 0.46 = 1.99 | 0.31 |

Momentum gets much closer to 0 in 3 steps (0.31 vs 2.56).

**RMSprop** ($\rho = 0.9$, $E = 0$), step 1:

```math
g = 10, \quad E = 0.1 \times 10^2 = 10, \quad \theta = 5 - \frac{0.1}{\sqrt{10}} \times 10 = 5 - 0.316 = 4.684
```

Step 2: $g = 9.37$, $E = 0.9(10) + 0.1(87.8) = 17.78$, $\theta = 4.684 - \frac{0.1 \times 9.37}{4.216} = 4.462$

---

## 10. Convergence Analysis

> 💡 **Convergence** = does gradient descent actually reach the minimum, and **how fast**?

### 10.1 Learning Rate Rule

For $J(\theta) = \frac{a}{2}\theta^2$, each GD step multiplies $\theta$ by $(1 - \eta a)$:

```math
\theta_{t+1} = (1 - \eta a)\,\theta_t
```

It converges only if $|1 - \eta a| < 1$:

```math
0 < \eta < \frac{2}{a}
```

In general, $a$ is replaced by $L$ (the largest curvature of the function). **Safe choice: $\eta \leq 1/L$.**

### 📝 Example 10.1 — Is the Learning Rate OK?

$J = \theta^2$ → $a = 2$ → converges only if $\eta < 1$. Start $\theta = 5$:

| $\eta$ | Multiply by | $\theta_1$ | $\theta_2$ | $\theta_3$ | Result |
|---|---|---|---|---|---|
| 0.1 | 0.8 | 4 | 3.2 | 2.56 | Slow but converges ✅ |
| 0.5 | 0 | 0 | 0 | 0 | Converges in 1 step ✅ |
| 0.9 | −0.8 | −4 | 3.2 | −2.56 | Zig-zags, converges ✅ |
| 1.1 | −1.2 | −6 | 7.2 | −8.64 | **Diverges** ❌ |

### 10.2 How Fast?

| Situation | Rate | Meaning |
|---|---|---|
| Convex function | Error $\propto \dfrac{1}{T}$ | 10× more steps → 10× smaller error |
| Strongly convex function | Error shrinks by a fixed factor $\left(1 - \dfrac{\mu}{L}\right)$ each step | Very fast (called "linear" convergence) |
| SGD | Error $\propto \dfrac{1}{\sqrt{T}}$ | Slower, needs a decreasing learning rate |

($T$ = number of steps. $\kappa = L/\mu$ is the **condition number** — large $\kappa$ means a long, narrow valley and slow convergence. Momentum helps a lot here.)

**For SGD to converge**, the learning rate should shrink over time, e.g. $\eta_t = \eta_0 / t$.

<details>
<summary>📘 Optional: Convergence bound formula</summary>

For a convex, L-smooth function with $\eta = 1/L$:

```math
J(\theta_T) - J^{\ast} \leq \frac{L\,\lVert\theta_0 - \theta^{\ast}\rVert^2}{2T}
```

Example: $L = 4$, starting distance 3, $T = 100$ → error ≤ $\frac{4 \times 9}{200} = 0.18$.

</details>

---

## ✅ Quick Revision

| Concept | Formula / Key point |
|---|---|
| VC dimension | Largest number of points that can be shattered |
| VC of line in 2D / hyperplane in $d$-D | 3 / $d + 1$ |
| VC bound | $\text{Test} \leq \text{Train} + \sqrt{\frac{h(\ln\frac{2m}{h}+1) + \ln\frac{4}{\delta}}{m}}$ |
| Rademacher | $\mathbb{E}_\sigma[\max_h\frac{1}{m}\sum\sigma_ih(x_i)]$ |
| Bias–variance | Error = Bias² + Variance + Noise |
| Ridge | $+\lambda\sum w^2$; $\mathbf{w} = (X^TX+\lambda I)^{-1}X^T\mathbf{y}$ |
| Lasso | $+\lambda\sum\lvert w\rvert$ (makes weights 0) |
| k-fold CV | $\frac{1}{k}\sum E_i$ |
| SRM | Minimise training error + complexity penalty |
| Gradient descent | $\theta \leftarrow \theta - \eta\nabla J$ |
| Momentum | $v = \beta v + \eta g$; $\theta \leftarrow \theta - v$ |
| RMSprop | $E = \rho E + (1-\rho)g^2$; $\theta \leftarrow \theta - \frac{\eta g}{\sqrt{E+\epsilon}}$ |
| Stable learning rate | $0 < \eta < 2/L$ (safe: $\eta \leq 1/L$) |

**Likely exam questions:** Define VC dimension, show VC of a 2D line = 3 · Bias–variance trade-off with numerical · Ridge vs Lasso · k-fold CV · ERM vs SRM · GD vs Momentum vs RMSprop numerical · Effect of learning rate.

---

[⬅ Unit V](Unit-5-Reinforcement-Learning.md) | [Back to Index](README.md)
