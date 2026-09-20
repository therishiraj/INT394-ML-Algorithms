# Unit II — Classification & Probabilistic Approaches

[⬅ Back to Index](README.md)

**In this unit:** what classification is → decision boundaries → linear classifiers (Perceptron, Logistic Regression) → handling many classes → Bayes theorem → Naïve Bayes → Bayesian decision theory.

---

## 1. Overview of Classification

> 💡 **In simple words:** Classification means putting things into **categories**. Is this email spam or not? Is this digit a 0, 1, … or 9?

| Type | Meaning | Example |
|---|---|---|
| Binary | 2 classes | Spam / Not spam |
| Multi-class | More than 2 classes, pick one | Digit 0–9 |
| Multi-label | Can belong to several classes at once | A movie that is both *Action* and *Comedy* |

**Steps:** collect labelled data → clean it → split into train/test → train the model → test it → use it.

### 1.1 Measuring a Classifier — Confusion Matrix

| | Predicted **Yes** | Predicted **No** |
|---|---|---|
| **Actually Yes** | TP (True Positive) | FN (False Negative) |
| **Actually No** | FP (False Positive) | TN (True Negative) |

```math
\text{Accuracy} = \frac{TP + TN}{\text{Total}} \qquad \text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}
```

```math
F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

- **Precision:** of everything I *said* was Yes, how many really were?
- **Recall:** of everything that *really* was Yes, how many did I catch?

### 📝 Example 1.1

TP = 30, FN = 10, FP = 5, TN = 55 (total 100).

```math
\text{Accuracy} = \frac{85}{100} = 0.85, \quad \text{Precision} = \frac{30}{35} = 0.857, \quad \text{Recall} = \frac{30}{40} = 0.75
```

```math
F_1 = \frac{2(0.857)(0.75)}{0.857 + 0.75} = 0.80
```

---

## 2. Decision Boundaries

> 💡 **In simple words:** The decision boundary is the **line (or curve) that separates the classes**. One side = class A, other side = class B.

### 2.1 Linear Boundary

In 2D it is a straight line; in 3D a plane; in more dimensions a "hyperplane":

```math
w_1x_1 + w_2x_2 + b = 0 \qquad \text{(general form: } \mathbf{w}^T\mathbf{x} + b = 0\text{)}
```

- If $w_1x_1 + w_2x_2 + b > 0$ → **Class +1**
- If $w_1x_1 + w_2x_2 + b < 0$ → **Class −1**
- $\mathbf{w}$ decides the **direction** of the line (it is perpendicular to it), $b$ **shifts** it.

**Distance of a point from the line:**

```math
d = \frac{w_1x_1 + w_2x_2 + b}{\sqrt{w_1^2 + w_2^2}}
```

### 2.2 Properties of Decision Boundaries

| Property | Meaning |
|---|---|
| Linear / non-linear | Straight line (logistic regression) vs curve (KNN, trees, neural nets) |
| Margin | Gap between the boundary and the nearest points — **bigger margin = better generalization** (SVM idea) |
| Linearly separable | A straight line can perfectly separate the classes. **XOR is not** linearly separable |
| Complexity | Very wiggly boundary → overfitting; too simple → underfitting |
| Shape by model | Decision tree → staircase (axis-parallel); 1-NN → piecewise; Gaussian Bayes → curves |

**SVM margin** (for reference):

```math
\text{Margin} = \frac{2}{\lVert\mathbf{w}\rVert} = \frac{2}{\sqrt{w_1^2 + w_2^2 + \dots}}
```

### 📝 Example 2.1

Line: $3x_1 + 4x_2 - 10 = 0$. Classify point $(4, 3)$.

```math
3(4) + 4(3) - 10 = 14 > 0 \Rightarrow \text{Class } +1
```

```math
d = \frac{14}{\sqrt{9 + 16}} = \frac{14}{5} = 2.8 \text{ units}
```

---

## 3. Linear Classifiers

> 💡 **In simple words:** A linear classifier multiplies each feature by a weight, adds them up, and checks if the total is positive or negative.

```math
\hat{y} = \text{sign}(w_1x_1 + w_2x_2 + \dots + w_dx_d + b)
```

### 3.1 Perceptron

**Rule:** output 1 if $z = \mathbf{w}^T\mathbf{x} + b > 0$, else 0. When the prediction is wrong, adjust:

```math
w_i \leftarrow w_i + \eta\,(y - \hat{y})\,x_i \qquad b \leftarrow b + \eta\,(y - \hat{y})
```

- $\eta$ = learning rate, $y - \hat{y}$ = error (0 if correct → no change).
- Works only if the data is **linearly separable** (fails on XOR).

### 📝 Example 3.1 — Perceptron Learns the AND Gate

Start: $w_1 = 1, w_2 = 1, b = 0$, $\eta = 0.5$.

**Epoch 1**

| $x_1$ | $x_2$ | $z$ | $\hat{y}$ | $y$ | error | $w_1$ | $w_2$ | $b$ |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| 0 | 1 | 1 | 1 | 0 | −1 | 1 | 0.5 | −0.5 |
| 1 | 0 | 0.5 | 1 | 0 | −1 | 0.5 | 0.5 | −1 |
| 1 | 1 | 0 | 0 | 1 | +1 | 1 | 1 | −0.5 |

(Row 2 update: $w_2 = 1 + 0.5(-1)(1) = 0.5$, $b = 0 + 0.5(-1) = -0.5$.)

**Epoch 2**

| $x_1$ | $x_2$ | $z$ | $\hat{y}$ | $y$ | error | $w_1$ | $w_2$ | $b$ |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | −0.5 | 0 | 0 | 0 | 1 | 1 | −0.5 |
| 0 | 1 | 0.5 | 1 | 0 | −1 | 1 | 0.5 | −1 |
| 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.5 | −1 |
| 1 | 1 | 0.5 | 1 | 1 | 0 | 1 | 0.5 | −1 |

**Epoch 3:** $z = -1, -0.5, 0, 0.5$ → outputs $0, 0, 0, 1$ ✅ all correct. **Done!**

**Final:** $w_1 = 1$, $w_2 = 0.5$, $b = -1$.

### 3.2 Logistic Regression

> 💡 Same linear score, but squashed into a **probability between 0 and 1** using the sigmoid function.

```math
z = \mathbf{w}^T\mathbf{x} + b \qquad \sigma(z) = \frac{1}{1 + e^{-z}} \qquad P(y = 1) = \sigma(z)
```

- $\sigma(0) = 0.5$. Predict 1 if $\sigma(z) \geq 0.5$ (i.e. $z \geq 0$).
- Useful fact: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

**Loss (binary cross-entropy):**

```math
J = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i\ln\hat{p}_i + (1 - y_i)\ln(1 - \hat{p}_i)\right]
```

**Update (gradient descent):**

```math
w_j \leftarrow w_j - \eta\,(\hat{p} - y)\,x_j \qquad b \leftarrow b - \eta\,(\hat{p} - y)
```

### 📝 Example 3.2 — Logistic Regression

$\mathbf{w} = (0.5, -0.25)$, $b = 0.2$, input $\mathbf{x} = (2, 4)$, true $y = 1$, $\eta = 0.1$.

**Step 1:** $z = 0.5(2) - 0.25(4) + 0.2 = 0.2$

**Step 2:** $\hat{p} = \dfrac{1}{1 + e^{-0.2}} = \dfrac{1}{1.8187} = 0.55$ → predict **class 1**.

**Step 3 (loss):** $J = -\ln(0.55) = 0.598$

**Step 4 (update):** error $= \hat{p} - y = 0.55 - 1 = -0.45$

```math
w_1 = 0.5 - 0.1(-0.45)(2) = 0.59, \quad w_2 = -0.25 - 0.1(-0.45)(4) = -0.07, \quad b = 0.2 - 0.1(-0.45) = 0.245
```

---

## 4. Multi-class Classification: One-vs-All and One-vs-One

> 💡 **In simple words:** Many classifiers only know "yes / no". To handle 3+ classes, we break the problem into several yes/no problems.

### 4.1 One-vs-All (One-vs-Rest)
- Train **one classifier per class**: "Is it class $k$, or anything else?"
- Predict the class whose classifier is **most confident**.

```math
\text{Number of classifiers} = K
```

### 4.2 One-vs-One
- Train **one classifier for every pair** of classes (A vs B, A vs C, …).
- Each classifier **votes**; the class with the most votes wins.

```math
\text{Number of classifiers} = \frac{K(K-1)}{2}
```

| | One-vs-All | One-vs-One |
|---|---|---|
| Classifiers | $K$ | $K(K-1)/2$ |
| Data per classifier | All data | Only 2 classes' data |
| Problem | Imbalanced (1 class vs many) | Many classifiers for large $K$ |
| Prediction | Highest score | Majority vote |

### 📝 Example 4.1

$K = 4$: OvA → 4 classifiers; OvO → $\frac{4 \times 3}{2} = 6$ classifiers.
$K = 10$: OvA → 10; OvO → 45.

### 📝 Example 4.2 — OvA Prediction

Scores: Cat = 0.62, **Dog = 0.81**, Bird = 0.15 → **Dog**.

### 📝 Example 4.3 — OvO Voting (classes A, B, C)

| Pair | A vs B | A vs C | B vs C |
|---|---|---|---|
| Winner | A | C | C |

Votes: A = 1, B = 0, **C = 2** → **C**.

### 4.3 Softmax (for reference)

Turns scores $z_k$ into probabilities that add up to 1:

```math
P(\text{class } k) = \frac{e^{z_k}}{\sum_j e^{z_j}}
```

Scores $(2, 1, 0.1)$ → $e^z = (7.39, 2.72, 1.11)$, sum 11.21 → probabilities $(0.66, 0.24, 0.10)$.

---

## 5. Bayes Theorem

> 💡 **In simple words:** Update your belief when you see new evidence.

```math
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}
```

In classification terms:

```math
\underbrace{P(\text{class} \mid \text{data})}_{\text{Posterior}} = \frac{\overbrace{P(\text{data} \mid \text{class})}^{\text{Likelihood}} \times \overbrace{P(\text{class})}^{\text{Prior}}}{\underbrace{P(\text{data})}_{\text{Evidence}}}
```

| Term | Meaning |
|---|---|
| Prior | How common the class is *before* seeing data |
| Likelihood | How likely this data is *if* it belongs to the class |
| Posterior | Probability of the class *after* seeing data |
| Evidence | Total probability of the data (same for all classes) |

**Evidence** (total probability):

```math
P(B) = P(B \mid A)P(A) + P(B \mid \text{not } A)P(\text{not } A)
```

**MAP rule:** choose the class with the highest posterior. Since the evidence is the same for every class, just compare **likelihood × prior**.

### 📝 Example 5.1 — Medical Test (very common exam question)

1% of people have a disease. The test catches 99% of sick people, but also wrongly says "positive" for 5% of healthy people. You test positive. Chance you are sick?

```math
P(+) = 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0099 + 0.0495 = 0.0594
```

```math
P(\text{Sick} \mid +) = \frac{0.0099}{0.0594} = 0.167 \approx 17\%
```

Only 17%! Because the disease is rare, most positives are false alarms.

### 📝 Example 5.2 — Which Machine?

Machines M1, M2, M3 make 50%, 30%, 20% of items; defect rates 2%, 3%, 4%. An item is defective — which machine made it?

```math
P(\text{Def}) = 0.5(0.02) + 0.3(0.03) + 0.2(0.04) = 0.010 + 0.009 + 0.008 = 0.027
```

```math
P(M1 \mid \text{Def}) = \frac{0.010}{0.027} = 0.37, \quad P(M2 \mid \text{Def}) = 0.33, \quad P(M3 \mid \text{Def}) = 0.30
```

**Answer: M1.**

---

## 6. Naïve Bayes Classifier

> 💡 **In simple words:** Use Bayes theorem, but assume all features are **independent** of each other (given the class). This "naïve" assumption makes the math very easy.

```math
P(x_1, x_2, \dots, x_d \mid C) = P(x_1 \mid C) \times P(x_2 \mid C) \times \dots \times P(x_d \mid C)
```

**Prediction rule:**

```math
\hat{y} = \arg\max_{C}\ P(C) \prod_{j=1}^{d} P(x_j \mid C)
```

($\prod$ means multiply all the terms.)

**How to get the numbers:**

```math
P(C) = \frac{\text{no. of examples in class } C}{\text{total examples}} \qquad P(x_j = v \mid C) = \frac{\text{no. of class-}C\text{ examples with } x_j = v}{\text{no. of examples in class } C}
```

### 6.1 Laplace Smoothing (fixing zero probabilities)

If a value never appeared with a class, its probability is 0 and the whole product becomes 0. Fix: add 1 to every count.

```math
P(x_j = v \mid C) = \frac{\text{count} + 1}{N_C + V}
```

($N_C$ = examples (or words) in class C, $V$ = number of possible values (or vocabulary size).)

### 6.2 Gaussian Naïve Bayes (numeric features)

```math
P(x \mid C) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
```

($\mu$, $\sigma$ = mean and standard deviation of that feature within class C.)

### 📝 Example 6.1 — Play Tennis (MOST IMPORTANT)

| Day | Outlook | Temp | Humidity | Wind | Play |
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

**Classify:** (Sunny, Cool, High, Strong)

**Step 1 — Priors:** 9 Yes, 5 No → $P(\text{Yes}) = 9/14$, $P(\text{No}) = 5/14$.

**Step 2 — Count from the table:**

| Feature | P(· \| Yes) | P(· \| No) |
|---|---|---|
| Sunny | 2/9 | 3/5 |
| Cool | 3/9 | 1/5 |
| High | 3/9 | 4/5 |
| Strong | 3/9 | 3/5 |

**Step 3 — Multiply:**

```math
\text{Yes: } \frac{9}{14} \times \frac{2}{9} \times \frac{3}{9} \times \frac{3}{9} \times \frac{3}{9} = 0.0053
```

```math
\text{No: } \frac{5}{14} \times \frac{3}{5} \times \frac{1}{5} \times \frac{4}{5} \times \frac{3}{5} = 0.0206
```

**Step 4 — Normalise:**

```math
P(\text{No} \mid \mathbf{x}) = \frac{0.0206}{0.0053 + 0.0206} = 0.795
```

**Answer: Play = No** (79.5%).

### 📝 Example 6.2 — Spam Filter with Laplace Smoothing

| Doc | Words | Class |
|---|---|---|
| 1 | free money | Spam |
| 2 | free offer now | Spam |
| 3 | meeting now | Ham |
| 4 | project meeting money | Ham |

Classify **"free money now"**. Vocabulary = {free, money, offer, now, meeting, project} → $V = 6$. Spam has 5 words, Ham has 5 words. Priors = 0.5 each.

| Word | Spam count | $P(\cdot \mid \text{Spam})$ | Ham count | $P(\cdot \mid \text{Ham})$ |
|---|---|---|---|---|
| free | 2 | $\frac{2+1}{5+6} = 0.273$ | 0 | $\frac{0+1}{11} = 0.091$ |
| money | 1 | $\frac{2}{11} = 0.182$ | 1 | $\frac{2}{11} = 0.182$ |
| now | 1 | $\frac{2}{11} = 0.182$ | 1 | $\frac{2}{11} = 0.182$ |

```math
\text{Spam: } 0.5 \times 0.273 \times 0.182 \times 0.182 = 0.00451 \qquad \text{Ham: } 0.5 \times 0.091 \times 0.182 \times 0.182 = 0.00150
```

**Answer: Spam** (75%). Without smoothing, "free" would give Ham a probability of 0.

### 📝 Example 6.3 — Gaussian Naïve Bayes

Fruit weight $x = 6$. Class A: $\mu = 5.5$, $\sigma = 0.5$, prior 0.4. Class B: $\mu = 7$, $\sigma = 1$, prior 0.6.

```math
P(6 \mid A) = \frac{1}{\sqrt{2\pi(0.25)}}e^{-\frac{(0.5)^2}{0.5}} = 0.798 \times 0.607 = 0.484
```

```math
P(6 \mid B) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}} = 0.399 \times 0.607 = 0.242
```

A: $0.4 \times 0.484 = 0.194$ · B: $0.6 \times 0.242 = 0.145$ → **Class A**.

**Pros:** fast, works well for text, needs little data. **Cons:** independence assumption is rarely true.

---

## 7. Bayesian Decision Theory

> 💡 **In simple words:** Choose the action with the **lowest expected cost**, not just the highest probability. Because some mistakes are worse than others (missing a disease is worse than a false alarm).

**Loss** $\lambda_{ij}$ = cost of taking action $i$ when the true class is $j$.

**Conditional risk** (expected cost of action $\alpha_i$):

```math
R(\alpha_i \mid x) = \sum_{j} \lambda_{ij}\,P(\omega_j \mid x)
```

**Bayes decision rule:** pick the action with the **minimum risk**.

**Two-class form** — decide class 1 if:

```math
\frac{P(\omega_1 \mid x)}{P(\omega_2 \mid x)} > \frac{\lambda_{12} - \lambda_{22}}{\lambda_{21} - \lambda_{11}}
```

**Special case — 0–1 loss** (every mistake costs 1): risk $= 1 - P(\omega_i \mid x)$, so minimum risk = **maximum posterior** (the MAP rule).

### 📝 Example 7.1 — Minimum Risk

$P(\text{Disease} \mid x) = 0.3$, $P(\text{Healthy} \mid x) = 0.7$.

| | True: Disease | True: Healthy |
|---|---|---|
| Say "Disease" | 0 | 1 |
| Say "Healthy" | **10** | 0 |

```math
R(\text{say Disease}) = 0(0.3) + 1(0.7) = 0.7
```

```math
R(\text{say Healthy}) = 10(0.3) + 0(0.7) = 3.0
```

**Decision: say "Disease"** (lower risk), even though disease is less likely! Missing it is 10× more costly.

<details>
<summary>📘 Optional: Decision boundary for two Gaussian classes</summary>

With 1-D Gaussians having equal variance $\sigma^2$, the boundary is:

```math
x^{\ast} = \frac{\mu_1 + \mu_2}{2} + \frac{\sigma^2}{\mu_1 - \mu_2}\ln\frac{P(\omega_2)}{P(\omega_1)}
```

Example: $\mu_1 = 2$, $\mu_2 = 6$, $\sigma^2 = 1$, priors 0.8 and 0.2:

```math
x^{\ast} = 4 + \frac{1}{-4}\ln(0.25) = 4 + 0.347 = 4.347
```

With equal priors the boundary is exactly halfway between the means.

</details>

---

## ✅ Quick Revision

| Concept | Formula |
|---|---|
| Accuracy / Precision / Recall | $\frac{TP+TN}{\text{Total}}$ / $\frac{TP}{TP+FP}$ / $\frac{TP}{TP+FN}$ |
| F1 score | $\frac{2PR}{P+R}$ |
| Linear boundary | $\mathbf{w}^T\mathbf{x} + b = 0$ |
| Distance from line | $\frac{\mathbf{w}^T\mathbf{x}+b}{\lVert\mathbf{w}\rVert}$ |
| SVM margin | $2/\lVert\mathbf{w}\rVert$ |
| Perceptron update | $w \leftarrow w + \eta(y - \hat{y})x$ |
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ |
| Logistic loss | $-[y\ln\hat{p} + (1-y)\ln(1-\hat{p})]$ |
| Logistic update | $w \leftarrow w - \eta(\hat{p} - y)x$ |
| OvA / OvO | $K$ / $\frac{K(K-1)}{2}$ classifiers |
| Softmax | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ |
| Bayes theorem | $P(C\mid x) = \frac{P(x\mid C)P(C)}{P(x)}$ |
| Naïve Bayes | $\arg\max_C P(C)\prod_j P(x_j\mid C)$ |
| Laplace smoothing | $\frac{\text{count}+1}{N_C + V}$ |
| Gaussian likelihood | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}$ |
| Conditional risk | $R(\alpha_i\mid x) = \sum_j\lambda_{ij}P(\omega_j\mid x)$ |

**Likely exam questions:** Naïve Bayes on Play Tennis · Perceptron for AND/OR · Medical-test Bayes problem · OvA vs OvO · Minimum-risk decision.

---

[⬅ Unit I](Unit-1-Foundations.md) | [Back to Index](README.md) | [Next: Unit III ➡](Unit-3-NonParametric-and-Ensembles.md)
