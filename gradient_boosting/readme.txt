# Gradient Boosting Machine

Before learning Gradient Boosting, you should understand:

1. Decision Trees
2. Ensemble Learning
3. Bagging
4. Boosting
5. AdaBoost

Gradient Boosting is essentially an improved version of AdaBoost.

---

# The Main Idea
Imagine a teacher checking your mathematics exam.

### First Attempt
You solve the paper.

Teacher checks it.
Marks: **60/100**

The teacher circles every mistake.
Now you know exactly where you made errors.

---

### Second Attempt

Instead of solving the whole paper again,
you only focus on correcting your mistakes.

Marks become:
**75/100**

---

### Third Attempt

Teacher again points out remaining mistakes.
You only work on those mistakes.
Marks become:
**88/100**

---

### Fourth Attempt

Again you fix only the remaining mistakes.
Marks become
**94/100**

---

Eventually

```
Attempt 1 → 60
Attempt 2 → 75
Attempt 3 → 88
Attempt 4 → 94
Attempt 5 → 97
```

Every new attempt is **learning from the previous mistakes.**
This is exactly how Gradient Boosting works.

---

# The Core Principle

Instead of building one perfect model,
Gradient Boosting builds **many weak models (usually shallow decision trees).**
Every new tree tries to **correct the mistakes made by all previous trees.**
Finally,all trees combine together to produce a highly accurate prediction.

---

# What is a Weak Learner?

A weak learner is simply a model that performs only slightly better than random 
guessing.

Example:

```
Decision Tree

Depth = 1

Accuracy = 60%
```

Not very good.

Gradient Boosting says:

"No problem.

Let's build another tree to fix its mistakes."

---

# High-Level Workflow

```
Dataset
   │
   ▼
Tree 1
   │
Errors
   ▼
Tree 2
   │
Remaining Errors
   ▼
Tree 3
   │
Remaining Errors
   ▼
Tree 4
   │
Remaining Errors
   ▼
Final Prediction
```

Notice:

Every tree focuses only on the errors left by previous trees.

---

# Step-by-Step Example

Suppose we want to predict house prices.

Actual Prices

| House | Actual Price |
| ----- | ------------ |
| A     | 100          |
| B     | 200          |
| C     | 300          |
| D     | 400          |

---

## Step 1

First tree predicts

| House | Prediction |
| ----- | ---------- |
| A     | 120        |
| B     | 180        |
| C     | 250        |
| D     | 420        |

Clearly,

there are mistakes.

---

Errors (Residuals)

Residual =

```
Actual − Prediction
```

| House | Error |
| ----- | ----- |
| A     | -20   |
| B     | 20    |
| C     | 50    |
| D     | -20   |

These errors are called **Residuals**.

---

## Step 2

Now,

instead of predicting house prices,

the second tree predicts these residuals.

```
Tree 2 learns

-20

20

50

-20
```

---

Suppose it predicts

```
-15

18

45

-15
```

---

Now update predictions.

New Prediction

```
Old Prediction

+

Learning Rate × Residual Prediction
```

Suppose learning rate = 0.1

For House A

```
120

+

0.1 × (-15)

=

118.5
```

Much closer to 100.

Do this for every house.

Prediction improves.

---

## Step 3

Now compute new residuals.

Again train another tree.

Again improve prediction.

Repeat until stopping criteria.

---

# Why is it Called Gradient Boosting?

There are two parts to the name:

### Boosting

Because each new model improves the previous one by focusing on mistakes.

### Gradient

Because the algorithm minimizes the loss function using the **gradient (direction of steepest decrease)**, similar to gradient descent in optimization.

Instead of directly predicting the target, each new tree approximates the **negative gradient** of the loss function (which, for squared error, is simply the residuals).

---

# Mathematical Intuition (Simple)

Suppose

```
Actual = 200
```

Current prediction

```
170
```

Error

```
200 -170 =30
```

The next tree learns to predict **30**, the correction needed.

Updated prediction

```
170 + 30 =200
```

In practice, the correction is scaled by the learning rate.

---

# Learning Rate

Learning rate determines **how much correction each tree contributes**.

```
New Prediction

=

Old Prediction

+

Learning Rate × Tree Prediction
```

Example

Learning rate = 1

```
170 +30 =200
```

Learning rate = 0.1

```
170 +3 =173
```

Learning rate = 0.01

```
170 +0.3 =170.3
```

### Small learning rate

* Slower learning
* Needs more trees
* Better generalization

### Large learning rate

* Faster learning
* May overfit

Common values:

```
0.01

0.05

0.1
```

---

# Why Use Decision Trees?

Decision trees:

* Handle numerical and categorical data.
* Capture non-linear relationships.
* Require little preprocessing.
* Naturally fit the boosting framework.

Gradient Boosting typically uses **shallow trees** (e.g., depth 3–8) as weak learners.

---

# Important Hyperparameters

| Parameter         | Meaning                                                            |
| ----------------- | ------------------------------------------------------------------ |
| n_estimators      | Number of trees                                                    |
| learning_rate     | Size of each correction                                            |
| max_depth         | Maximum tree depth                                                 |
| min_samples_split | Minimum samples to split a node                                    |
| min_samples_leaf  | Minimum samples in a leaf                                          |
| subsample         | Fraction of data used for each tree (stochastic gradient boosting) |
| max_features      | Number of features considered for each split                       |

---

# Advantages

✅ High accuracy

✅ Handles non-linear data

✅ Works for regression and classification

✅ Feature importance available

✅ Handles missing values better in some implementations (e.g., XGBoost)

---

# Disadvantages

❌ Training can be slow

❌ Sequential training (cannot fully parallelize tree creation)

❌ Sensitive to hyperparameters

❌ Can overfit if too many trees or deep trees are used

---

# Gradient Boosting vs AdaBoost

| AdaBoost                                                     | Gradient Boosting                                              |
| ------------------------------------------------------------ | -------------------------------------------------------------- |
| Focuses on misclassified samples by increasing their weights | Focuses on residual errors or negative gradients               |
| Reweights training examples                                  | Fits new trees to prediction errors                            |
| Mainly designed around classification                        | Supports both classification and regression                    |
| Simpler algorithm                                            | More flexible because it can optimize different loss functions |

---

# Gradient Boosting vs Random Forest

| Random Forest                 | Gradient Boosting                           |
| ----------------------------- | ------------------------------------------- |
| Trees are built independently | Trees are built sequentially                |
| Uses bagging                  | Uses boosting                               |
| Reduces variance              | Reduces bias (and can also reduce variance) |
| Faster to train               | Slower to train                             |
| Less prone to overfitting     | Can overfit if not tuned                    |

---

# Interview Questions with Answers

## 1. What is Gradient Boosting?

**Answer:**
Gradient Boosting is an ensemble learning algorithm that builds multiple weak learners 
sequentially. Each new model learns from the errors of previous models to improve overall 
prediction accuracy by minimizing a loss function.

---

## 2. Why is it called "Gradient" Boosting?

**Answer:**
The algorithm minimizes a loss function by fitting each new tree to the **negative gradient** 
of that loss. For squared error regression, this negative gradient is simply the residual 
(actual − predicted).

---

## 3. Why are decision trees used in Gradient Boosting?

**Answer:**
Decision trees can model complex, non-linear relationships, require little preprocessing, 
and work well as weak learners. Shallow trees also help control overfitting.

---

## 4. What are residuals?

**Answer:**
Residuals are the differences between the actual values and the predicted values.

**Formula:**

```
Residual = Actual − Prediction
```

Each new tree tries to predict these residuals (or more generally, the negative gradients).

---

## 5. What is the learning rate?

**Answer:**
The learning rate controls how much each new tree contributes to the final prediction. 
Smaller learning rates usually require more trees but often improve generalization.

---

## 6. What happens if the learning rate is too high?

**Answer:**
The model may learn too aggressively, overfit the training data, and perform poorly 
on unseen data.

---

## 7. What happens if the learning rate is too low?

**Answer:**
Training becomes slower and requires more trees, but the resulting model often 
generalizes better if enough trees are used.

---

## 8. Is Gradient Boosting a bagging algorithm?

**Answer:**
No. Gradient Boosting is a **boosting** algorithm. Trees are trained sequentially, 
with each tree correcting the mistakes of previous trees.

---

## 9. Can Gradient Boosting perform classification and regression?

**Answer:**
Yes. By choosing an appropriate loss function, Gradient Boosting can solve both 
regression and classification problems.

---

## 10. What are popular implementations of Gradient Boosting?

**Answer:**

* XGBoost
* LightGBM
* CatBoost
* Scikit-learn's GradientBoostingClassifier and GradientBoostingRegressor

These implementations add optimizations such as faster training, regularization, efficient tree growth, and better handling of categorical features.

---