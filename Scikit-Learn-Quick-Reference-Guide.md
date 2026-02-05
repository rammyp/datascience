# Scikit-learn Quick Reference Guide

> **With Explanations, Code Examples & Quick Reference**
> Python Machine Learning Fundamentals

---

## 1. Installation & Import

Scikit-learn (sklearn) is Python's most popular machine learning library. It provides simple, consistent APIs for classification, regression, clustering, dimensionality reduction, model selection, and preprocessing. Every algorithm follows the same pattern: create a model, fit it to data, and use it to predict. This consistency makes it easy to swap algorithms without rewriting code.

```bash
pip install scikit-learn
```

```python
import numpy as np
import pandas as pd
from sklearn import datasets                    # built-in datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
```

> ⚡ Scikit-learn is imported as `sklearn`. You almost never do `import sklearn` directly — instead, import specific modules.

---

## 2. The Scikit-learn Workflow

Every machine learning task in sklearn follows the same 5-step pattern. This consistency is one of sklearn's greatest strengths — once you learn the pattern, every algorithm works the same way.

```python
# Step 1: Load / prepare data
X = df[['feature1', 'feature2']]    # features (input)
y = df['target']                     # target (output)

# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Create a model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Step 4: Train (fit) the model
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

> ⚡ `random_state=42` makes results reproducible. Without it, every run gives different train/test splits.

---

## 3. Built-in Datasets

Sklearn includes several toy datasets for learning and testing. These are small, clean, and well-documented — perfect for experimenting before using your own data.

```python
from sklearn import datasets

# Classification datasets
iris = datasets.load_iris()              # 150 flowers, 4 features, 3 classes
digits = datasets.load_digits()          # 1797 handwritten digits, 64 features
wine = datasets.load_wine()              # 178 wines, 13 features, 3 classes
breast_cancer = datasets.load_breast_cancer()  # 569 tumors, 30 features, 2 classes

# Regression datasets
boston = datasets.load_diabetes()         # 442 patients, 10 features
california = datasets.fetch_california_housing()  # 20640 houses

# Access data
X = iris.data            # features (numpy array)
y = iris.target          # labels (numpy array)
names = iris.feature_names    # column names
target_names = iris.target_names  # class names

# Load directly as DataFrame
df = datasets.load_iris(as_frame=True).frame

# Generate synthetic data
X, y = datasets.make_classification(n_samples=1000, n_features=10, random_state=42)
X, y = datasets.make_regression(n_samples=1000, n_features=5, random_state=42)
X, y = datasets.make_blobs(n_samples=300, centers=3, random_state=42)
```

---

## 4. Train-Test Split

Splitting data into training and testing sets is the most fundamental step in machine learning. The model learns from the training set and is evaluated on the test set (data it has never seen). This tells you how well the model will perform on new, real-world data. A typical split is 80% train, 20% test.

```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42      # reproducible split
)

# Stratified split (preserves class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,          # same % of each class in train and test
    random_state=42
)

# Three-way split (train / validation / test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)
# Result: 60% train, 20% validation, 20% test
```

---

## 5. Preprocessing

Real-world data needs to be cleaned and scaled before feeding it to a model. Many algorithms (like SVM and KNN) are sensitive to feature scales — if one feature ranges from 0–1 and another from 0–1000, the larger one dominates. Preprocessing puts all features on equal footing.

### Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler — mean=0, std=1 (most common)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled = scaler.transform(X_test)          # only transform on test!

# MinMaxScaler — range [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# RobustScaler — handles outliers better (uses median/IQR)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

> ⚡ **Critical rule:** Always `fit_transform()` on training data and only `transform()` on test data. Fitting on test data causes data leakage.

### Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Label Encoding — categories → numbers
le = LabelEncoder()
y_encoded = le.fit_transform(['cat', 'dog', 'cat', 'bird'])
# [1, 2, 1, 0]

# One-Hot Encoding — categories → binary columns
ohe = OneHotEncoder(sparse_output=False)
X_encoded = ohe.fit_transform(df[['color']])

# Ordinal Encoding — ordered categories → numbers
oe = OrdinalEncoder(categories=[['S', 'M', 'L', 'XL']])
X_encoded = oe.fit_transform(df[['size']])
```

### Handling Missing Values

```python
from sklearn.impute import SimpleImputer

# Fill with mean
imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X)

# Other strategies
SimpleImputer(strategy='median')
SimpleImputer(strategy='most_frequent')
SimpleImputer(strategy='constant', fill_value=0)
```

---

## 6. Classification Algorithms

Classification predicts which category a sample belongs to (e.g., spam/not spam, dog/cat/bird). Sklearn offers dozens of classifiers, all with the same `.fit()` / `.predict()` interface. Here are the most commonly used ones.

```python
# Logistic Regression — simple, fast, good baseline
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=200)

# K-Nearest Neighbors — classifies based on closest training examples
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

# Decision Tree — learns if/else rules from data
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5)

# Random Forest — ensemble of many decision trees (very popular)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Support Vector Machine — finds optimal boundary between classes
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0)

# Gradient Boosting — powerful ensemble method
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100)

# Naive Bayes — fast, works well with text data
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# All follow the same pattern:
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)   # probability of each class
```

### When to Use Which

| Algorithm | Best For | Speed | Interpretable? |
|-----------|----------|-------|----------------|
| Logistic Regression | Binary classification, baseline | Fast | Yes |
| KNN | Small datasets, few features | Slow on large data | Somewhat |
| Decision Tree | Interpretable rules | Fast | Yes |
| Random Forest | General purpose, robust | Medium | Somewhat |
| SVM | High-dimensional data | Slow on large data | No |
| Gradient Boosting | Best accuracy (often) | Slow | No |
| Naive Bayes | Text classification, fast baseline | Very fast | Yes |

---

## 7. Regression Algorithms

Regression predicts a continuous number (e.g., house price, temperature, stock price). The interface is identical to classification — only the algorithms and metrics differ.

```python
# Linear Regression — simplest, assumes linear relationship
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Ridge Regression — linear + L2 regularization (prevents overfitting)
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)

# Lasso Regression — linear + L1 regularization (feature selection)
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=5)

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)

# SVR — Support Vector Regression
from sklearn.svm import SVR
model = SVR(kernel='rbf')

# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100)

# All follow the same pattern:
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 8. Clustering (Unsupervised)

Clustering groups similar data points together without any labels. Unlike classification, there's no "correct answer" to learn from — the algorithm discovers structure on its own. Clustering is useful for customer segmentation, anomaly detection, and exploratory analysis.

```python
# K-Means — most common, groups into K clusters
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X)
centers = model.cluster_centers_

# DBSCAN — density-based, finds arbitrarily shaped clusters
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)

# Agglomerative — hierarchical, builds tree of clusters
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)

# Finding optimal K (Elbow Method)
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
# Plot inertias — look for the "elbow" bend
```

---

## 9. Evaluation Metrics

Metrics tell you how well your model performs. Using the wrong metric can give a misleading picture. For example, accuracy can be 95% even if the model fails on the class you care about (when classes are imbalanced).

### Classification Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)       # % correct overall
precision_score(y_test, y_pred)      # of predicted positives, how many correct?
recall_score(y_test, y_pred)         # of actual positives, how many found?
f1_score(y_test, y_pred)             # balance of precision and recall
confusion_matrix(y_test, y_pred)     # [[TN, FP], [FN, TP]]
roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # area under ROC

# Full report
print(classification_report(y_test, y_pred))
```

| Metric | Question It Answers |
|--------|-------------------|
| Accuracy | What % of all predictions were correct? |
| Precision | When model says "yes", how often is it right? |
| Recall | Of all actual "yes" cases, how many did model find? |
| F1 Score | Balance between precision and recall |
| AUC-ROC | How well does model separate classes overall? |

### Regression Metrics

```python
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
    r2_score, root_mean_squared_error)

mean_absolute_error(y_test, y_pred)     # average absolute error
mean_squared_error(y_test, y_pred)      # average squared error
root_mean_squared_error(y_test, y_pred) # RMSE
r2_score(y_test, y_pred)               # 0–1, higher is better (% variance explained)
```

---

## 10. Cross-Validation

A single train/test split can be lucky or unlucky. Cross-validation trains and tests the model multiple times on different splits, then averages the results. This gives a much more reliable estimate of model performance. K-fold is the most common: it divides data into K parts, trains on K-1, tests on 1, and rotates.

```python
from sklearn.model_selection import cross_val_score, cross_validate

# Basic cross-validation (5-fold)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")

# Multiple metrics at once
results = cross_validate(model, X, y, cv=5,
    scoring=['accuracy', 'f1', 'precision', 'recall'])

# Stratified K-Fold (preserves class proportions)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)

# Leave-One-Out (use every sample as test once)
from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut())
```

---

## 11. Hyperparameter Tuning

Hyperparameters are settings you choose before training (like `n_neighbors` in KNN or `max_depth` in Decision Trees). Tuning finds the best combination. Grid Search tries every combination; Random Search samples random combinations (faster for large search spaces).

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1          # use all CPU cores
)
grid.fit(X_train, y_train)

print(grid.best_params_)      # best combination
print(grid.best_score_)       # best CV score
best_model = grid.best_estimator_   # trained model with best params
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20)
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,        # try 50 random combinations
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
print(search.best_params_)
```

---

## 12. Pipelines

A Pipeline chains preprocessing steps and a model into a single object. This prevents data leakage (preprocessing is applied correctly during cross-validation), keeps code clean, and makes deployment easier. You can call `.fit()` and `.predict()` on the entire pipeline.

```python
from sklearn.pipeline import Pipeline

# Simple pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)

# Pipeline with multiple steps
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100))
])

# Use pipeline in GridSearch
param_grid = {
    'model__n_estimators': [50, 100, 200],    # double underscore!
    'model__max_depth': [3, 5, 10]
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
```

### Column Transformer (Different preprocessing for different columns)

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),           # scale numbers
    ('cat', OneHotEncoder(), ['city', 'gender'])            # encode categories
])

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)
```

---

## 13. Feature Selection

Not all features are useful. Irrelevant or redundant features can hurt model performance and increase training time. Feature selection identifies the most important features and removes the rest.

```python
# Feature importance from tree-based models
model = RandomForestClassifier().fit(X_train, y_train)
importances = model.feature_importances_

# Sort and display
import pandas as pd
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feat_imp)

# SelectKBest — statistical test
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)       # keep top 5 features
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
X_selected = rfe.fit_transform(X, y)
print(rfe.support_)       # True/False for each feature
print(rfe.ranking_)       # ranking (1 = selected)

# Variance Threshold — remove low-variance features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
```

---

## 14. Dimensionality Reduction

When you have many features, dimensionality reduction compresses them into fewer dimensions while preserving the most important information. PCA is the most common method. It's useful for visualization (reduce to 2D/3D), speeding up training, and removing noise.

```python
# PCA — Principal Component Analysis
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)   # how much info each component holds
print(pca.explained_variance_ratio_.sum())  # total info retained

# Keep 95% of variance (auto-select components)
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
print(f"Reduced to {pca.n_components_} components")

# t-SNE — for visualization only (non-linear)
from sklearn.manifold import TSNE
X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)
```

---

## 15. Model Saving & Loading

After training a model, you need to save it so you can use it later without retraining. `joblib` is the recommended method for sklearn models because it handles NumPy arrays efficiently.

```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')
joblib.dump(pipe, 'pipeline.pkl')     # save entire pipeline

# Load model
model = joblib.load('model.pkl')
y_pred = model.predict(X_new)

# Alternative: pickle
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## 16. Complete Example — End to End

Putting it all together: loading data, preprocessing, training, evaluating, and tuning.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Build pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# 4. Tune hyperparameters
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [5, 10, None]
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

# 5. Evaluate
print(f"Best params: {grid.best_params_}")
print(f"Best CV F1:  {grid.best_score_:.3f}")

y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 6. Save
joblib.dump(grid.best_estimator_, 'best_model.pkl')
```

---

## Quick Reference Cheat Sheet

| Category | Key Functions | Purpose |
|----------|---------------|---------|
| **Split** | `train_test_split, StratifiedKFold` | Divide data for training/testing |
| **Scale** | `StandardScaler, MinMaxScaler, RobustScaler` | Normalize feature ranges |
| **Encode** | `LabelEncoder, OneHotEncoder, OrdinalEncoder` | Convert categories to numbers |
| **Impute** | `SimpleImputer` | Fill missing values |
| **Classify** | `LogisticRegression, RandomForest, SVC, KNN, GradientBoosting` | Predict categories |
| **Regress** | `LinearRegression, Ridge, Lasso, RandomForestRegressor, SVR` | Predict numbers |
| **Cluster** | `KMeans, DBSCAN, AgglomerativeClustering` | Group similar data |
| **Evaluate** | `accuracy, precision, recall, f1, r2, MSE, confusion_matrix` | Measure performance |
| **CV** | `cross_val_score, cross_validate` | Reliable evaluation |
| **Tune** | `GridSearchCV, RandomizedSearchCV` | Find best hyperparameters |
| **Pipeline** | `Pipeline, ColumnTransformer` | Chain steps, prevent leakage |
| **Features** | `SelectKBest, RFE, feature_importances_` | Select important features |
| **Reduce** | `PCA, TSNE` | Compress dimensions |
| **Save** | `joblib.dump, joblib.load` | Persist trained models |

---

## Performance Tips

1. **Always use pipelines** — they prevent data leakage and simplify deployment
2. **Scale your data** — most algorithms perform better with standardized features
3. **Use stratified splits** for imbalanced classification problems
4. **Start simple** — try Logistic Regression or Decision Tree before complex models
5. **Use `n_jobs=-1`** in GridSearch and ensemble methods to use all CPU cores
6. **Cross-validate** — never trust a single train/test split
