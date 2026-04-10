# Heart Disease Prediction — EDA & Machine Learning

Exploratory data analysis and predictive modeling on the UCI Heart Disease (Cleveland) dataset to identify key clinical risk factors and build classifiers for detecting heart disease.

---

## Overview

This project investigates which physiological and clinical features are most predictive of heart disease using the [UCI Heart Disease Cleveland dataset](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland/data) (303 patients, 14 features). The analysis moves from data cleaning through exploratory visualization, dimensionality reduction, and supervised classification, with a focus on maximizing recall — ensuring no at-risk patient is missed.

---

## Dataset

- **Source:** UCI Machine Learning Repository / Kaggle
- **Samples:** 303 patients
- **Features:** 13 clinical attributes (age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, ST slope, major vessels, thalassemia type)
- **Target:** Binary — presence (1) or absence (0) of heart disease

---

## Project Structure

```
Heart_Disease_Project.ipynb   # Full analysis notebook
README.md
```

---

## Methods

### 1. Data Cleaning
- Replaced `?` placeholder values with `NaN` in `ca` and `thal` columns
- Imputed missing values using column medians
- Converted all columns to numeric types
- Removed outliers using the IQR method (303 → 216 samples)

### 2. Exploratory Data Analysis
- Target variable distribution (balanced: ~54% disease present)
- Age and sex distributions across outcomes
- Chest pain type, exercise indicators, and resting BP vs. heart disease
- Correlation heatmap and feature ranking by absolute correlation with the target
- Bivariate interaction scatter plots for key feature pairs

**Top correlated features with target:**
| Feature | Correlation |
|---|---|
| `ca` (major vessels) | 0.521 |
| `thal` (thalassemia) | 0.507 |
| `oldpeak` (ST depression) | 0.504 |
| `thalach` (max heart rate) | 0.415 |
| `cp` (chest pain type) | 0.407 |

### 3. Principal Component Analysis (PCA)
- Standardized all features before PCA
- PC1 (~24%) driven by exercise-induced ischemia markers (`oldpeak`, `slope`, `exang`)
- PC2 (~12%) driven by demographics and lipid-related features (`age`, `sex`, `chol`)
- 10 components required to explain 90% of the total variance
- 2D and 3D projections show partial class separation between healthy and diseased patients

### 4. Logistic Regression (Baseline)
- Hyperparameter tuning via `GridSearchCV` (L1/L2 penalty, C range, 5-fold CV)
- Best params: `C=0.1`, `penalty=l1`, `solver=liblinear`
- Evaluated with confusion matrix, ROC curve, precision-recall curve, and learning curves
- Recursive Feature Elimination (RFE) reduced features from 13 to 8 with no loss in AUC

### 5. Ensemble Classifier (Soft Voting)
- Combined accuracy-optimized and recall-optimized logistic regression models
- Threshold tuned to achieve **Recall = 1.00** (no missed diagnoses) at threshold ≈ 0.23
- AUC ≈ 0.998, Average Precision ≈ 0.999

---

## Results

| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| Logistic Regression | ~0.82 | ~0.78 | ~0.83 | ~0.95 |
| Logistic (RFE) | ~0.82 | ~0.78 | ~0.83 | ~0.95 |
| Voting Ensemble | ~0.82 | ~0.60 | **1.00** | **~0.998** |

The Voting Ensemble at threshold 0.23 achieves perfect recall — all heart disease cases detected — making it suitable for clinical screening where false negatives carry the highest cost.

---

## Tools & Libraries

- Python 3
- pandas, numpy
- scikit-learn (LogisticRegression, GridSearchCV, RFE, PCA, VotingClassifier)
- matplotlib, seaborn, plotly

---

## Key Findings

- High `oldpeak` (ST depression) and low `thalach` (max heart rate) are the strongest individual indicators of heart disease
- Asymptomatic chest pain (`cp = 3`) is paradoxically most associated with disease presence
- Exercise-induced angina (`exang = 1`) strongly predicts positive diagnosis
- PCA confirms a shared cardiovascular stress pattern captured in the first 3 components
- A soft-voting ensemble with a lowered decision threshold can flag every at-risk patient while maintaining roughly 60% precision.
-  This means fewer missed diagnoses at the cost of some false alarms, an acceptable trade-off in a clinical screening context.
