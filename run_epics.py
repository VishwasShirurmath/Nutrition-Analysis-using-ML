"""
EPICSproject - Nutri-Score Prediction Pipeline
Converted from EPICSproject.ipynb to run as a standalone Python script.
Uses the local en.openfoodfacts.org.products.csv dataset.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
import sys

# Base directory for saving all outputs
save_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Cell 1: Nutritional Safe Ranges Visualization
# ============================================================
print("=" * 60)
print("STEP 1: Nutritional Safe Ranges Visualization")
print("=" * 60)

nutrients = [
    "Energy (kcal)", "Protein (g)", "Carbohydrates (g)", "Sugars (g)",
    "Fiber (g)", "Saturated Fat (g)", "Unsaturated Fat (g)",
    "Salt (g)", "Calcium (mg)", "Iron (mg)"
]
safe_ranges = [
    (50, 250), (5, 25), (15, 60), (0, 5), (3, 100), (0, 1.5), (5, 20),
    (0, 0.3), (100, 300), (1, 3)
]

plt.figure(figsize=(12, 8))
x = np.arange(len(nutrients))
for i, (low, high) in enumerate(safe_ranges):
    plt.barh(i, high - low, left=low, color='red', edgecolor='black', label='Safe' if i == 0 else "")
plt.yticks(x, nutrients)
plt.xlabel("Values (per 100g)")
plt.title("Nutritional Safe Ranges")
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'plot_safe_ranges.png'), dpi=150)
plt.close()
print("Saved: plot_safe_ranges.png")

# ============================================================
# Cell 2: Load the Open Food Facts CSV file
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Loading Dataset...")
print("=" * 60)

csv_path = os.path.join(save_dir, 'en.openfoodfacts.org.products.csv', 'en.openfoodfacts.org.products.csv')
if not os.path.exists(csv_path):
    print(f"ERROR: CSV file not found at: {csv_path}")
    sys.exit(1)

# The file is ~11.5 GB - way too large to load entirely.
# We only load the columns we need and limit to 500,000 rows.
needed_columns = [
    'energy-kcal_100g', 'energy_100g',
    'saturated-fat_100g', 'sugars_100g', 'fiber_100g',
    'proteins_100g', 'salt_100g',
    'fruits-vegetables-nuts-estimate_100g', 'carbohydrates_100g',
    'nutriscore_grade'
]

# First, read just the header to find which columns exist
header_df = pd.read_csv(csv_path, sep='\t', nrows=0, encoding='utf-8')
available_cols = header_df.columns.tolist()
usecols = [c for c in needed_columns if c in available_cols]
print(f"Available columns to load: {usecols}")

MAX_ROWS = 100000
print(f"Loading first {MAX_ROWS} rows with {len(usecols)} columns...")

df_foodfacts = pd.read_csv(
    csv_path,
    sep='\t',
    encoding='utf-8',
    usecols=usecols,
    nrows=MAX_ROWS,
    low_memory=False,
    on_bad_lines='skip'
)

print(f"Dataset loaded successfully!")
print(f"Shape: {df_foodfacts.shape}")
print(f"\nFirst 5 rows:")
print(df_foodfacts.head())

# ============================================================
# Cell 3: Check dataset info
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Dataset Overview")
print("=" * 60)

print("Shape of the dataset:", df_foodfacts.shape)
print("\nColumns:", df_foodfacts.columns.tolist())

# Check the distribution of the target variable
print("\nDistribution of nutriscore_grade:")
print(df_foodfacts['nutriscore_grade'].value_counts())

# ============================================================
# Cell 4: Feature selection and preprocessing
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Feature Selection & Preprocessing")
print("=" * 60)

# The CSV uses 'energy-kcal_100g' instead of 'energy_100g'
# Check which column name exists
if 'energy_100g' in df_foodfacts.columns:
    energy_col = 'energy_100g'
elif 'energy-kcal_100g' in df_foodfacts.columns:
    energy_col = 'energy-kcal_100g'
else:
    print("WARNING: Neither 'energy_100g' nor 'energy-kcal_100g' found in columns!")
    print("Available energy columns:", [c for c in df_foodfacts.columns if 'energy' in c.lower()])
    energy_col = 'energy_100g'  # fallback

features = [energy_col, 'saturated-fat_100g', 'sugars_100g', 'fiber_100g',
            'proteins_100g', 'salt_100g', 'fruits-vegetables-nuts-estimate_100g', 'carbohydrates_100g']
target = 'nutriscore_grade'

# Some Open Food Facts exports do not include all expected feature columns.
# Create missing columns so downstream selection/modeling does not fail.
missing_feature_cols = [f for f in features if f not in df_foodfacts.columns]
if missing_feature_cols:
    print("WARNING: Missing feature columns in dataset:", missing_feature_cols)
    print("Adding missing feature columns with NaN values before imputation.")
    for feature in missing_feature_cols:
        df_foodfacts[feature] = np.nan

# Check for missing values in selected features and target
print("Missing values in features:")
print(df_foodfacts[features].isnull().sum())
print("\nMissing values in target:")
print(df_foodfacts[target].isnull().sum())

# Impute missing values with mean for numerical features
for feature in features:
    df_foodfacts[feature] = pd.to_numeric(df_foodfacts[feature], errors='coerce')
    mean_value = df_foodfacts[feature].mean()
    if pd.isna(mean_value):
        mean_value = 0.0
    df_foodfacts[feature] = df_foodfacts[feature].fillna(mean_value)

# Drop rows where target is missing
df_foodfacts = df_foodfacts.dropna(subset=[target])

# Keep only valid Nutri-Score grades (a, b, c, d, e) - remove 'not-applicable', 'unknown', etc.
valid_grades = ['a', 'b', 'c', 'd', 'e']
before_filter = len(df_foodfacts)
df_foodfacts = df_foodfacts[df_foodfacts[target].str.lower().isin(valid_grades)]
print(f"\nRemoved {before_filter - len(df_foodfacts)} rows with invalid grades (not-applicable, unknown, etc.)")
print(f"Dataset shape after filtering: {df_foodfacts.shape}")

# ============================================================
# Cell 5: Encode and Split
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Encoding Target & Splitting Data")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
df_foodfacts[target] = le.fit_transform(df_foodfacts[target])
print(f"Label classes: {le.classes_}")

X = df_foodfacts[features]
y = df_foodfacts[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================================
# Cell 6: Scale features
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Scaling Features")
print("=" * 60)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler.")

# ============================================================
# Cell 7: Logistic Regression
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Logistic Regression")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train_scaled, y_train)
y_pred_logregight = logreg.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logregight))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logregight))
print("Classification Report:\n", classification_report(y_test, y_pred_logregight))

# ============================================================
# Cell 8: SVM (sample data if too large for performance)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: SVM Models")
print("=" * 60)

from sklearn.svm import SVC

# SVM is very slow on large datasets, so sample if needed
MAX_SVM_SAMPLES = 20000
if X_train_scaled.shape[0] > MAX_SVM_SAMPLES:
    print(f"Sampling {MAX_SVM_SAMPLES} rows for SVM training (full dataset has {X_train_scaled.shape[0]} rows)...")
    np.random.seed(42)
    svm_idx = np.random.choice(X_train_scaled.shape[0], MAX_SVM_SAMPLES, replace=False)
    X_train_svm = X_train_scaled[svm_idx]
    y_train_svm = y_train.iloc[svm_idx]

    svm_test_idx = np.random.choice(X_test_scaled.shape[0], min(10000, X_test_scaled.shape[0]), replace=False)
    X_test_svm = X_test_scaled[svm_test_idx]
    y_test_svm = y_test.iloc[svm_test_idx]
else:
    X_train_svm = X_train_scaled
    y_train_svm = y_train
    X_test_svm = X_test_scaled
    y_test_svm = y_test

# SVM with linear kernel
print("\nTraining SVM Linear...")
svm_linear = SVC(kernel='linear', class_weight='balanced')
svm_linear.fit(X_train_svm, y_train_svm)
y_pred_svm_linear = svm_linear.predict(X_test_svm)
print("SVM Linear Accuracy:", accuracy_score(y_test_svm, y_pred_svm_linear))
print("Classification Report:\n", classification_report(y_test_svm, y_pred_svm_linear))

# SVM with polynomial kernel
print("\nTraining SVM Polynomial...")
svm_poly = SVC(kernel='poly', degree=3, class_weight='balanced')
svm_poly.fit(X_train_svm, y_train_svm)
y_pred_svm_poly = svm_poly.predict(X_test_svm)
print("SVM Polynomial Accuracy:", accuracy_score(y_test_svm, y_pred_svm_poly))
print("Classification Report:\n", classification_report(y_test_svm, y_pred_svm_poly))

# SVM with RBF kernel
print("\nTraining SVM RBF...")
svm_rbf = SVC(kernel='rbf', class_weight='balanced')
svm_rbf.fit(X_train_svm, y_train_svm)
y_pred_svm_rbf = svm_rbf.predict(X_test_svm)
print("SVM RBF Accuracy:", accuracy_score(y_test_svm, y_pred_svm_rbf))
print("Classification Report:\n", classification_report(y_test_svm, y_pred_svm_rbf))

# ============================================================
# Cell 9: Random Forest
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Random Forest")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# ============================================================
# Cell 10: Decision Tree
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Decision Tree")
print("=" * 60)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# ============================================================
# Cell 11: Plot Decision Tree
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: Decision Tree Visualization")
print("=" * 60)

from sklearn.tree import plot_tree

plt.figure(figsize=(10, 30))
plot_tree(
    dt,
    feature_names=features,
    class_names=list(map(str, le.classes_)),
    filled=True,
    max_depth=3
)
plt.savefig(os.path.join(save_dir, 'plot_decision_tree.png'), dpi=100, bbox_inches='tight')
plt.close()
print("Saved: plot_decision_tree.png")

# ============================================================
# Cell 12: Feature Importances
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: Feature Importances")
print("=" * 60)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'plot_feature_importances.png'), dpi=150)
plt.close()
print("Saved: plot_feature_importances.png")

for i in indices:
    print(f"  {features[i]}: {importances[i]:.4f}")

# ============================================================
# Cell 13: SVM RBF Decision Boundaries with PCA
# ============================================================
print("\n" + "=" * 60)
print("STEP 13: SVM RBF Decision Boundaries (PCA)")
print("=" * 60)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Use sampled data for SVM PCA visualization
if X_train_scaled.shape[0] > MAX_SVM_SAMPLES:
    X_train_pca_svm = pca.transform(X_train_svm)
else:
    X_train_pca_svm = X_train_pca

svm_rbf_pca = SVC(kernel='rbf', class_weight='balanced')
svm_rbf_pca.fit(X_train_pca_svm, y_train_svm)

x_min, x_max = X_train_pca_svm[:, 0].min() - 1, X_train_pca_svm[:, 0].max() + 1
y_min, y_max = X_train_pca_svm[:, 1].min() - 1, X_train_pca_svm[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = svm_rbf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

nutriscore_colors = {
    0: "#008000",  # Green (A)
    1: "#ADFF2F",  # Light Green (B)
    2: "#FFA500",  # Orange (C)
    3: "#FF8C00",  # Light Orange (D)
    4: "#FF0000"   # Red (E)
}

# Only use colors for classes that exist
unique_classes = sorted(set(y_train_svm.unique()))
cmap = mcolors.ListedColormap([nutriscore_colors.get(i, "#808080") for i in unique_classes])

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
scatter = plt.scatter(X_train_pca_svm[:, 0], X_train_pca_svm[:, 1],
                      c=y_train_svm, cmap=cmap, edgecolor='k', linewidth=0.5, s=5)
plt.colorbar(scatter, ticks=unique_classes, label="Nutri-Score Grades")
plt.title("SVM RBF Decision Boundaries with PCA & Nutri-Score Colors")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig(os.path.join(save_dir, 'plot_svm_rbf_pca.png'), dpi=150)
plt.close()
print("Saved: plot_svm_rbf_pca.png")

# ============================================================
# Cell 14: Logistic Regression Decision Boundaries with PCA
# ============================================================
print("\n" + "=" * 60)
print("STEP 14: Logistic Regression Decision Boundaries (PCA)")
print("=" * 60)

logreg_pca = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg_pca.fit(X_train_pca_svm, y_train_svm)

Z_logreg = logreg_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z_logreg = Z_logreg.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z_logreg, alpha=0.4, cmap=cmap)
scatter = plt.scatter(X_train_pca_svm[:, 0], X_train_pca_svm[:, 1],
                      c=y_train_svm, cmap=cmap, edgecolor='k', linewidth=0.5, s=5)
plt.colorbar(scatter, ticks=unique_classes, label="Nutri-Score Grades")
plt.title("Logistic Regression Decision Boundaries with PCA & Nutri-Score Colors")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig(os.path.join(save_dir, 'plot_logreg_pca.png'), dpi=150)
plt.close()
print("Saved: plot_logreg_pca.png")

# ============================================================
# Cell 15: Model Accuracies Summary
# ============================================================
print("\n" + "=" * 60)
print("STEP 15: Model Accuracies Summary")
print("=" * 60)

accuracies = {
    'Logistic Regression': accuracy_score(y_test, y_pred_logregight),
    'SVM Linear': accuracy_score(y_test_svm, y_pred_svm_linear),
    'SVM Polynomial': accuracy_score(y_test_svm, y_pred_svm_poly),
    'SVM RBF': accuracy_score(y_test_svm, y_pred_svm_rbf),
    'Random Forest': accuracy_score(y_test, y_pred_rf),
    'Decision Tree': accuracy_score(y_test, y_pred_dt)
}

print("\nModel Accuracies:")
for model, acc in accuracies.items():
    print(f"  {model}: {acc:.4f}")

# ============================================================
# Cell 16: Confusion Matrix Plots
# ============================================================
print("\n" + "=" * 60)
print("STEP 16: Confusion Matrix Plots")
print("=" * 60)

import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {os.path.basename(filename)}")

plot_confusion_matrix(y_test, y_pred_logregight, "Logistic Regression",
                      os.path.join(save_dir, "cm_logistic_regression.png"))
plot_confusion_matrix(y_test_svm, y_pred_svm_linear, "SVM Linear",
                      os.path.join(save_dir, "cm_svm_linear.png"))
plot_confusion_matrix(y_test_svm, y_pred_svm_poly, "SVM Polynomial",
                      os.path.join(save_dir, "cm_svm_poly.png"))
plot_confusion_matrix(y_test_svm, y_pred_svm_rbf, "SVM RBF",
                      os.path.join(save_dir, "cm_svm_rbf.png"))
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest",
                      os.path.join(save_dir, "cm_random_forest.png"))
plot_confusion_matrix(y_test, y_pred_dt, "Decision Tree",
                      os.path.join(save_dir, "cm_decision_tree.png"))

# ============================================================
# Cell 17: Scaler/Model diagnostics
# ============================================================
print("\n" + "=" * 60)
print("STEP 17: Scaler & Model Diagnostics")
print("=" * 60)
print("Scaler mean:", scaler.mean_)
print("Scaler std:", scaler.scale_)
print("Logistic Coefficients shape:", logreg.coef_.shape)
print("Logistic Intercept:", logreg.intercept_)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)
print(f"\nAll plots saved to: {save_dir}")

# ============================================================
# Cell 18: Save trained model artifacts for test_product.py
# ============================================================
print("\n" + "=" * 60)
print("STEP 18: Saving Model Artifacts")
print("=" * 60)

import joblib

model_dir = os.path.join(save_dir, 'saved_models')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(rf, os.path.join(model_dir, 'random_forest_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))
joblib.dump(features, os.path.join(model_dir, 'feature_list.pkl'))

print(f"Saved model artifacts to: {model_dir}")
print("  - random_forest_model.pkl")
print("  - scaler.pkl")
print("  - label_encoder.pkl")
print("  - feature_list.pkl")
print("\nYou can now run 'python test_product.py' to predict Nutri-Score for any product!")
