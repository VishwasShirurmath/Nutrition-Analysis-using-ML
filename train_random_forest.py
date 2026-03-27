"""
train_random_forest.py - Fast Random Forest Only Training
==========================================================
Trains ONLY the Random Forest model (the best performer) on the
Open Food Facts dataset. Much faster than the full pipeline.

Saves model artifacts to 'saved_models_rf/' folder.

Usage:
    python train_random_forest.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import sys
import time

# Base directory
save_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  RANDOM FOREST - FAST TRAINING PIPELINE")
print("=" * 60)

# ============================================================
# Step 1: Load Dataset
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Loading Dataset...")
print("=" * 60)

csv_path = os.path.join(save_dir, 'en.openfoodfacts.org.products.csv', 'en.openfoodfacts.org.products.csv')
if not os.path.exists(csv_path):
    print(f"ERROR: CSV file not found at: {csv_path}")
    sys.exit(1)

needed_columns = [
    'energy-kcal_100g', 'energy_100g',
    'saturated-fat_100g', 'sugars_100g', 'fiber_100g',
    'proteins_100g', 'salt_100g',
    'fruits-vegetables-nuts-estimate_100g', 'carbohydrates_100g',
    'nutriscore_grade'
]

# Read header to check available columns
header_df = pd.read_csv(csv_path, sep='\t', nrows=0, encoding='utf-8')
available_cols = header_df.columns.tolist()
usecols = [c for c in needed_columns if c in available_cols]
print(f"Available columns: {usecols}")

print(f"Loading dataset (sampling 250,000 rows)...")

start_time = time.time()
df = pd.read_csv(
    csv_path,
    sep='\t',
    encoding='utf-8',
    usecols=usecols,
    low_memory=False,
    on_bad_lines='skip'
)
if len(df) > 250000:
    df = df.sample(n=250000, random_state=42)
print(f"Loaded in {time.time() - start_time:.1f}s | Shape: {df.shape}")

# ============================================================
# Step 2: Preprocessing
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)

target = 'nutriscore_grade'

# Determine energy column name
if 'energy_100g' in df.columns:
    energy_col = 'energy_100g'
elif 'energy-kcal_100g' in df.columns:
    energy_col = 'energy-kcal_100g'
else:
    print("ERROR: No energy column found!")
    sys.exit(1)

features = [energy_col, 'saturated-fat_100g', 'sugars_100g', 'fiber_100g',
            'proteins_100g', 'salt_100g', 'fruits-vegetables-nuts-estimate_100g', 'carbohydrates_100g']

# Handle missing values - impute with mean
for feature in features:
    if df[feature].dtype in ['float64', 'int64']:
        df[feature] = df[feature].fillna(df[feature].mean())

# Drop rows with missing target
df = df.dropna(subset=[target])

# Keep only valid grades (a-e)
valid_grades = ['a', 'b', 'c', 'd', 'e']
before = len(df)
df = df[df[target].str.lower().isin(valid_grades)]
print(f"Removed {before - len(df)} invalid grade rows")
print(f"Clean dataset: {df.shape}")
print(f"\nGrade distribution:")
print(df[target].value_counts().sort_index())

# ============================================================
# Step 3: Encode & Split
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Encoding & Splitting")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
df[target] = le.fit_transform(df[target])
print(f"Classes: {list(le.classes_)}")

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# Step 4: Train Random Forest
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Training Random Forest")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Using more trees and tuned hyperparameters for better accuracy
rf = RandomForestClassifier(
    n_estimators=200,          # More trees = better accuracy
    max_depth=None,            # Let trees grow fully
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1                  # Use all CPU cores
)

print("Training with 200 trees (using all CPU cores)...")
start_time = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Training completed in {train_time:.1f}s")

# ============================================================
# Step 5: Evaluate
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Evaluation")
print("=" * 60)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

# Feature importances
print("Feature Importances:")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for i in indices:
    print(f"  {features[i]}: {importances[i]:.4f}")

# ============================================================
# Step 6: Save Model
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Saving Model")
print("=" * 60)

import joblib

model_dir = os.path.join(save_dir, 'saved_models_rf')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(rf, os.path.join(model_dir, 'random_forest_model.pkl'))
joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))
joblib.dump(features, os.path.join(model_dir, 'feature_list.pkl'))

# Save training info for display in test_product.py
training_info = {
    'accuracy': accuracy,
    'n_estimators': 200,
    'train_samples': X_train.shape[0],
    'test_samples': X_test.shape[0],
    'train_time': train_time,
    'classes': list(le.classes_),
}
joblib.dump(training_info, os.path.join(model_dir, 'training_info.pkl'))

print(f"Saved to: {model_dir}")
print("  - random_forest_model.pkl")
print("  - label_encoder.pkl")
print("  - feature_list.pkl")
print("  - training_info.pkl")

print("\n" + "=" * 60)
print("  DONE! Run 'python test_product.py' to test predictions.")
print("=" * 60)
