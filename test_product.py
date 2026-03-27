"""
test_product.py - Nutri-Score Prediction Tester
================================================
Supports two model sources:
  1. Full Pipeline model (from run_epics.py)    -> saved_models/
  2. RF-Only model (from train_random_forest.py) -> saved_models_rf/

Usage:
    python test_product.py
"""

import os
import sys
import numpy as np

try:
    import joblib
except ImportError:
    print("ERROR: joblib not installed. Run: pip install joblib")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Model sources
# ============================================================
MODEL_SOURCES = {
    "1": {
        "name": "Full Pipeline (run_epics.py)",
        "dir": os.path.join(script_dir, 'saved_models'),
        "description": "Trained with all 6 models, uses Random Forest for prediction",
        "train_cmd": "python run_epics.py"
    },
    "2": {
        "name": "RF-Only (train_random_forest.py)",
        "dir": os.path.join(script_dir, 'saved_models_rf'),
        "description": "Dedicated Random Forest with 200 trees, tuned hyperparameters",
        "train_cmd": "python train_random_forest.py"
    },
}

# Global model references
rf_model = None
le = None
features = None
current_model_name = None


def load_model(source_key):
    """Load model artifacts from the specified source."""
    global rf_model, le, features, current_model_name

    source = MODEL_SOURCES[source_key]
    model_dir = source["dir"]

    if not os.path.exists(model_dir):
        print(f"\n  ERROR: Model not found at: {model_dir}")
        print(f"  Please run '{source['train_cmd']}' first to train the model.")
        return False

    print(f"\n  Loading model from: {os.path.basename(model_dir)}/")

    rf_model = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
    le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    features = joblib.load(os.path.join(model_dir, 'feature_list.pkl'))
    current_model_name = source["name"]

    print(f"  Model loaded: {current_model_name}")
    print(f"  Classes: {list(le.classes_)}")

    # Show training info if available (RF-only model saves this)
    info_path = os.path.join(model_dir, 'training_info.pkl')
    if os.path.exists(info_path):
        info = joblib.load(info_path)
        print(f"  Accuracy: {info['accuracy']*100:.2f}%")
        print(f"  Trees: {info['n_estimators']} | Train samples: {info['train_samples']}")

    return True


# ============================================================
# Preset food products for quick demo
# ============================================================
PRESET_PRODUCTS = {
    "1": {
        "name": "Coca-Cola (per 100ml)",
        "values": [42, 0, 10.6, 0, 0, 0.01, 0, 10.6],
        "expected": "E"
    },
    "2": {
        "name": "Apple (raw, per 100g)",
        "values": [52, 0.03, 10.4, 2.4, 0.26, 0.0, 85, 13.8],
        "expected": "A"
    },
    "3": {
        "name": "Whole Milk (per 100ml)",
        "values": [61, 1.9, 5.0, 0, 3.2, 0.04, 0, 4.8],
        "expected": "B"
    },
    "4": {
        "name": "Frozen Pizza (per 100g)",
        "values": [250, 5.0, 3.5, 1.5, 10.0, 1.2, 5, 30.0],
        "expected": "D"
    },
    "5": {
        "name": "Chocolate Bar (per 100g)",
        "values": [535, 18.0, 48.0, 3.4, 7.6, 0.12, 0, 56.0],
        "expected": "E"
    },
    "6": {
        "name": "Greek Yogurt (per 100g)",
        "values": [97, 3.0, 3.6, 0, 9.0, 0.07, 0, 3.6],
        "expected": "A/B"
    },
    "7": {
        "name": "White Bread (per 100g)",
        "values": [265, 0.7, 5.0, 2.7, 9.0, 1.0, 0, 49.0],
        "expected": "C/D"
    },
    "8": {
        "name": "Orange Juice (per 100ml)",
        "values": [45, 0.03, 8.4, 0.2, 0.7, 0.0, 100, 10.4],
        "expected": "B/C"
    },
}

# Feature labels for user-friendly input
FEATURE_LABELS = {
    "energy-kcal_100g":   "Energy (kcal per 100g)",
    "energy_100g":        "Energy (kcal per 100g)",
    "saturated-fat_100g": "Saturated Fat (g per 100g)",
    "sugars_100g":        "Sugars (g per 100g)",
    "fiber_100g":         "Fiber (g per 100g)",
    "proteins_100g":      "Proteins (g per 100g)",
    "salt_100g":          "Salt (g per 100g)",
    "fruits-vegetables-nuts-estimate_100g": "Fruits/Vegetables/Nuts estimate (%)",
    "carbohydrates_100g": "Carbohydrates (g per 100g)",
}

# Nutri-Score grade descriptions
GRADE_INFO = {
    "a": {"grade": "A", "emoji": "🟢", "label": "Excellent nutritional quality"},
    "b": {"grade": "B", "emoji": "🟡", "label": "Good nutritional quality"},
    "c": {"grade": "C", "emoji": "🟠", "label": "Average nutritional quality"},
    "d": {"grade": "D", "emoji": "🟠", "label": "Poor nutritional quality"},
    "e": {"grade": "E", "emoji": "🔴", "label": "Bad nutritional quality"},
    "not-applicable": {"grade": "N/A", "emoji": "⚪", "label": "Nutri-Score not applicable for this product"},
    "unknown":        {"grade": "N/A", "emoji": "⚪", "label": "Insufficient data to determine grade"},
}


def get_grade_display(raw_grade):
    """Convert any grade format to a display-friendly result."""
    grade_str = str(raw_grade).strip().lower()

    # Direct letter match (a, b, c, d, e, not-applicable, unknown)
    if grade_str in GRADE_INFO:
        return GRADE_INFO[grade_str]

    # Numeric index match
    try:
        idx = int(float(grade_str))
        classes = list(le.classes_)
        if 0 <= idx < len(classes):
            letter = str(classes[idx]).strip().lower()
            if letter in GRADE_INFO:
                return GRADE_INFO[letter]
        grade_letters = ['a', 'b', 'c', 'd', 'e']
        if 0 <= idx < len(grade_letters):
            return GRADE_INFO[grade_letters[idx]]
    except (ValueError, TypeError):
        pass

    return {"grade": str(raw_grade).upper(), "emoji": "⚪", "label": "Unknown grade"}


def predict_nutriscore(values):
    """Predict Nutri-Score for given nutritional values."""
    input_array = np.array([values])
    prediction = rf_model.predict(input_array)
    grade = le.inverse_transform(prediction)[0]
    return grade


def display_result(product_name, values, grade):
    """Display the prediction result in a formatted way."""
    info = get_grade_display(grade)

    print("\n" + "─" * 50)
    print(f"  Product: {product_name}")
    print(f"  Model: {current_model_name}")
    print("─" * 50)
    print(f"  Nutritional Values (per 100g):")

    for i, feature in enumerate(features):
        label = FEATURE_LABELS.get(feature, feature)
        print(f"    {label}: {values[i]}")

    print("─" * 50)
    print(f"  {info['emoji']}  PREDICTED NUTRI-SCORE:  {info['grade']}")
    print(f"     {info['label']}")
    print("─" * 50)


def run_preset_demo():
    """Run predictions on preset food products."""
    print("\n" + "=" * 60)
    print("  PRESET FOOD PRODUCTS")
    print("=" * 60)

    for key, product in PRESET_PRODUCTS.items():
        print(f"  [{key}] {product['name']} (expected: {product['expected']})")

    print(f"  [A] Run ALL presets")
    print(f"  [0] Go back")

    choice = input("\nSelect a product: ").strip()

    if choice == "0":
        return
    elif choice.upper() == "A":
        print("\n" + "=" * 60)
        print(f"  RUNNING ALL PRESETS  |  Model: {current_model_name}")
        print("=" * 60)
        for key, product in PRESET_PRODUCTS.items():
            grade = predict_nutriscore(product["values"])
            display_result(product["name"], product["values"], grade)
            match = "✅" if str(grade).lower() in product["expected"].lower() else "⚠️"
            print(f"     Expected: {product['expected']}  {match}")
    elif choice in PRESET_PRODUCTS:
        product = PRESET_PRODUCTS[choice]
        grade = predict_nutriscore(product["values"])
        display_result(product["name"], product["values"], grade)
        match = "✅" if str(grade).lower() in product["expected"].lower() else "⚠️"
        print(f"     Expected: {product['expected']}  {match}")
    else:
        print("Invalid choice!")


def run_custom_prediction():
    """Let the user enter custom nutritional values."""
    print("\n" + "=" * 60)
    print("  CUSTOM PRODUCT PREDICTION")
    print("=" * 60)
    print("  Enter nutritional values per 100g of the product.\n")

    product_name = input("  Product name (optional): ").strip() or "Custom Product"
    values = []

    for feature in features:
        label = FEATURE_LABELS.get(feature, feature)
        while True:
            try:
                val = float(input(f"  {label}: "))
                values.append(val)
                break
            except ValueError:
                print("  Please enter a valid number!")

    grade = predict_nutriscore(values)
    display_result(product_name, values, grade)


def switch_model():
    """Let the user switch between model sources."""
    print("\n" + "=" * 60)
    print("  SWITCH MODEL")
    print("=" * 60)
    print(f"  Currently using: {current_model_name}\n")

    for key, source in MODEL_SOURCES.items():
        exists = "✅" if os.path.exists(source["dir"]) else "❌ (not trained)"
        active = " ◄ ACTIVE" if source["name"] == current_model_name else ""
        print(f"  [{key}] {source['name']} {exists}{active}")
        print(f"      {source['description']}")

    print(f"  [0] Cancel")

    choice = input("\nSelect model: ").strip()

    if choice == "0":
        return
    elif choice in MODEL_SOURCES:
        if load_model(choice):
            print(f"\n  Switched to: {current_model_name}")
        else:
            print("  Failed to load model. Keeping current model.")
    else:
        print("Invalid choice!")


def main():
    """Main interactive loop."""

    # ── Initial model selection ──
    print("=" * 60)
    print("  NUTRI-SCORE PREDICTOR")
    print("=" * 60)

    # Check which models are available
    available = {k: v for k, v in MODEL_SOURCES.items() if os.path.exists(v["dir"])}

    if not available:
        print("\n  ERROR: No trained models found!")
        print("\n  Please run one of these first:")
        print("    python run_epics.py              (full pipeline)")
        print("    python train_random_forest.py     (RF only, faster)")
        sys.exit(1)

    if len(available) == 1:
        # Only one model available, auto-load it
        key = list(available.keys())[0]
        load_model(key)
    else:
        # Both available, let user choose
        print("\n  Multiple trained models found! Choose one:\n")
        for key, source in MODEL_SOURCES.items():
            exists = "✅" if key in available else "❌"
            print(f"  [{key}] {source['name']} {exists}")
            print(f"      {source['description']}")

        while True:
            choice = input("\nSelect model: ").strip()
            if choice in available:
                load_model(choice)
                break
            else:
                print("Invalid choice or model not trained!")

    # ── Main menu loop ──
    while True:
        print("\n" + "=" * 60)
        print(f"  MENU  |  Model: {current_model_name}")
        print("=" * 60)
        print("  [1] Test with PRESET food products (quick demo)")
        print("  [2] Enter CUSTOM product nutritional values")
        print("  [3] Switch MODEL")
        print("  [4] Exit")

        choice = input("\nYour choice: ").strip()

        if choice == "1":
            run_preset_demo()
        elif choice == "2":
            run_custom_prediction()
        elif choice == "3":
            switch_model()
        elif choice == "4":
            print("\nGoodbye! 🍎")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
