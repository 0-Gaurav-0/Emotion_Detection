import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# === SETTINGS ===
DATASET_PATH = "female_emotion_dataset.csv"
LABEL_COLUMN = "emotion"
MODEL_SAVE_PATH = "emotion_model.pkl"

# === LOAD DATASET ===
print(f"📂 Loading dataset from {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)

if LABEL_COLUMN not in df.columns:
    raise ValueError(f"❌ '{LABEL_COLUMN}' column not found in CSV.")

print(f"✅ Dataset loaded with {len(df)} samples and {len(df.columns)} columns.")

# === SPLIT FEATURES & LABELS ===
X = df.drop(columns=[LABEL_COLUMN])
y = df[LABEL_COLUMN]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === SCALE FEATURES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === FEATURE SELECTION ===
print("🔍 Selecting best features...")
selector = SelectKBest(score_func=f_classif, k="all")  # change k to limit features
X_selected = selector.fit_transform(X_scaled, y_encoded)

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === DEFINE MODELS & PARAMETERS FOR GRID SEARCH ===
param_grid = {
    "RandomForest": {
        "n_estimators": [200, 300, 400],
        "max_depth": [None, 10, 15],
        "min_samples_split": [2, 5]
    },
    "GradientBoosting": {
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    },
    "XGBoost": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
        "subsample": [0.8, 1.0]
    }
}

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss", use_label_encoder=False)
}

# === CROSS VALIDATION SETUP ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_acc = 0
best_model_name = ""

for name, model in models.items():
    print(f"\n🚀 Tuning {name}...")
    grid = GridSearchCV(model, param_grid[name], cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_estimator = grid.best_estimator_

    # Evaluate on test set
    y_pred = best_estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Best Params for {name}: {grid.best_params_}")
    print(f"📊 {name} Accuracy: {acc * 100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_model = best_estimator
        best_model_name = name

# === SAVE BEST MODEL + SCALER + LABEL ENCODER + SELECTOR ===
with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump({
        "model": best_model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "selector": selector
    }, f)

print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {best_acc * 100:.2f}%")
print(f"💾 Model saved as '{MODEL_SAVE_PATH}'.")
