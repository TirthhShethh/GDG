# model.py
"""
Modeling script for F1 DNF prediction.
Usage:
    python model.py
Assumes f1_dnf_cleaned.csv is present in the same folder.
Outputs: model_results/ folder with metrics, plots, and saved model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay)
import joblib

# Config
INPUT = "f1_dnf_cleaned.csv"
TARGET_COL = "target_finish"   # 1 = finished, 0 = did not finish (adjust if different)
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTDIR = "model_results"
os.makedirs(OUTDIR, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    return df

def prepare_features(df, target_col):
    # If target not present, try to create it from known columns
    if target_col not in df.columns:
        # Try to infer: common columns might be 'status' or 'position' etc.
        # Here we look for 'positionOrder' or 'status' and create a binary target
        if 'positionOrder' in df.columns:
            df[target_col] = df['positionOrder'].apply(lambda x: 1 if not pd.isna(x) and int(x)>0 else 0)
        elif 'status' in df.columns:
            # assume 'Finished' indicates finish; adjust as needed
            df[target_col] = df['status'].apply(lambda s: 1 if str(s).lower().strip() in ['finished','finished'] else 0)
        else:
            raise ValueError(f"Target column {target_col} not found and cannot be inferred automatically.")
    # Drop rows with null target
    df = df[df[target_col].notna()].copy()

    # Basic feature selection: remove id-like columns and target
    drop_candidates = ['resultId','raceId','driverId','constructorId','position','positionText','positionOrder']
    candidates = [c for c in drop_candidates if c in df.columns]
    # Also drop columns that are obviously not features or high cardinality raw text (adjust as needed)
    # Keep numeric and some categorical
    # Auto select numeric and categorical feature lists
    exclude = set(candidates + [target_col])
    feature_cols = [c for c in df.columns if c not in exclude and c not in ['date']]

    # For demonstration, further filter out columns that are likely identifiers
    # Remove columns with too many unique values (very high cardinality)
    filtered = []
    for c in feature_cols:
        if df[c].nunique() / len(df) > 0.9:
            # skip very high-cardinality columns (like unique IDs)
            continue
        filtered.append(c)
    feature_cols = filtered

    # Separate numeric and categorical
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    return X, y, numeric_cols, categorical_cols

def build_pipeline(numeric_cols, categorical_cols):
    # Numeric pipeline: impute then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # Categorical pipeline: impute then one-hot (drop='if_binary' or handle_unknown)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))

    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    # Full pipeline with RandomForest
    pipe = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'))
    ])
    return pipe

def evaluate_model(pipe, X_train, X_test, y_train, y_test):
    # Fit
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X_test)[:,1]
    except:
        pass

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print("Evaluation metrics (test):")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    if roc is not None:
        print(f"ROC AUC: {roc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    # Save classification report
    clf_report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    print("\nClassification Report:")
    print(clf_report)

    # Save metrics to file
    with open(os.path.join(OUTDIR, "metrics.txt"), "w") as f:
        f.write("Evaluation metrics (test):\n")
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")
        if roc is not None:
            f.write(f"ROC AUC: {roc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(clf_report)

    # Plot and save confusion matrix
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix (test)')
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['DNF','Finish'])
    plt.yticks(ticks, ['DNF','Finish'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "confusion_matrix.png"))
    plt.close()

    # ROC curve plot (if probabilities available)
    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.savefig(os.path.join(OUTDIR, "roc_curve.png"))
        plt.close()

    return pipe

def cross_validate(pipe, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, scoring='f1', cv=cv, n_jobs=-1)
    print(f"Cross-validated F1 (5-fold): mean={scores.mean():.4f}, std={scores.std():.4f}")
    with open(os.path.join(OUTDIR, "cv_f1.txt"), "w") as f:
        f.write(f"CV F1 (5-fold): mean={scores.mean():.4f}, std={scores.std():.4f}\n")

def tune_hyperparameters(pipe, X_train, y_train):
    # Example grid: tune number of estimators and max depth
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("Best params from GridSearchCV:", grid.best_params_)
    with open(os.path.join(OUTDIR, "best_params.txt"), "w") as f:
        f.write(str(grid.best_params_))
    return grid.best_estimator_

def save_model(pipe):
    path = os.path.join(OUTDIR, "rf_model.joblib")
    joblib.dump(pipe, path)
    print(f"Saved model to {path}")

def main():
    df = load_data(INPUT)
    X, y, numeric_cols, categorical_cols = prepare_features(df, TARGET_COL)
    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", categorical_cols)
    # Train-test split (stratify by target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE, stratify=y)
    # Pipeline
    pipe = build_pipeline(numeric_cols, categorical_cols)
    # Cross-validate baseline
    cross_validate(pipe, X_train, y_train)
    # Train baseline and evaluate
    model = evaluate_model(pipe, X_train, X_test, y_train, y_test)
    # Hyperparameter tuning (optional — may take longer)
    best = tune_hyperparameters(pipe, X_train, y_train)
    # Evaluate tuned model
    evaluate_model(best, X_train, X_test, y_train, y_test)
    # Save best model
    save_model(best)

if __name__ == "__main__":
    main()