# model.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score
)

FEATURES = [
    'RIDAGEYR',
    'RIAGENDR',
    'RIDRETH3',
    'INDFMPIR',
    'BMXBMI',
    'BPXSY1',
    'BPXDI1',
    'LBXSGL',
    'LBXGH',
    'LBXTC',
]

TARGET = 'diabetes'

MODEL_PATH = 'data/model.joblib'

RANDOM_STATE = 42


def load_clean_data(data_dir='data'):
   
    
    filepath = os.path.join(data_dir, 'nhanes_cleaned.csv')
    

    if not os.path.exists(filepath):
        print("[ERROR] nhanes_cleaned.csv not found.")
        print("        Run data_loader.py first to generate it.")
        return None, None
    
    df = pd.read_csv(filepath)
    print(f"[OK] Loaded cleaned data — shape: {df.shape}")
    
    
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        print(f"[ERROR] These expected features are missing: {missing_features}")
        print("        Re-run data_loader.py or check your column names.")
        return None, None
    
    
    if TARGET not in df.columns:
        print(f"[ERROR] Target column '{TARGET}' not found in dataset.")
        return None, None
    
    
    X = df[FEATURES]
    y = df[TARGET]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape:   {y.shape}")
    print(f"Class balance:  {y.value_counts().to_dict()}")
    
    return X, y



def split_data(X, y):
   
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       
        random_state=RANDOM_STATE,
        stratify=y            
    )
    
    print(f"\n── Train/Test Split ──────────────────────────────")
    print(f"Training set:   {X_train.shape[0]} patients")
    print(f"Test set:       {X_test.shape[0]} patients")
    print(f"Train diabetic: {y_train.sum()} ({100*y_train.mean():.1f}%)")
    print(f"Test diabetic:  {y_test.sum()} ({100*y_test.mean():.1f}%)")
   
    
    return X_train, X_test, y_train, y_test



def build_pipeline():
   
    
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   LogisticRegression(
                        class_weight='balanced',
                        max_iter=1000,
                        random_state=RANDOM_STATE
                    ))
    ])
    
    return pipeline


def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
 
    
    print(f"\n── Training Model ────────────────────────────────")
    
    
    pipeline.fit(X_train, y_train)
    print("[OK] Model trained successfully.")
    
   
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
   
    y_pred = pipeline.predict(X_test)
    
   
    print(f"\n── Evaluation at Default Threshold (0.5) ────────")
    print(classification_report(y_test, y_pred, 
                                 target_names=['Not Diabetic', 'Diabetic']))
   
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr  = average_precision_score(y_test, y_prob)
    
    print(f"AUC-ROC score:  {auc_roc:.4f}")
    print(f"AUC-PR score:   {auc_pr:.4f}")
    print()
    print("Note: For imbalanced medical data, AUC-PR is more honest.")
    print("A high AUC-ROC can be misleading when classes are unequal.")
    print("Watch AUC-PR — that is the real signal.")
    
    return y_prob, y_test



def save_model(pipeline, path=MODEL_PATH):
 
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[SAVED] Pipeline saved to: {path}")




def train_model(data_dir='data'):
   
    
  
    X, y = load_clean_data(data_dir)
    if X is None:
        return None, None, None
    
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
   
    pipeline = build_pipeline()
    
   
    y_prob, y_test = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)
    
   
    save_model(pipeline)
    
    return pipeline, y_prob, y_test


if __name__ == '__main__':
    pipeline, y_prob, y_test = train_model(data_dir='data')
    
    if y_prob is not None:
        print(f"\nFirst 10 probability scores:")
        print(y_prob[:10].round(4))
        print("\nThese are the scores threshold_sweep.py will work with next.")
