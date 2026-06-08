# data_loader.py

import pandas as pd
import numpy as np
import os
import warnings



warnings.filterwarnings('ignore')

DEMOGRAPHIC_COLS = [
    'SEQN',
    'RIDAGEYR',
    'RIAGENDR',
    'RIDRETH3',
    'INDFMPIR',
]

EXAMINATON_COLS = [
    'SEQN',
    'BMXBMI',
    'BPXSY1',
    'BPXDI1',
]

LAB_COLS = [
    'SEQN',
    'LBXGLU',
    'LBXGH',
    'LBXTC',
]

QUESTIONNAIRE_COLS = [
    'SEQN',
    'DIQ010',
]

def load_csv_safe(filepath, columns):

    if not os.path.exists(filepath):
        print(f"[WARNING] file not found: {filepath}")
        return None 
    
    df = pd.read_csv(filepath, encoding='latin-1')

    if df.empty:
        print(f"[WARNING] File is empty: {filepath}")
        return None
    
    print(f"[OK] loaded {filepath} shape: {df.shape}")

    available_cols = [col for col in columns if col in df.columns]

    missing_cols = [col for col in columns if col not in df.columns]

    if missing_cols:
        print(f"[WARNING] these columns not found in {filepath}: {missing_cols}")
        

    if 'SEQN' not in available_cols:
        print(f"[ERROR] SEQN (patient ID) missing from {filepath}. cannot use this file")
        return None
    
    return df[available_cols]

def merge_nhanes_files(data_dir):

    print("\n-- loading NHANES files --")

    df_demo = load_csv_safe(os.path.join(data_dir, 'demographic.csv'), DEMOGRAPHIC_COLS)
    df_exam = load_csv_safe(os.path.join(data_dir, 'examination.csv'), EXAMINATON_COLS)
    df_labs = load_csv_safe(os.path.join(data_dir, 'labs.csv'),        LAB_COLS)
    df_quest = load_csv_safe(os.path.join(data_dir, 'questionnaire.csv'), QUESTIONNAIRE_COLS)

    files = {'demographics': df_demo, 'examination': df_exam,
             'labs': df_labs, 'questionaire': df_quest}
    
    failed = [name for name, df in files.items() if df is None]
    if failed:
        print(f"\n[ERROR] These files failed to load: {failed}")
        print("          CANNOT CONTINUE WITHOUT ALL 4 FILS.")
        return None

    print("\n── Merging Files on SEQN ──────")
    
    
    merged = df_demo
    print(f"Base (demographic):           {merged.shape}")
    
    
    merged = merged.merge(df_exam, on='SEQN', how='left')
    print(f"After adding examination:     {merged.shape}")
    
    
    merged = merged.merge(df_labs, on='SEQN', how='left')
    print(f"After adding labs:            {merged.shape}")
    
    
    merged = merged.merge(df_quest, on='SEQN', how='left')
    print(f"After adding questionnaire:   {merged.shape}")
    
    return merged

def create_target_and_clean(df):

    if 'DIQ010' not in df.columns:
        print("[ERROR] target column DIQ010 not found.")
        return None
    
    print(f"Raw DIQ010 value counts:\n{df['DIQ010'].value_counts()}\n")

    target_map = {1: 1, 2: 0}
    df['diabetes'] = df['DIQ010'].map(target_map)


    df = df.drop(columns=['DIQ010'])


    before = len(df)
    df = df.dropna(subset=['diabetes'])
    after = len(df)
    print(f"Raw dropped due to missing target: {before - after}")
    print(f"Rows remaining: {after}")

    if 'RIDAGEYR' in df.columns:
        before = len(df)
        df = df[df['RIDAGEYR'] >= 18]
        print(f"Rows removed (under 18): {before - len(df)}")


    diabetes_counts = df['diabetes'].value_counts()
    total = len(df)
    print(f"\nClass distribution:")
    print(f"  Not diabetic (0): {diabetes_counts.get(0,0)} ({100*diabetes_counts.get(0,0)/total:.1f}%)")
    print(f"  Diabetic     (1): {diabetes_counts.get(1,0)} ({100*diabetes_counts.get(1,0)/total:.1f}%)")
    print(f"\n  → Imbalance ratio: {diabetes_counts.get(0,0)/max(diabetes_counts.get(1,1),1):.1f}:1")
    # This ratio tells you how skewed the classes are
    # If it's 10:1 — accuracy is a useless metric (the hospital story from earlier)
    
    return df


def run_eda(df):
  
    
    print("\n── EDA Report ───────")
    
    
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    

    print("\nMissing values per column:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_report = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    })
    
    missing_report = missing_report[missing_report['missing_count'] > 0]
    
    if missing_report.empty:
        print("  No missing values found.")
    else:
        print(missing_report.sort_values('missing_pct', ascending=False))
    
    
    print("\nBasic statistics for numeric columns:")
    print(df.describe().round(2))
   
    print("\nSanity checks (physically impossible values):")
    
    checks = {
        'RIDAGEYR': (0, 120),    
        'BMXBMI':   (10, 80),    
        'LBXGLU':   (20, 600),   
        'LBXGH':    (2, 20),     
        'BPXSY1':   (60, 250),   
    }
    
    for col, (low, high) in checks.items():
        if col in df.columns:
            # Count values outside the realistic range
            outliers = df[(df[col] < low) | (df[col] > high)][col].count()
            if outliers > 0:
                print(f"  [!] {col}: {outliers} values outside [{low}, {high}]")
            else:
                print(f"  [OK] {col}: all values within realistic range")
    
    return df

def load_nhanes(data_dir='data'):



    df = merge_nhanes_files(data_dir)
    if df is None:
        return None
    
    df = run_eda(df)

    df = create_target_and_clean(df)
    if df is None:
        return None
    
    output_path = os.path.join(data_dir, 'nhanes_cleaned.csv')
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] clean dataset saved to: {output_path}")
    print(f"Final dataset shape: {df.shape}")

    return df


if __name__ == '__main__':
    df = load_nhanes(data_dir='data')
    if df is not None:
        print("\nFirst 5 rows:")
        print(df.head())
