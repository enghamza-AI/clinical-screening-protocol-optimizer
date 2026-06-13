# threshold_sweep.py

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import confusion_matrix

THRESHOLDS = np.arange(0.01, 1.0, 0.01)



def sweep_thresholds(y_true, y_prob):

    results = []
    
    for threshold in THRESHOLDS:
        
        
        y_pred = (y_prob >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
      
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        
        f1 = (2 * precision * recall / (precision + recall) 
              if (precision + recall) > 0 else 0.0)
        
        total = len(y_true)
        
        
        flagged_per_1000 = (tp + fp) / total * 1000
        
        
        missed_per_1000 = fn / total * 1000
        
        
        results.append({
            'threshold':        round(threshold, 2),
            'TP':               int(tp),
            'FP':               int(fp),
            'FN':               int(fn),
            'TN':               int(tn),
            'precision':        round(precision, 4),
            'recall':           round(recall, 4),
            'f1':               round(f1, 4),
            'flagged_per_1000': round(flagged_per_1000, 1),
            'missed_per_1000':  round(missed_per_1000, 1),
        })
    
    
    results_df = pd.DataFrame(results)
    
    print(f"[OK] Threshold sweep complete — {len(results_df)} thresholds evaluated.")
    
    return results_df



def find_best_thresholds(sweep_df):

    
    
    best_f1_idx = sweep_df['f1'].idxmax()
    best_f1     = sweep_df.loc[best_f1_idx]
    
    
    high_recall = sweep_df[sweep_df['recall'] >= 0.90]
    if len(high_recall) > 0:
        best_recall_idx = high_recall['precision'].idxmax()
        best_recall     = sweep_df.loc[best_recall_idx]
    else:
        
        best_recall = sweep_df.loc[sweep_df['recall'].idxmax()]
    

    high_precision = sweep_df[sweep_df['precision'] >= 0.90]
    if len(high_precision) > 0:
        best_prec_idx = high_precision['recall'].idxmax()
        best_precision = sweep_df.loc[best_prec_idx]
    else:
        best_precision = sweep_df.loc[sweep_df['precision'].idxmax()]
    
    print(f"\n── Best Thresholds Summary ──")
    print(f"\nBest F1 threshold:        {best_f1['threshold']}")
    print(f"  Precision: {best_f1['precision']} | Recall: {best_f1['recall']} | F1: {best_f1['f1']}")
    
    print(f"\nBest Recall threshold:    {best_recall['threshold']}")
    print(f"  Precision: {best_recall['precision']} | Recall: {best_recall['recall']} | F1: {best_recall['f1']}")
    
    print(f"\nBest Precision threshold: {best_precision['threshold']}")
    print(f"  Precision: {best_precision['precision']} | Recall: {best_precision['recall']} | F1: {best_precision['f1']}")
    
    return {
        'best_f1':        best_f1,
        'best_recall':    best_recall,
        'best_precision': best_precision
    }

# ── PR CURVE DATA ─────────────────────────────────────────────────────────────

def get_pr_curve_data(y_true, y_prob):
    """
    Uses sklearn's built-in precision_recall_curve to get the
    data points for plotting the PR curve.
    
    Why use sklearn here instead of our sweep DataFrame?
    sklearn's precision_recall_curve computes at every unique probability
    value in y_prob — that can be hundreds of points, smoother than
    our 99-step sweep. Better looking curve.
    
    Returns precision array, recall array for plotting in visualizer.py
    """
    
    precision_curve, recall_curve, curve_thresholds = precision_recall_curve(
        y_true, y_prob
    )
    
    # precision_recall_curve returns arrays where:
    # precision_curve[i] = precision at curve_thresholds[i]
    # recall_curve[i]    = recall at curve_thresholds[i]
    # Note: len(precision_curve) = len(curve_thresholds) + 1
    # sklearn adds a final point at (recall=0, precision=1) by convention
    
    return precision_curve, recall_curve, curve_thresholds


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_sweep(y_true, y_prob):
    """
    Master function — call this from app.py or directly.
    Takes true labels and predicted probabilities.
    Returns sweep DataFrame and best threshold recommendations.
    """
    
    # ── DEFENSIVE CHECK ──
    if len(y_true) != len(y_prob):
        print("[ERROR] y_true and y_prob must have the same length.")
        return None, None
    
    if len(y_true) == 0:
        print("[ERROR] Empty arrays passed to sweep.")
        return None, None
    
    # Run the sweep
    sweep_df = sweep_thresholds(y_true, y_prob)
    
    # Find best thresholds
    best = find_best_thresholds(sweep_df)
    
    # Get PR curve data
    precision_curve, recall_curve, _ = get_pr_curve_data(y_true, y_prob)
    
    return sweep_df, best, precision_curve, recall_curve



if __name__ == '__main__':
   
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.model import train_model
    
    print("Training model to get probabilities for sweep test...")
    pipeline, y_prob, y_test = train_model(data_dir='data')
    
    if y_prob is not None:
        sweep_df, best, pr, rc = run_sweep(y_test, y_prob)
        
        print(f"\nFirst 10 rows of sweep DataFrame:")
        print(sweep_df.head(10))
        
        print(f"\nLast 10 rows of sweep DataFrame:")
        print(sweep_df.tail(10))
        
     