# cost_engine.py

import pandas as pd
import numpy as np

DEFAULT_COST_FN = 10000  
DEFAULT_COST_FP = 150    

def compute_costs(sweep_df, cost_fn=DEFAULT_COST_FN, cost_fp=DEFAULT_COST_FP):

    
    df = sweep_df.copy()
    
    
    total_patients = df['TP'].iloc[0] + df['FP'].iloc[0] + \
                     df['FN'].iloc[0] + df['TN'].iloc[0]
    
    df['total_harm'] = (df['FN'] * cost_fn) + (df['FP'] * cost_fp)
    
    total_diabetic = df['TP'].iloc[0] + df['FN'].iloc[0]
    
    baseline_harm = total_diabetic * cost_fn
    
    df['net_benefit'] = baseline_harm - df['total_harm']
    
    
    df['lives_saved_per_1000'] = (df['TP'] / total_patients * 1000).round(1)
    
    
    df['overtreated_per_1000'] = (df['FP'] / total_patients * 1000).round(1)
    
    
    df['cost_per_life_saved'] = df.apply(
        lambda row: round((row['FP'] * cost_fp) / row['TP'], 2) 
                    if row['TP'] > 0 else float('inf'),
        axis=1
    )
   
    
    return df




def find_optimal_threshold(cost_df):
   
    
    
    if 'total_harm' not in cost_df.columns:
        print("[ERROR] Run compute_costs() before find_optimal_threshold()")
        return None
    
    
    optimal_idx = cost_df['total_harm'].idxmin()
    optimal_row = cost_df.loc[optimal_idx]
    
    return optimal_row




def generate_recommendation(optimal_row, cost_fn, cost_fp):

    
    threshold    = optimal_row['threshold']
    lives        = optimal_row['lives_saved_per_1000']
    overtreated  = optimal_row['overtreated_per_1000']
    net_benefit  = optimal_row['net_benefit']
    precision    = optimal_row['precision']
    recall       = optimal_row['recall']
    
    recommendation = f"""
── POLICY RECOMMENDATION ─────────────────────────────────────────

  Set detection threshold at: {threshold}

  Per 1,000 patients screened:
     Diabetic cases caught early : {lives} patients
     Unnecessary follow-up tests : {overtreated} healthy patients

  Model performance at this threshold:
    Precision : {precision:.1%}  (of flagged patients, this many are truly diabetic)
    Recall    : {recall:.1%}  (of all diabetic patients, this many are caught)

  Economic impact vs no screening:
    Estimated harm reduction : ${net_benefit:,.0f}
    (based on ${cost_fn:,} per missed case, ${cost_fp:,} per unnecessary test)

  Interpretation:
    For every {overtreated:.0f} healthy people who receive an unnecessary test,
    {lives:.0f} diabetic patients are identified and can begin treatment early.

──────────────────────────────────────────────────────────────────
    """
    
    return recommendation




def run_cost_engine(sweep_df, cost_fn=DEFAULT_COST_FN, cost_fp=DEFAULT_COST_FP):
   
    
    
    if sweep_df is None or len(sweep_df) == 0:
        print("[ERROR] Empty sweep DataFrame passed to cost engine.")
        return None, None, None
    
    cost_df = compute_costs(sweep_df, cost_fn, cost_fp)
    
    
    optimal = find_optimal_threshold(cost_df)
    
    
    recommendation = generate_recommendation(optimal, cost_fn, cost_fp)
    
    print(recommendation)
    
    return cost_df, optimal, recommendation



if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.model import train_model
    from src.threshold_sweep import run_sweep
    
    pipeline, y_prob, y_test = train_model(data_dir='data')
    sweep_df, best, pr, rc   = run_sweep(y_test, y_prob)
    cost_df, optimal, rec    = run_cost_engine(sweep_df)