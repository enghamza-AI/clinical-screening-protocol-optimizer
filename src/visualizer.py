
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


plt.rcParams.update({
    'figure.facecolor': '#0e1117',   
    'axes.facecolor':   '#0e1117',
    'axes.edgecolor':   '#444444',
    'axes.labelcolor':  '#ffffff',
    'text.color':       '#ffffff',
    'xtick.color':      '#aaaaaa',
    'ytick.color':      '#aaaaaa',
    'grid.color':       '#333333',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
})




def plot_pr_curve(precision_curve, recall_curve, optimal_threshold_row=None):

    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    
    ax.plot(recall_curve, precision_curve, 
            color='#00d4aa', linewidth=2.5, label='PR Curve')
    
   
    baseline = precision_curve.min()
    ax.axhline(y=baseline, color='#ff6b6b', linestyle='--', 
               alpha=0.7, label=f'Random baseline ({baseline:.2f})')
    
   
    if optimal_threshold_row is not None:
        opt_recall    = optimal_threshold_row['recall']
        opt_precision = optimal_threshold_row['precision']
        ax.scatter(opt_recall, opt_precision, 
                   color='#ffd700', s=150, zorder=5,
                   label=f"Optimal (t={optimal_threshold_row['threshold']})")
        
    
    ax.set_xlabel('Recall (Sensitivity) — fraction of sick patients caught', 
                  fontsize=11)
    ax.set_ylabel('Precision — fraction of flagged patients truly sick', 
                  fontsize=11)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig




def plot_threshold_sweep(sweep_df, optimal_threshold=None):
   
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sweep_df['threshold'], sweep_df['precision'], 
            color='#4fc3f7', linewidth=2, label='Precision')
    ax.plot(sweep_df['threshold'], sweep_df['recall'],    
            color='#ff8a65', linewidth=2, label='Recall')
    ax.plot(sweep_df['threshold'], sweep_df['f1'],        
            color='#a5d6a7', linewidth=2, label='F1 Score', linestyle='--')
    
  
    if optimal_threshold is not None:
        ax.axvline(x=optimal_threshold, color='#ffd700', 
                   linestyle=':', linewidth=2,
                   label=f'Optimal threshold ({optimal_threshold})')
    
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Precision, Recall & F1 vs Threshold', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig




def plot_cost_curve(cost_df, optimal_threshold=None):
   
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(cost_df['threshold'], cost_df['total_harm'],
            color='#ef5350', linewidth=2.5, label='Total Harm ($)')
    
    ax.fill_between(cost_df['threshold'], cost_df['total_harm'],
                    alpha=0.15, color='#ef5350')
    
    if optimal_threshold is not None:
        min_harm = cost_df[cost_df['threshold'] == optimal_threshold]['total_harm'].values
        if len(min_harm) > 0:
            ax.scatter(optimal_threshold, min_harm[0],
                      color='#ffd700', s=200, zorder=5,
                      label=f'Minimum harm (t={optimal_threshold})')
            ax.axvline(x=optimal_threshold, color='#ffd700',
                      linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Total Harm ($)', fontsize=11)
    ax.set_title('Total Economic Harm vs Threshold', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True)
    
   
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
    )
    
    plt.tight_layout()
    return fig




def plot_lives_vs_overtreated(cost_df, optimal_threshold=None):
   
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    
    scatter = ax.scatter(
        cost_df['overtreated_per_1000'],
        cost_df['lives_saved_per_1000'],
        c=cost_df['threshold'],
        cmap='RdYlGn_r',   
        s=40, alpha=0.7
    )
    
    plt.colorbar(scatter, ax=ax, label='Threshold value')
    
   
    if optimal_threshold is not None:
        opt_row = cost_df[cost_df['threshold'] == optimal_threshold]
        if len(opt_row) > 0:
            ax.scatter(
                opt_row['overtreated_per_1000'],
                opt_row['lives_saved_per_1000'],
                color='#ffd700', s=300, zorder=5,
                marker='*', label=f'Optimal (t={optimal_threshold})'
            )
    
    ax.set_xlabel('Overtreated per 1,000 patients (healthy people flagged)', 
                  fontsize=11)
    ax.set_ylabel('Lives saved per 1,000 patients (diabetic cases caught)', 
                  fontsize=11)
    ax.set_title('Lives Saved vs Over-Treatment Tradeoff', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig