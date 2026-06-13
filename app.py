
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model           import train_model
from src.threshold_sweep import run_sweep
from src.cost_engine     import run_cost_engine, DEFAULT_COST_FN, DEFAULT_COST_FP
from src.visualizer      import (plot_pr_curve, plot_threshold_sweep,
                                  plot_cost_curve, plot_lives_vs_overtreated)


st.set_page_config(
    page_title = "Clinical Screening Protocol Optimizer",
    page_icon  = "🩺",
    layout     = "wide"
)


st.title("🩺 Clinical Screening Protocol Optimizer")
st.markdown("""
> *Don't just measure your model — measure the lives it saves and the harm it causes.*

Built on real CDC NHANES data (10,000+ patients). 
This tool converts ML metrics into actionable health policy recommendations.
""")

st.divider()



st.sidebar.header(" Cost Parameters")
st.sidebar.markdown("""
Adjust the economic cost of each type of error.
These values drive the optimal threshold recommendation.
""")

cost_fn = st.sidebar.slider(
    label   = "Cost of missing one diabetic patient ($)",
    min_value = 1000,
    max_value = 50000,
    value   = DEFAULT_COST_FN,
    step    = 500,
    help    = "Untreated diabetes leads to expensive complications. "
              "Conservative estimate: $10,000 per missed case."
)

cost_fp = st.sidebar.slider(
    label   = "Cost of one unnecessary test ($)",
    min_value = 50,
    max_value = 1000,
    value   = DEFAULT_COST_FP,
    step    = 25,
    help    = "Cost of follow-up tests for a healthy patient who was "
              "incorrectly flagged as diabetic."
)

st.sidebar.divider()
st.sidebar.markdown(f"""
**Current cost ratio:** `{cost_fn // cost_fp}:1`  
Missing one patient costs **{cost_fn // cost_fp}×** more than one unnecessary test.
""")


@st.cache_resource
def load_model_and_sweep():
   
    pipeline, y_prob, y_test = train_model(data_dir='data')
    
    if pipeline is None:
        return None
    
    sweep_df, best, precision_curve, recall_curve = run_sweep(y_test, y_prob)
    
    return {
        'pipeline':       pipeline,
        'y_prob':         y_prob,
        'y_test':         y_test,
        'sweep_df':       sweep_df,
        'best':           best,
        'precision_curve': precision_curve,
        'recall_curve':   recall_curve,
    }



with st.spinner("Training model on NHANES data... (first load only)"):
    cached = load_model_and_sweep()


if cached is None:
    st.error("""
     Could not load data. Make sure:
    1. Your CSV files are in the `data/` folder
    2. You have run `data_loader.py` first
    3. `nhanes_cleaned.csv` exists in `data/`
    """)
    st.stop()
  
cost_df, optimal, recommendation = run_cost_engine(
    cached['sweep_df'], cost_fn=cost_fn, cost_fp=cost_fp
)

optimal_threshold = float(optimal['threshold'])


st.subheader("📋 Policy Recommendation")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label = "Optimal Threshold",
        value = f"{optimal_threshold:.2f}",
        help  = "Set your model's decision boundary here"
    )

with col2:
    st.metric(
        label = "Lives Saved per 1,000",
        value = f"{optimal['lives_saved_per_1000']:.0f}",
        help  = "Diabetic patients identified early per 1,000 screened"
    )

with col3:
    st.metric(
        label = "Unnecessary Tests per 1,000",
        value = f"{optimal['overtreated_per_1000']:.0f}",
        help  = "Healthy patients flagged for unnecessary follow-up"
    )

st.code(recommendation, language=None)

st.divider()


st.subheader("Analysis Charts")

# Row 1: PR Curve + Threshold Sweep
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Precision-Recall Curve")
    st.markdown("""
    Each point = one possible threshold.  
    Top-right corner = perfect model.  
     Gold star = cost-optimal threshold.
    """)
    fig_pr = plot_pr_curve(
        cached['precision_curve'],
        cached['recall_curve'],
        optimal
    )
    st.pyplot(fig_pr)

with col_right:
    st.markdown("#### Precision, Recall & F1 vs Threshold")
    st.markdown("""
    Move the threshold → watch how each metric changes.  
    The crossing point = maximum F1 score.  
    Vertical gold line = cost-optimal threshold.
    """)
    fig_sweep = plot_threshold_sweep(cached['sweep_df'], optimal_threshold)
    st.pyplot(fig_sweep)

st.divider()


col_left2, col_right2 = st.columns(2)

with col_left2:
    st.markdown("#### Total Economic Harm vs Threshold")
    st.markdown("""
    Lower = better.  
    The minimum point is the optimal threshold.  
    Left side = over-treating healthy people.  
    Right side = missing sick people.
    """)
    fig_cost = plot_cost_curve(cost_df, optimal_threshold)
    st.pyplot(fig_cost)

with col_right2:
    st.markdown("#### Lives Saved vs Over-Treatment Tradeoff")
    st.markdown("""
    Each dot = one threshold.  
    Up-left = ideal (more lives saved, fewer overtreated).  
     Gold star = cost-optimal threshold.
    """)
    fig_lives = plot_lives_vs_overtreated(cost_df, optimal_threshold)
    st.pyplot(fig_lives)

st.divider()


st.subheader(" Full Threshold Sweep Data")
st.markdown("Every threshold from 0.01 to 0.99 with all computed metrics.")


def highlight_optimal(row):
    
    if abs(row['threshold'] - optimal_threshold) < 0.001:
        return ['background-color: #3d3200'] * len(row)
    return [''] * len(row)

display_df = cost_df[[
    'threshold', 'precision', 'recall', 'f1',
    'lives_saved_per_1000', 'overtreated_per_1000',
    'total_harm', 'net_benefit'
]].copy()

st.dataframe(
    display_df.style.apply(highlight_optimal, axis=1),
    use_container_width=True,
    height=400
)


st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
Built by <a href='https://github.com/enghamza-AI' style='color:#00d4aa'>Hamza</a> · 
Stage 2 Week 2 · Diamond AI Roadmap · 
<a href='https://huggingface.co/spaces/enghamza-AI/kairon' style='color:#00d4aa'>Live Demo</a>
</div>
""", unsafe_allow_html=True)