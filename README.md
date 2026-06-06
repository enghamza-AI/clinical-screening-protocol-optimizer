# 🩺 Clinical Screening Protocol Optimizer

> *Don't just measure your model — measure the lives it saves and the harm it causes.*

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-kairon-yellow)](https://huggingface.co/spaces/enghamza-AI/kairon)
[![GitHub](https://img.shields.io/badge/GitHub-enghamza--AI-black?logo=github)](https://github.com/enghamza-AI/clinical-screening-protocol-optimizer)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Dataset](https://img.shields.io/badge/Data-CDC%20NHANES-green)](https://wwwn.cdc.gov/nchs/nhanes/)

---

## What Is This?

Most ML engineers optimize F1 score. This tool goes further.

Given a trained diabetes classifier on **real CDC health data (10,000+ patients)**, it answers the question that actually matters to a health official:

> *"If I set the detection threshold at X — how many additional lives do I save, and how many people do I over-treat per 1000 patients?"*

This bridges machine learning and health economics. It converts abstract metrics (precision, recall, F1) into concrete policy decisions — the way epidemiologists and public health researchers actually think.

---

## The Problem

A model that gets **95% accuracy** on diabetes prediction can still be dangerously useless.

If only 8% of patients in the dataset have diabetes, a model that predicts "no diabetes" for **everyone** gets 92% accuracy — and misses every single sick person.

Accuracy is a lie in imbalanced medical data.

This tool replaces accuracy with a **cost-aware threshold optimizer** that speaks the language of healthcare: lives, tests, and tradeoffs.

---

## Live Demo

🚀 **[Try it on HuggingFace Spaces → kairon](https://huggingface.co/spaces/enghamza-AI/kairon)**

Upload predictions + ground truth → adjust cost sliders → get a policy recommendation instantly.

---

## What It Does

```
Input: CDC NHANES patient data (age, BMI, blood glucose, HbA1c, blood pressure...)
         ↓
Train logistic regression classifier
         ↓
Sweep all thresholds from 0.01 → 0.99
         ↓
At each threshold, compute:
  • Precision, Recall, F1
  • Lives saved vs baseline (doing nothing)
  • Over-treatment cost (unnecessary tests per 1000 patients)
  • Net harm score (user-defined cost weights)
         ↓
Output: "Set threshold at 0.38 — saves 47 additional lives
         at cost of 312 extra tests per 1000 patients."
```

---

## Key Concepts Demonstrated

| Concept | What It Means Here |
|---|---|
| **Precision** | Of everyone flagged as diabetic — how many actually are? |
| **Recall** | Of all truly diabetic patients — how many did we catch? |
| **Threshold tuning** | Where you draw the line between "sick" and "healthy" |
| **PR Curve** | Visual tradeoff across every possible threshold |
| **Cost matrix** | Assigning real-world harm values to false negatives and false positives |
| **Pareto tradeoff** | No perfect answer — every gain in recall costs precision |

---

## Dataset

**NHANES — National Health and Nutrition Examination Survey**  
Source: CDC (Centers for Disease Control and Prevention)  
Patients: 10,000+ real Americans  
Features used: age, BMI, blood glucose, HbA1c, blood pressure, cholesterol, income ratio, gender  
Target: `DIQ010` — doctor-confirmed diabetes diagnosis (binary: yes/no)

This is not a toy dataset. NHANES is used by epidemiologists and health researchers globally.

---

## Project Structure

```
clinical-screening-protocol-optimizer/
│
├── data/
│   ├── demographic.csv
│   ├── examination.csv
│   ├── labs.csv
│   ├── questionnaire.csv
│   └── nhanes_cleaned.csv        ← auto-generated after first run
│
├── src/
│   ├── data_loader.py            ← loads, merges, cleans NHANES files
│   ├── model.py                  ← trains classifier, outputs probabilities
│   ├── threshold_sweep.py        ← core engine: sweeps all thresholds
│   ├── cost_engine.py            ← converts metrics → lives/costs
│   └── visualizer.py             ← PR curve, cost curves, plots
│
├── app.py                        ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/enghamza-AI/clinical-screening-protocol-optimizer
cd clinical-screening-protocol-optimizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your NHANES CSVs to the data/ folder

# 4. Load and clean the data
python src/data_loader.py

# 5. Run the Streamlit app
streamlit run app.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
streamlit
matplotlib
scipy
```

---

## How The Cost Engine Works

The tool lets you define two costs:

- `cost_per_FN` — the harm of missing a sick person (they go untreated)
- `cost_per_FP` — the harm of a false alarm (unnecessary test, anxiety, cost)

At each threshold:

```python
total_harm = (false_negatives × cost_per_FN) + (false_positives × cost_per_FP)
```

The optimal threshold is the one that minimizes total harm given **your** cost weights.

A hospital in a low-resource setting weights FP cost high (can't afford unnecessary tests).  
A cancer screening program weights FN cost high (missing a case is catastrophic).  
Your tool handles both — just move the sliders.

---

## Results

On the NHANES dataset:

- Default threshold (0.5): catches ~71% of diabetic patients
- Cost-optimized threshold: catches ~84% of diabetic patients at 2.1× test volume
- **Net effect at population scale (1000 patients): ~13 additional cases caught**

---

## Part of the Diamond AI Roadmap

This project is **Stage 2, Week 2** of an 11-stage self-directed AI systems engineering curriculum.

**Stage 2 theme:** Decision Intelligence & Metric Engineering  
**Goal:** Learn that ML metrics only matter when attached to real-world decisions.

Previous work → [github.com/enghamza-AI](https://github.com/enghamza-AI)  
Full portfolio → [huggingface.co/spaces/enghamza-AI](https://huggingface.co/spaces/enghamza-AI)

---

## Author

**Hamza** — BSAI student, self-studying AI systems engineering.  
Building toward roles at Anthropic, xAI, OpenAI, Perplexity, and YC-backed startups.

GitHub: [@enghamza-AI](https://github.com/enghamza-AI)  
HuggingFace: [@enghamza-AI](https://huggingface.co/spaces/enghamza-AI)

---

*Build rare. Ship often. Let the portfolio speak.*
