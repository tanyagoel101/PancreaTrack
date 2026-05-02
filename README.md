# PancreaTrack

PancreaTrack is a Streamlit dashboard for educational early-risk exploration in pancreatic cancer. It combines urinary biomarker data from the Kaggle "Urinary biomarkers for pancreatic cancer" dataset with patient-entered symptom and metabolic context to show interpretable warning signals, simulated longitudinal trends, and comparative model performance.

## Dataset

- Kaggle dataset: [Urinary biomarkers for pancreatic cancer](https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer)
- Default dataset path used by the app:
  - `data/archive/Debernardi et al 2020 data.csv`

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

If the dataset is not found at the default path, the app will offer a CSV upload fallback in the UI.

## Features

- Overview tab with patient counts, PDAC prevalence, and best-model metrics
- Patient dashboard with:
  - PDAC risk score from urinary biomarkers and demographics
  - heuristic overlay for glucose, weight loss, and symptom warnings
  - interpretable feature contribution breakdown
  - plain-language clinical follow-up recommendation
- Biomarker Trends tab with simulated 6-month trajectories for:
  - CA19-9
  - glucose/HbA1c
  - weight
  - LYVE1
  - REG1B
  - TFF1
- Model Performance tab comparing logistic regression and random forest using:
  - accuracy
  - ROC-AUC
  - sensitivity
  - specificity
  - confusion matrix
- Explainability tab using SHAP when available, with fallback feature importance otherwise

## Important Note

This tool is for research and educational purposes only and does not provide medical diagnosis.

## Limitations

- Small dataset
- Simulated trends rather than real longitudinal observations
- Not clinically validated
- Several user-entered clinical fields are heuristic overlays and are not part of the learned Kaggle model

## Future Improvements

- Real longitudinal data
- Integration of additional biomarkers
- Improved ML models
- External validation and calibration analysis
