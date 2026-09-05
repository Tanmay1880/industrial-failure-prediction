# Industrial Failure Prediction

## Project Overview

An experimental study of machine learning models for imbalanced
industrial equipment failure prediction.

The project uses the UCI AI4I 2020 Predictive Maintenance Dataset
to investigate how model choice, decision thresholds, asymmetric
false-positive/false-negative costs, and minimum failure-detection
requirements affect model selection.

## Research Question

How do decision thresholds and asymmetric false-positive/false-negative
costs affect model selection when a minimum failure-detection requirement
is imposed?

## Models

The initial study evaluates:

- Logistic Regression
- Decision Tree
- Random Forest

## Dataset

UCI AI4I 2020 Predictive Maintenance Dataset.

The target is binary:

- `0` — Normal
- `1` — Failure

## Experimental Methodology

The research will use:

- Stratified development/final-test split
- Stratified 5-fold cross-validation on development data
- Leakage-safe preprocessing
- Probability-based predictions
- Decision-threshold analysis
- Minimum-recall constraints
- Asymmetric false-positive/false-negative cost scenarios
- Final evaluation on an untouched test set

## Evaluation

Primary evaluation measures include:

- Recall
- Precision
- F1-score
- False Alarm Rate (FAR)
- False Positives (FP)
- False Negatives (FN)
- Total Cost

ROC-AUC will be used as a supporting metric.

Accuracy is not treated as the primary metric because of class
imbalance.

## Project Structure

```text
ml/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   ├── experiments/
│   └── utils/
├── experiments/
├── results/
├── figures/
├── models/
├── tests/
└── notes/