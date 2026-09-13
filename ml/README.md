# Industrial Failure Prediction

## Overview

Machine learning project for industrial equipment failure prediction using the UCI AI4I 2020 Predictive Maintenance Dataset.

The project includes:
- Dataset analysis
- Experimental evaluation of three ML models
- Threshold analysis
- Minimum-recall analysis
- Cost-sensitive analysis
- Final model selection
- Final test evaluation
- Trained ML prediction service
- FastAPI API
- Automated tests

## Models

- Logistic Regression
- Decision Tree
- Random Forest

All three models are used for every prediction request.

The current research-selected operating configuration is:

- Model: Random Forest
- Threshold: 0.20
- Minimum recall: 80%
- Cost scenario: C3
- FP cost: 1
- FN cost: 5

## Dataset

Dataset:
UCI AI4I 2020 Predictive Maintenance Dataset

Target:
- 0 = Normal
- 1 = Failure

Dataset size:
- 10,000 records
- 9,661 normal
- 339 failures

## Project Structure

industrial-failure-prediction/
|
├── ml/
│   ├── data/
│   │   └── raw/
│   │       └── ai4i2020.csv
│   │
│   ├── src/
│   │   ├── api/
│   │   │   ├── main.py
│   │   │   ├── routes.py
│   │   │   ├── schemas.py
│   │   │   └── mappers.py
│   │   │
│   │   ├── data/
│   │   │   └── load_dataset.py
│   │   │
│   │   ├── models/
│   │   │   └── model_factory.py
│   │   │
│   │   ├── prediction/
│   │   │   ├── predictor.py
│   │   │   └── schemas.py
│   │   │
│   │   ├── preprocessing/
│   │   │   └── pipeline.py
│   │   │
│   │   ├── training/
│   │   │   └── train_models.py
│   │   │
│   │   └── utils/
│   │       └── config.py
│   │
│   ├── experiments/
│   │   ├── experiment_00_dataset/
│   │   ├── experiment_01_baseline/
│   │   ├── experiment_02_probability/
│   │   ├── experiment_03_threshold/
│   │   ├── experiment_04_recall_constraint/
│   │   ├── experiment_05_cost/
│   │   ├── experiment_06_model_selection/
│   │   └── experiment_07_final_test/
│   │
│   ├── results/
│   ├── figures/
│   ├── models/
│   ├── tests/
│   │   ├── test_api.py
│   │   └── test_prediction.py
│   ├── notes/
│   └── requirements.txt
│
└── README.md

## Setup

### Requirements

- Python 3.14
- pip

### Clone the repository

git clone <https://github.com/Tanmay1880/industrial-failure-prediction.git>
cd industrial-failure-prediction
cd ml

### Create virtual environment

Windows PowerShell:

python -m venv .venv

Activate:

.\.venv\Scripts\Activate.ps1

### Install dependencies

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

## Train Models

The trained model files are generated locally and are not stored in Git.

From the ml/ directory:

python -m src.training.train_models

This creates:

models/
- logistic_regression.joblib
- decision_tree.joblib
- random_forest.joblib

A new user can recreate all required model files by running the training command above.

## Run the API

From the ml/ directory:

python -m uvicorn src.api.main:app --reload

API:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

Health check:

GET /health

## API Endpoints

### User Prediction

POST /predict

Returns the final user-facing prediction.

### Admin Prediction

POST /admin/predict

Returns:
- Final status
- Failure probability
- Logistic Regression result
- Decision Tree result
- Random Forest result
- Selected model
- Selected threshold

## Prediction Input

Example:

{
  "type": "M",
  "air_temperature": 300.0,
  "process_temperature": 310.0,
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 100.0
}

## Run Tests

From the ml/ directory:

python -m pytest -v

The test suite covers the prediction engine and API.

## Research Experiments

The experiments are organized as:

- Experiment 00: Dataset Analysis
- Experiment 01: Baseline Evaluation
- Experiment 02: Out-of-Fold Probability Generation
- Experiment 03: Threshold Analysis
- Experiment 04: Minimum Recall Constraint
- Experiment 05: Cost-Sensitive Analysis
- Experiment 06: Final Model Selection
- Experiment 07: Final Test Evaluation

Results are stored under:

ml/results/

Research notes are stored under:

ml/notes/

The consolidated results are stored in:

ml/results/master_results.csv

## Current Final Test Result

Final selected configuration:

- Model: Random Forest
- Threshold: 0.20

Final test:

- Test rows: 2,000
- Actual failures: 68
- True positives: 56
- False negatives: 12
- False positives: 43
- True negatives: 1,889
- Recall: 82.35%
- Precision: 56.57%
- F1: 0.6707
- False alarm rate: 2.23%
- ROC-AUC: 0.9653
- Total cost: 103

## Development Status

ML research experiments: Complete

ML training pipeline: Complete

Prediction engine: Complete

FastAPI service: Complete

Automated tests: Complete

Final test evaluation: Complete

Application integration with the main backend/frontend: Pending