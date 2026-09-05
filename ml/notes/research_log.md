# Research Log

## Experiment 0 — Dataset Investigation

**Date:** 2026-09-03

### WHAT WE DID

Loaded and investigated the UCI AI4I 2020 Predictive Maintenance Dataset
from `data/raw/ai4i2020.csv`.

We examined:

- dataset dimensions
- column names and data types
- missing values
- duplicate rows
- target distribution
- failure-mode indicators
- identifier columns
- categorical variables
- numerical variables
- numerical feature means by target
- consistency between failure-mode indicators and the machine-failure target

The investigation was implemented in:

`experiments/experiment_00_dataset/run_experiment.py`

### WHY

The purpose of Experiment 0 was to understand the dataset before
training any machine-learning models and to identify data-quality,
feature-selection, and class-imbalance considerations that affect the
experimental methodology.

### RESULT

The dataset contains:

- 10,000 observations
- 14 columns
- 9,661 normal observations
- 339 failure observations
- 96.61% normal observations
- 3.39% failure observations
- 0 missing values
- 0 duplicate rows

The dataset contains numerical and categorical variables.

`UDI` and `Product ID` were identified as identifier columns.

`Type` was identified as a categorical variable.

The physical/process measurements are:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

The dataset also contains five failure-mode indicators:

- TWF
- HDF
- PWF
- OSF
- RNF

### IMPORTANT NUMBERS

- Total observations: 10,000
- Normal: 9,661 (96.61%)
- Failure: 339 (3.39%)
- Missing values: 0
- Duplicate rows: 0
- Failure-mode disagreement rows: 27

### OBSERVATION

The target is severely imbalanced, with machine failures representing
only 3.39% of observations.

Several numerical variables show different average values between
normal and failure observations, particularly torque, tool wear, and
rotational speed.

These differences indicate that the variables may contain predictive
information, but they do not establish causal relationships.

The failure-mode indicators are strongly related to the machine-failure
target but are not exactly equivalent to it. Their use as predictors
therefore requires careful consideration of prediction-time information
and target leakage risk.

### DECISION

Accuracy will not be treated as the primary evaluation metric.

The experimental evaluation will emphasize:

- Recall
- Precision
- F1
- False positives
- False negatives
- False alarm rate
- Total cost

ROC-AUC will be used as a supporting metric.

`UDI` and `Product ID` will not be treated as ordinary predictive
measurements.

The failure-mode indicators will be considered separately from the
ordinary machine/process features when defining the modeling feature
set.

### WHY

The severe class imbalance makes accuracy potentially misleading.
In predictive maintenance, missed failures can be operationally
important, which motivates explicit analysis of recall, false negatives,
false positives, and asymmetric costs.

Identifier columns do not represent machine behavior and should not be
used as ordinary predictive measurements.

Failure-mode indicators contain information directly associated with
failure outcomes, so their use must be justified according to the
intended prediction setting.

### PAPER / LITERATURE CONNECTION

Experiment 0 establishes the characteristics of the dataset that
motivate the research methodology.

In particular, the strong class imbalance provides the motivation for
evaluating models using metrics and decision criteria beyond accuracy.

The final paper should connect these observations with relevant
literature on imbalanced classification, predictive maintenance,
classification thresholds, and cost-sensitive decision-making.

### NEXT STEP

Define the final predictor and target variables, establish the
stratified development/final-test split, and begin Experiment 1:
baseline model evaluation.


## Experiment 1 — Baseline Models

**Date:** 2026-09-04

### WHAT WE DID

Established an 80/20 stratified development/final-test split and
evaluated three baseline classification models:

- Logistic Regression
- Decision Tree
- Random Forest

The 8,000-row development set was evaluated using Stratified 5-Fold
Cross-Validation.

The final 2,000-row test set was kept untouched during model
development.

The modeling feature set consisted of:

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

Numerical features were standardized using `StandardScaler`, while the
categorical `Type` feature was transformed using `OneHotEncoder`.

Preprocessing and model fitting were combined in a scikit-learn
`Pipeline` with a `ColumnTransformer` to ensure that preprocessing was
fitted only on the training portion of each cross-validation fold.

Baseline predictions used a decision threshold of 0.5.

The experiment was implemented in:

`experiments/experiment_01_baseline/run_experiment.py`

### WHY

The purpose of Experiment 1 was to establish baseline performance for
the selected classical machine-learning models before investigating
decision thresholds, minimum recall requirements, and asymmetric
false-positive/false-negative costs.

The baseline provides a reference point against which the later
experiments can be compared.

### RESULT

The dataset was split into:

- 8,000 development observations
- 2,000 final-test observations

The development set contained:

- 7,729 normal observations
- 271 failure observations

The final test set contained:

- 1,932 normal observations
- 68 failure observations

The final test set was not used during cross-validation or baseline
model selection.

Mean Stratified 5-Fold Cross-Validation results at a threshold of 0.5
were:

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.1993 | 0.7607 | 0.3119 | 0.8938 |
| Decision Tree | 0.6716 | 0.7051 | 0.6868 | 0.8308 |
| Random Forest | 0.4762 | 0.8623 | 0.6108 | 0.9665 |

### IMPORTANT NUMBERS

- Development observations: 8,000
- Development failures: 271
- Final-test observations: 2,000
- Final-test failures: 68
- Development/final-test split: 80/20
- Stratified split: Yes
- Cross-validation: Stratified 5-Fold
- Number of folds: 5
- Random seed: 42
- Baseline decision threshold: 0.5

Mean CV results:

- Logistic Regression recall: 0.1993
- Logistic Regression precision: 0.7607
- Logistic Regression F1: 0.3119
- Logistic Regression ROC-AUC: 0.8938

- Decision Tree recall: 0.6716
- Decision Tree precision: 0.7051
- Decision Tree F1: 0.6868
- Decision Tree ROC-AUC: 0.8308

- Random Forest recall: 0.4762
- Random Forest precision: 0.8623
- Random Forest F1: 0.6108
- Random Forest ROC-AUC: 0.9665

### OBSERVATION

At the default decision threshold of 0.5, the three models showed
substantially different failure-detection behavior.

Decision Tree achieved the highest mean recall and F1 among the three
baseline models.

Random Forest achieved the highest mean precision and ROC-AUC.

Logistic Regression achieved substantially lower mean recall at the
default threshold.

An important observation was the difference between Random Forest's
high ROC-AUC (0.9665) and its lower recall (0.4762) at the 0.5
threshold. This demonstrates that a model's discrimination across
thresholds and its performance at one particular decision threshold
are different properties.

The baseline results therefore do not provide sufficient justification
for selecting a final model.

### DECISION

No final model was selected at this stage.

The three models will be retained for further threshold and
cost-sensitive analysis.

### WHY

The research question concerns the joint effect of model choice,
decision threshold, asymmetric false-positive/false-negative costs,
and minimum recall requirements.

Selecting a final model solely from default-threshold performance
could therefore overlook important differences that emerge when the
decision threshold and operational costs are changed.

The strong ROC-AUC of Random Forest together with its lower recall at
the default threshold provides additional motivation for investigating
its probability predictions across different thresholds.

### PAPER / LITERATURE CONNECTION

Experiment 1 establishes the baseline performance against which the
subsequent decision-threshold and cost-sensitive experiments will be
compared.

The results also demonstrate why model evaluation in an imbalanced
failure-prediction problem should not rely on a single metric or on
accuracy alone.

Recall and precision describe performance at a selected decision
threshold, whereas ROC-AUC provides a supporting measure of the
model's ability to discriminate between failure and normal observations
across thresholds.

The final paper should connect these observations with relevant
literature on imbalanced classification, predictive maintenance,
classification threshold selection, and cost-sensitive decision-making.

### NEXT STEP

Generate out-of-fold probability predictions for the development set.

These predictions will allow the same development observations to be
evaluated across different decision thresholds without using the final
test set.

The resulting probability predictions will form the basis for the
subsequent threshold, minimum-recall, and cost-sensitive experiments.

## Experiment 2 — Out-of-Fold Probability Predictions

**Date:** 2026-09-05

### WHAT WE DID

Generated out-of-fold (OOF) probability predictions for all three
baseline models on the 8,000-row development dataset:

- Logistic Regression
- Decision Tree
- Random Forest

The same Stratified 5-Fold Cross-Validation configuration from
Experiment 1 was used.

For each fold, the model was trained on four folds and generated
failure probabilities for the held-out fifth fold. After all five
folds, every development observation had exactly one probability
prediction from each model.

The original development-set row index and the fold associated with
each prediction were preserved.

The final 2,000-row test set remained untouched.

### WHY

The research question requires investigating the effect of changing
the decision threshold.

Threshold analysis requires model probability outputs rather than only
fixed 0/1 predictions.

Using OOF probabilities prevents threshold analysis from relying on
predictions made on observations that were used to train the
corresponding model.

This provides a leakage-safe basis for the subsequent threshold
experiments.

### RESULT

Generated:

- 8,000 development-set OOF predictions
- 3 model probability predictions per observation
- 5 folds
- 1,600 observations assigned to each fold

Output file:

`experiments/experiment_02_probability/oof_predictions.csv`

Metadata file:

`results/experiment_metadata/experiment_02_probability.json`

The probability columns are:

- `logistic_regression_probability`
- `decision_tree_probability`
- `random_forest_probability`

All probability values passed validation checks and were within the
range [0, 1].

### IMPORTANT NUMBERS

- Development observations: 8,000
- Models: 3
- CV folds: 5
- Observations per fold: 1,600
- OOF predictions per model: 8,000
- Total model probability predictions: 24,000
- Random seed: 42
- Final test observations: 2,000
- Final test used: No

### OBSERVATION

OOF prediction generation provides one independent probability estimate
for every development observation from each of the three models.

The probability outputs show that the models can assign different
failure probabilities to the same observation. These probabilities can
therefore be evaluated under different decision thresholds.

Decision Tree probabilities can include values such as 0.0 because
tree-based probability estimates are determined by the class
distribution within terminal leaves.

### DECISION

Use the OOF probabilities as the fixed development-set predictions
for the subsequent threshold analysis.

Do not use the final test set for threshold selection.

### WHY

The next research question is not simply which model performs best at
the default threshold of 0.5, but how model performance changes when
the decision threshold is modified.

Using OOF probabilities allows this analysis to be performed on
development data while keeping the final test set reserved for the
final evaluation.

### PAPER / LITERATURE CONNECTION

This experiment establishes the probability outputs required for
decision-threshold analysis in an imbalanced classification setting.

The experiment supports the broader research design in which model
selection is considered jointly with the operating threshold rather
than assuming that a threshold of 0.5 is universally appropriate.

### NEXT STEP

Perform Experiment 3 — Threshold Analysis.

Evaluate all three models across a range of decision thresholds using
the OOF probabilities and calculate recall, precision, F1, false
positives, false negatives, and false alarm rate.

Do not use the final test set.

## Experiment 3 — Threshold Analysis

**Date:** 2026-09-05

### WHAT WE DID

Evaluated the effect of decision thresholds on the three baseline
models using the 8,000 out-of-fold probability predictions generated in
Experiment 2.

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Nineteen thresholds from 0.05 to 0.95 were evaluated for each model,
resulting in 57 model-threshold combinations.

For each combination, the following were calculated:

- Recall
- Precision
- F1
- False Alarm Rate (FAR)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- True Positives (TP)

The evaluation used pooled OOF predictions across all 8,000
development observations.

The final test set was not used.

### WHY

The research question investigates how the decision threshold affects
failure detection and false alarms.

A default threshold of 0.5 does not necessarily provide the desired
operating point for an imbalanced industrial failure-prediction
problem.

Threshold analysis therefore provides the evidence needed to determine
how recall and false-alarm behavior change before applying explicit
minimum-recall requirements and asymmetric failure/false-alarm costs.

### RESULT

All three models were successfully evaluated at 19 thresholds.

#### Logistic Regression

At threshold 0.50:

- Recall: 0.1993
- Precision: 0.7297
- F1: 0.3130
- FP: 20
- FN: 217

The best F1 among the tested thresholds occurred at threshold 0.20:

- Recall: 0.5277
- Precision: 0.4689
- F1: 0.4965
- FP: 162
- FN: 128

Lowering the threshold substantially increased recall while also
increasing false positives.

#### Decision Tree

The Decision Tree produced the same classification results across all
tested thresholds:

- Recall: approximately 0.6716
- Precision: approximately 0.7027
- F1: approximately 0.6868
- FP: 77
- FN: 89

This indicates that the OOF probability outputs of the fitted tree are
effectively concentrated such that the tested thresholds do not change
the resulting classifications.

#### Random Forest

At threshold 0.50:

- Recall: 0.4760
- Precision: 0.8600
- F1: 0.6128
- FP: 21
- FN: 142

At threshold 0.30:

- Recall: 0.7417
- Precision: 0.7336
- F1: 0.7376
- FP: 73
- FN: 70

The best F1 among the tested thresholds occurred at threshold 0.30.

At threshold 0.20:

- Recall: 0.8118
- Precision: 0.5699
- F1: 0.6697
- FP: 166
- FN: 51

At threshold 0.05:

- Recall: 0.9188
- Precision: 0.2663
- F1: 0.4129
- FP: 686
- FN: 22

### IMPORTANT NUMBERS

- Development observations: 8,000
- Models: 3
- Thresholds: 19
- Model-threshold combinations: 57
- Final test used: No

Random Forest:

- Threshold 0.50 → Recall 0.4760
- Threshold 0.30 → Recall 0.7417
- Threshold 0.20 → Recall 0.8118
- Threshold 0.10 → Recall 0.8782
- Threshold 0.05 → Recall 0.9188

Best tested F1:

- Logistic Regression → threshold 0.20, F1 0.4965
- Decision Tree → threshold 0.05, F1 0.6868
- Random Forest → threshold 0.30, F1 0.7376

### OBSERVATION

Decision threshold has a substantial effect on Logistic Regression and
Random Forest.

For Random Forest, reducing the threshold from 0.50 to 0.20 increased
recall from 0.4760 to 0.8118, but increased false positives from 21 to
166.

Reducing the threshold further to 0.05 increased recall to 0.9188, but
also increased false positives to 686 and reduced precision to 0.2663.

This demonstrates the central trade-off between detecting more failures
and generating more false alarms.

The Decision Tree showed almost no change across the tested thresholds,
indicating that its probability outputs have limited resolution for
this threshold sweep.

### DECISION

Do not select a final model or threshold based on F1 alone.

Retain all three models for the minimum-recall and cost-sensitive
experiments.

Use pooled OOF predictions consistently for subsequent threshold
selection.

Keep the final test set untouched.

### WHY

The project's research question explicitly includes minimum failure
detection requirements and asymmetric costs.

Therefore, the threshold producing the highest F1 is not necessarily
the operationally preferred threshold.

A threshold should instead be evaluated against the required recall
and the relative consequences of false positives and false negatives.

### PAPER / LITERATURE CONNECTION

The experiment demonstrates that classification performance in an
imbalanced predictive-maintenance setting depends not only on the
underlying model but also on the operating decision threshold.

The results support evaluating threshold-dependent metrics rather than
relying exclusively on accuracy or a default threshold of 0.5.

The observed recall-versus-false-alarm trade-off provides the empirical
basis for the subsequent minimum-recall and asymmetric-cost analysis.

### NEXT STEP

Perform Experiment 4 — Minimum Recall Constraint.

For each model, determine which tested threshold satisfies specified
minimum failure-recall requirements and compare the resulting false
alarms, precision, F1, FP, and FN.

Do not use the final test set.