# Research Findings

## 1. Dataset Characteristics and Class Imbalance

The study uses the UCI AI4I 2020 Predictive Maintenance Dataset, which contains
10,000 observations of industrial equipment operating conditions.

The target variable, `Machine failure`, contains:

- Normal observations: 9,661 (96.61%)
- Failure observations: 339 (3.39%)

This shows a strong class imbalance, with failure observations representing only
a small proportion of the dataset.

### Observation

A classifier evaluated primarily using accuracy could appear successful simply
by predicting the majority class in many cases. Therefore, accuracy is not used
as the primary measure of model performance in this study.

Instead, the analysis focuses on metrics that directly describe failure
detection and false alarms:

- Recall: proportion of actual failures that are detected.
- Precision: proportion of predicted failures that are actual failures.
- F1-score: balance between precision and recall.
- False alarm rate (FAR): proportion of normal observations incorrectly
  classified as failures.
- False positives (FP): normal observations incorrectly flagged as failures.
- False negatives (FN): actual failures that are missed.

ROC-AUC is used as a supporting metric because it evaluates the model's ability
to rank positive and negative observations across classification thresholds.

### Interpretation

The class imbalance makes the distinction between false positives and false
negatives particularly important. In an equipment-failure prediction setting,
a false negative represents a missed failure, while a false positive represents
an unnecessary alarm.

The experiments therefore treat classification as a decision problem rather
than evaluating models only at the default classification threshold.

### Research significance

This dataset characteristic motivates the subsequent experiments on:

1. classification thresholds,
2. minimum failure-recall requirements, and
3. asymmetric costs for false positives and false negatives.

These experiments investigate whether model selection changes when the desired
failure-detection requirement and the relative cost of prediction errors are
changed.

### Evidence to preserve for the paper

- Dataset size: 10,000 observations.
- Normal observations: 9,661 (96.61%).
- Failure observations: 339 (3.39%).
- Failure is the minority class.
- Accuracy is therefore not treated as the primary evaluation metric.
- Recall, precision, F1, FAR, FP, FN and ROC-AUC are used to describe model
  behavior.

## 2. Baseline Model Behavior

Experiment 1 evaluated Logistic Regression, Decision Tree, and Random Forest
using the default classification threshold of 0.50.

The models produced noticeably different failure-detection behavior.

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 19.93% | 76.07% | 0.3119 | 0.8938 |
| Decision Tree | 67.16% | 70.51% | 0.6868 | 0.8308 |
| Random Forest | 47.62% | 86.23% | 0.6108 | 0.9665 |

### Observation

At the default threshold of 0.50, the three models behaved differently.

Logistic Regression detected only about one-fifth of the actual failures,
although its precision was relatively high.

Decision Tree achieved the highest recall and F1-score at the default
threshold among the three models.

Random Forest achieved the highest precision and ROC-AUC, but its recall at
the default threshold was only 47.62%.

### Interpretation

The baseline results show that selecting a model based on a single metric can
lead to different conclusions.

For example, Random Forest had the highest ROC-AUC, indicating strong overall
ranking ability across thresholds, but it did not have the highest recall at
the default 0.50 threshold.

Similarly, Decision Tree had the highest recall at 0.50, but this did not
automatically establish it as the best model for the complete research
objective.

This indicates that model evaluation depends on the operating threshold and
the metric or decision requirement being considered.

### Research significance

The baseline experiment therefore motivated the next question:

> If the same trained model produces probability scores, how does changing the
> classification threshold affect failure detection and false alarms?

This question became the focus of Experiment 3.

Experiment 2 was used to generate out-of-fold probabilities for the development
data so that threshold behavior could be studied without evaluating the final
test set.

### Evidence to preserve for the paper

- Logistic Regression recall at 0.50: 19.93%.
- Decision Tree recall at 0.50: 67.16%.
- Random Forest recall at 0.50: 47.62%.
- Random Forest precision at 0.50: 86.23%.
- Random Forest ROC-AUC: 0.9665.
- Decision Tree had the highest recall and F1 at the default threshold.
- Random Forest had the highest ROC-AUC and precision.
- No final model was selected from the baseline experiment.

## 3. Classification Threshold Creates a Recall–False-Alarm Trade-off

Experiment 3 evaluated model predictions across thresholds from 0.05 to 0.95
using pooled out-of-fold predictions from the development data.

The Random Forest showed a clear change in failure-detection behavior as the
threshold was lowered.

| Threshold | Recall | Precision | F1 | False Positives | False Negatives |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 47.60% | 86.00% | 0.6128 | 21 | 142 |
| 0.30 | 74.17% | 73.36% | 0.7376 | 73 | 70 |
| 0.20 | 81.18% | 56.99% | 0.6697 | 166 | 51 |
| 0.10 | 87.82% | 38.26% | 0.5330 | 384 | 33 |
| 0.05 | 91.88% | 26.63% | 0.4129 | 686 | 22 |

### Observation

Lowering the Random Forest threshold increased the proportion of actual
failures detected.

For example, reducing the threshold from 0.50 to 0.20 increased recall from
47.60% to 81.18%. At the same time, false positives increased from 21 to 166,
while false negatives decreased from 142 to 51.

Further lowering the threshold from 0.20 to 0.05 increased recall to 91.88%
and reduced false negatives to 22, but increased false positives substantially
to 686.

Precision also decreased as the threshold was lowered, reflecting the larger
number of normal observations being classified as failures.

### Interpretation

The results demonstrate a clear trade-off between failure detection and false
alarms.

A lower threshold makes the classifier more willing to classify an observation
as a failure. This helps reduce missed failures but produces more false alarms.

Within the tested threshold range, no operating point simultaneously maximized failure recall and minimized false positives.

The appropriate operating point depends on what level of failure detection is
required and how costly false positives and false negatives are.

### Important finding

The default threshold of 0.50 would have detected only 47.60% of failures for
the Random Forest in the pooled out-of-fold development predictions.

The threshold of 0.20 increased failure recall to 81.18%, but this improvement
came with a substantial increase in false positives.

This demonstrates why model selection alone is insufficient for the decision
problem investigated in this study.

### Research significance

The threshold experiment provides the first direct evidence supporting the
central research question.

The trained model and the final classification decision are separate parts of
the prediction system. A model can produce useful probability estimates, but
the threshold determines how those estimates are converted into failure or
normal decisions.

Consequently, model evaluation should consider both model performance and the
operating threshold.

### Evidence to preserve for the paper

- Random Forest recall at threshold 0.50: 47.60%.
- Random Forest recall at threshold 0.30: 74.17%.
- Random Forest recall at threshold 0.20: 81.18%.
- Random Forest recall at threshold 0.10: 87.82%.
- Random Forest recall at threshold 0.05: 91.88%.
- False positives increased from 21 at threshold 0.50 to 686 at threshold
  0.05.
- False negatives decreased from 142 at threshold 0.50 to 22 at threshold
  0.05.
- The threshold of 0.20 provided the operating point later selected under the
  80% minimum-recall and 1:5 FP/FN cost policy.

## 4. Minimum Recall Requirements Can Eliminate Models

Experiment 4 investigated whether each model could satisfy predefined minimum
failure-recall requirements.

The requirements examined were 60%, 70%, 80%, 90%, and 95% recall. For each
requirement, a model was considered feasible if at least one tested threshold
achieved recall equal to or greater than the required value.

| Minimum Recall Requirement | Logistic Regression | Decision Tree | Random Forest |
|---:|:---:|:---:|:---:|
| 60% | Feasible | Feasible | Feasible |
| 70% | Feasible | Not feasible | Feasible |
| 80% | Not feasible | Not feasible | Feasible |
| 90% | Not feasible | Not feasible | Feasible |
| 95% | Not feasible | Not feasible | Not feasible |

The threshold search used in this experiment covered values from 0.05 to 0.95
in increments of 0.05.

### Observation

At a minimum recall requirement of 60%, all three models had at least one
tested threshold satisfying the requirement.

When the requirement was increased to 70%, Decision Tree no longer had a
feasible tested threshold.

At an 80% requirement, only Random Forest remained feasible.

Random Forest also remained feasible at the 90% requirement, whereas Logistic
Regression and Decision Tree did not.

At 95%, no tested model/threshold combination reached the required recall.

### Interpretation

Increasing the minimum acceptable recall progressively reduced the set of
feasible model and threshold combinations.

This means that a model that appears competitive under conventional metrics
may not be suitable when the application imposes a strict failure-detection
requirement.

For example, at the 80% minimum-recall requirement, Random Forest was the only
model with a tested threshold that satisfied the requirement. Its selected
threshold under the Experiment 4 F1-based rule was 0.20, producing 81.18%
recall.

At the 90% requirement, Random Forest required a much lower threshold of 0.05,
which increased recall to 91.88% but also produced 686 false positives.

### Research significance

The results show that minimum recall can act as a practical model-selection
constraint rather than merely being another metric reported after training.

Instead of asking only which model has the highest average score, the
experiment asks whether a model can operate within the required failure
detection level.

This provides a more decision-oriented way of comparing models for imbalanced
failure prediction.

### Important finding

Random Forest was the only model that remained feasible at both the 80% and
90% minimum-recall requirements within the tested threshold grid.

This finding contributed directly to the final model selection.

However, feasibility alone does not determine the final operating point. The
false-positive and false-negative consequences must also be considered. This
motivated the cost-sensitive analysis in Experiment 5.

### Evidence to preserve for the paper

- Minimum recall requirements tested: 60%, 70%, 80%, 90%, and 95%.
- At 80% minimum recall, only Random Forest was feasible.
- At 90% minimum recall, only Random Forest was feasible.
- At 95%, no tested model/threshold combination reached the requirement.
- Random Forest at threshold 0.20 achieved 81.18% recall with 166 FP and 51 FN.
- Random Forest at threshold 0.05 achieved 91.88% recall with 686 FP and 22 FN.
- Higher recall requirements can substantially increase the false-alarm burden.

## 5. Asymmetric Error Costs Change the Preferred Operating Point

Experiment 5 investigated how the preferred model and classification threshold
change when false negatives are assigned progressively higher relative costs.

Four illustrative cost scenarios were evaluated:

| Scenario | FP Cost | FN Cost | Interpretation |
|---|---:|---:|---|
| C1 | 1 | 1 | FP and FN have equal relative cost |
| C2 | 1 | 2 | FN costs twice as much as FP |
| C3 | 1 | 5 | FN costs five times as much as FP |
| C4 | 1 | 10 | FN costs ten times as much as FP |

These are relative experimental cost values rather than measured monetary
costs from an industrial deployment.

For each scenario, total cost was calculated from the numbers of false
positives and false negatives. The lowest-cost configuration among the tested
models and thresholds was selected for each scenario.

### Cost-only model and threshold selection

| Cost Scenario | Selected Model | Threshold | Recall | FP | FN | Total Cost |
|---|---|---:|---:|---:|---:|---:|
| C1 (1:1) | Random Forest | 0.35 | 67.53% | 47 | 88 | 135 |
| C2 (1:2) | Random Forest | 0.30 | 74.17% | 73 | 70 | 213 |
| C3 (1:5) | Random Forest | 0.25 | 78.23% | 111 | 59 | 406 |
| C4 (1:10) | Random Forest | 0.15 | 84.87% | 255 | 41 | 665 |

### Observation

Random Forest produced the lowest total cost among the three models in all
four cost scenarios.

As the relative cost of a missed failure increased, the cost-minimizing
threshold for Random Forest generally decreased:

- C1: 0.35
- C2: 0.30
- C3: 0.25
- C4: 0.15

The lower thresholds resulted in higher recall and fewer false negatives, but
also increased the number of false positives.

### Interpretation

The preferred operating threshold depends on the relative consequences of
false positives and false negatives.

When false negatives become more expensive, the selection process favors
operating points that reduce missed failures, even when doing so produces more
false alarms.

Therefore, there is no single universally optimal threshold independent of
the decision costs.

### Interaction with the minimum-recall requirement

Experiment 5 also combined the cost scenarios with minimum-recall
requirements.

Under the final policy of at least 80% recall and a relative cost of
FP = 1 and FN = 5, Random Forest at threshold 0.20 was the selected feasible
configuration.

This configuration produced:

- Recall: 81.18%
- Precision: 56.99%
- F1-score: 0.6697
- False positives: 166
- False negatives: 51
- Total relative cost: 421

At the 80% minimum-recall requirement, Logistic Regression and Decision Tree
were infeasible within the tested threshold grid, leaving Random Forest as the
only feasible model.

### Research significance

The cost-sensitive experiment demonstrates that model selection and threshold
selection cannot be considered independently when the consequences of
prediction errors are asymmetric.

The same model may require a different operating threshold depending on the
relative importance assigned to missed failures and false alarms.

This provides evidence for treating model choice, threshold selection, recall
requirements, and error costs as connected components of the prediction
decision.

### Important finding

Increasing the relative cost of false negatives generally shifted the
cost-optimal Random Forest threshold downward, increasing failure detection
while accepting a larger false-alarm burden.

This supports the use of application-specific decision criteria rather than
selecting a threshold solely from a generic metric such as F1-score.

### Evidence to preserve for the paper

- Four relative cost scenarios were evaluated: 1:1, 1:2, 1:5, and 1:10
  (FP:FN).
- Random Forest was the lowest-cost model in all four cost-only scenarios.
- RF cost-optimal threshold changed from 0.35 under C1 to 0.15 under C4.
- Under the final C3 policy with an 80% minimum-recall requirement, RF at
  threshold 0.20 was selected.
- Final development/OOF policy result: recall 81.18%, FP 166, FN 51,
  total relative cost 421.
- The cost values are illustrative relative costs and should not be presented
  as real industrial monetary estimates.

## 6. Final Model and Operating-Point Selection

Experiment 6 combined the findings from the previous experiments into a single
operating policy.

The final policy required:

- Minimum recall: 80%
- False-positive relative cost: 1
- False-negative relative cost: 5
- Cost scenario: C3
- Model: Random Forest
- Threshold: 0.20

The selection was performed using the development data and the out-of-fold
predictions generated during the earlier experiments. The final test set was
not used during this selection.

### Selection result

The selected Random Forest configuration at threshold 0.20 produced:

| Metric | Development / OOF result |
|---|---:|
| Recall | 81.18% |
| Precision | 56.99% |
| F1-score | 0.6697 |
| False alarm rate | 2.15% |
| False positives | 166 |
| False negatives | 51 |
| True positives | 220 |
| True negatives | 7,563 |
| Total relative cost | 421 |

The total relative cost was calculated using the selected cost policy, where a
false positive has relative cost 1 and a false negative has relative cost 5.

### Observation

The selected configuration satisfies the required minimum recall of 80% while
using the Random Forest operating point that was identified under the
specified cost-sensitive decision policy.

At threshold 0.20, the model detected 81.18% of failures in the development
out-of-fold predictions.

### Interpretation

The final configuration was not selected solely because Random Forest had the
highest ROC-AUC or because threshold 0.20 produced the highest F1-score.

Instead, the selection resulted from combining:

1. model feasibility under the minimum-recall requirement,
2. the relative costs of false positives and false negatives, and
3. the resulting operating threshold.

This makes the selected configuration a policy-driven operating point rather
than simply the model with the highest value of one performance metric.

### Research significance

Experiment 6 connects the individual experimental findings into the final
decision rule used for evaluation.

Importantly, the selected configuration was frozen before the final test set
was evaluated. This prevents the final test observations from influencing the
model or threshold selection.

The threshold should be described as the selected operating point within the
tested threshold grid rather than as a universally optimal threshold.

### Evidence to preserve for the paper

- Final model: Random Forest.
- Final threshold: 0.20.
- Minimum recall requirement: 80%.
- Cost scenario: C3.
- Relative costs: FP = 1, FN = 5.
- Development/OOF recall: 81.18%.
- Development/OOF precision: 56.99%.
- Development/OOF F1: 0.6697.
- Development/OOF FP: 166.
- Development/OOF FN: 51.
- Development/OOF total relative cost: 421.
- Final configuration was frozen before final-test evaluation.

## 7. Final Evaluation on the Untouched Test Set

Experiment 7 evaluated the frozen Random Forest configuration on the final
2,000-row test set.

The final test set was separated before model development and was not used for
model selection, threshold selection, or cost-sensitive analysis.

The selected configuration was:

- Model: Random Forest
- Threshold: 0.20
- Minimum recall requirement: 80%
- Cost scenario: C3
- Relative FP cost: 1
- Relative FN cost: 5

### Final test result

| Metric | Final Test |
|---|---:|
| Test observations | 2,000 |
| Actual failures | 68 |
| Actual normal observations | 1,932 |
| True positives | 56 |
| False negatives | 12 |
| False positives | 43 |
| True negatives | 1,889 |
| Recall | 82.35% |
| Precision | 56.57% |
| F1-score | 0.6707 |
| False alarm rate | 2.23% |
| ROC-AUC | 0.9653 |
| Total relative cost | 103 |

### Observation

The frozen Random Forest configuration detected 56 of the 68 actual failures in
the final test set, resulting in a recall of 82.35%.

It missed 12 failures and generated 43 false alarms among the 1,932 normal
observations.

The resulting F1-score was 0.6707 and ROC-AUC was 0.9653.

### Development-to-test comparison

The selected configuration produced similar recall and F1-score on the
development out-of-fold predictions and the untouched final test set.

| Metric | Development / OOF | Final Test |
|---|---:|---:|
| Recall | 81.18% | 82.35% |
| Precision | 56.99% | 56.57% |
| F1-score | 0.6697 | 0.6707 |
| False alarm rate | 2.15% | 2.23% |

The similarity of these values indicates that the frozen operating point
showed broadly consistent behavior when applied to the unseen test data.

The comparison should be interpreted as evidence of consistency for this
dataset and experimental split, rather than as evidence of generalization to
other industrial environments.

### Interpretation

The final test evaluation provides an independent assessment of the
configuration selected through the development experiments.

Because the test set remained untouched until the final evaluation, the
reported test metrics were not used to choose the model or threshold.

The final recall of 82.35% also remained above the predefined 80% minimum-recall
requirement.

However, the model still produced 43 false alarms and missed 12 actual
failures. Therefore, increasing failure detection does not eliminate the
trade-off between false positives and false negatives.

### Research significance

The final test result completes the experimental process:

1. Models were compared on development data.
2. Out-of-fold probabilities were generated.
3. Threshold behavior was analyzed.
4. Minimum recall requirements were evaluated.
5. Asymmetric error costs were investigated.
6. A model and operating threshold were selected and frozen.
7. The frozen configuration was evaluated once on the untouched test set.

This provides a controlled experimental basis for studying how model choice,
threshold selection, minimum recall requirements, and asymmetric error costs
affect machine-failure prediction decisions.

### Important finding

The final Random Forest configuration achieved 82.35% recall on the untouched
test set, compared with 81.18% recall during development/out-of-fold
evaluation.

Its F1-score was also very similar between development (0.6697) and final test
(0.6707).

This indicates that the selected operating point behaved consistently on the
held-out test set for this experimental setup.

### Evidence to preserve for the paper

- Final test set size: 2,000 observations.
- Actual failures: 68.
- True positives: 56.
- False negatives: 12.
- False positives: 43.
- True negatives: 1,889.
- Recall: 82.35%.
- Precision: 56.57%.
- F1-score: 0.6707.
- False alarm rate: 2.23%.
- ROC-AUC: 0.9653.
- Relative cost under C3: 103.
- Final test recall remained above the 80% minimum requirement.
- Development and final-test recall and F1-score were similar.
- Test results were obtained after freezing the model and threshold.