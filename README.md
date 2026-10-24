# Credit Risk Prediction & Default Analysis

A machine-learning analysis that predicts which loan applicants are likely to default, so a bank can catch high-risk applications **before** lending, without turning away good customers unnecessarily.

**Data:** 1,000 historical loan applicants (a version of the German Credit dataset), 30% of whom defaulted.
**Stack:** Python, pandas, scikit-learn, matplotlib, seaborn.

## Headline results

On 200 applicants the model had never seen (60 of them defaulters):

| Metric | Default cut-off (0.50) | Chosen cut-off (0.38) |
|---|---|---|
| Defaulters caught (recall) | 65% (39 of 60) | **83% (50 of 60)** |
| Good customers wrongly flagged | 19% (26 of 140) | 41% (58 of 140) |
| Precision | 60% | 46% |
| Accuracy | 76.5% | 66.0% |
| ROC-AUC | 0.804 | 0.804 |

- Flagging at the chosen cut-off covers about **93% of the defaulted loan value** in the test set (259,305 of 280,001).
- With only 200 test applicants, the 95% bootstrap range for recall is **74% to 92%** and for AUC **0.74 to 0.87**. Treat the exact figures as estimates.
- Accuracy is a poor yardstick here: an "approve everyone" rule is already 70% accurate and catches no defaulters.

## Approach

1. **Data audit:** missing values, duplicates, identifiers, mixed units (months and years in text columns), skewed amounts.
2. **EDA:** class balance, whether "missing" carries information, default rate per category, outliers, correlations.
3. **Preprocessing in one scikit-learn pipeline:** median imputation plus "was missing" indicators, scaling, one-hot encoding. Everything is learned from the training set only, so there is no test-set leakage.
4. **Models:** Logistic Regression, Decision Tree, Random Forest and SVM, all with class weighting for the 70/30 imbalance. Each was compared with 5-fold cross-validation and tuned with grid search on ROC-AUC.
5. **Model choice:** the rule was fixed in advance, best cross-validated AUC on the training data only. The test set was used once at the end.
6. **Cut-off policy:** chosen on cross-validated training predictions as "catch at least 80% of defaulters, and decline as few good customers as possible".
7. **Uncertainty:** 1,000-sample bootstrap on the test set, plus permutation importance.
8. **Fairness check:** does the model need gender, age or foreign-worker status, and does it treat groups differently?
9. **Business translation:** risk bands and loan-value impact.

## Model comparison

| Model | CV AUC (tuned) | Test AUC |
|---|---|---|
| Random Forest (selected) | 0.798 | 0.804 |
| SVM (RBF) | 0.796 | 0.841 |
| Logistic Regression | 0.773 | 0.820 |
| Decision Tree | 0.697 | 0.686 |

Random Forest, SVM and Logistic Regression are separated by less than their own fold-to-fold variation, so on 1,000 rows they are effectively tied. Only the Decision Tree is clearly worse. Random Forest was picked because of the pre-set selection rule; Logistic Regression is the natural fallback if every coefficient must be explainable to a regulator.

## Choosing the cut-off

The cut-off is a business policy, not a statistical fact. Trade-offs on the training data (cross-validated):

| Policy | Cut-off | Defaulters caught | Good customers declined |
|---|---|---|---|
| Catch at least 70% | 0.47 | 70% | 22% |
| **Catch at least 80% (chosen)** | **0.38** | **81%** | **38%** |
| Catch at least 90% | 0.30 | 90% | 57% |
| Lowest cost at 5:1 cost ratio | 0.24 | 98% | 68% |

The 80% target and the 5:1 cost ratio are assumptions. A real bank would replace them with its own loss and margin figures.

## What drives default

- **Checking-account status** is by far the strongest signal. Applicants with no checking-account record default far less (12%) than those with one (42%), so "missing" is informative and is kept as a feature.
- Next come **loan amount**, **purpose**, **monthly repayment size**, **loan duration** and **credit history**.
- One counter-intuitive pattern: applicants with a "critical" credit history default less (17%) than those who fully repaid earlier loans (57% to 62%). The labels may not mean what they appear to, so this should be checked with whoever recorded the data.

## Fairness

- Removing gender, personal status, age and foreign-worker status changes cross-validated AUC only from 0.798 to 0.792 (test AUC 0.804 to 0.798). **The model does not need them.**
- Even so, among *good* customers, 58% of those under 25 and 44% of women were wrongly flagged, against 32% to 36% for other groups. Part of that reflects genuinely different default rates, but it is the fairness risk to watch.
- Some groups are small (31 non-foreign workers), so these differences are indicative, not proof.
- The notebook's final model still uses all features. The recommendation is to train the production model without these four and repeat the group audit on every retrain. Legal and compliance teams should confirm which attributes are permitted.

## How a bank could use it

| Band (predicted risk) | Share of applicants | Actual default rate | Suggested handling |
|---|---|---|---|
| Low (< 0.25) | 24.5% | 8.2% | Fast-track approval |
| Medium (0.25 to 0.38) | 21.5% | 14.0% | Fast-track approval |
| High (0.38 to 0.60) | 37.0% | 36.5% | Manual review, or tighter terms |
| Very high (>= 0.60) | 17.0% | 67.6% | Decline or escalate |

## Limitations

- Only 1,000 applicants, so results are noisy. Re-validate on a larger, more recent sample before any real use.
- The data contains only approved loans, so we never see what would have happened to rejected applicants. This can bias any model trained on it.
- Loan amounts are used as the money at risk, with no recoveries assumed. The interest margin lost on declined good customers is not in the data, so it is reported as volume rather than profit.
- Monitor for drift and repeat the fairness audit after each retrain.

## Reproducibility

Random seed 42 throughout. The numbers above are from the executed notebook in this repo. A fresh re-run of the same code gave slightly different values (about 82% recall at a 0.27 cut-off), so small differences are possible; the main conclusions hold.

## Run it

Python 3.10+ (tested on 3.12).

```bash
pip install -r requirements.txt
# put credit.csv in this folder (see "Data" below), then:
jupyter notebook Credit_Risk_Analysis.ipynb
```

Running the notebook creates `figures/` (charts) and `results/` (cut-off tables, key metrics and the saved model).

## Data

`credit.csv` has 1,000 rows and 23 columns. The target is `default` (0 = good loan, 1 = defaulting loan). Inputs cover financial indicators (checking and savings balance, existing credits), loan terms (amount, duration, installment rate and plan, purpose), history and stability (credit history, employment length, residence history, property, housing, job) and applicant details (age, gender, personal status, foreign worker, dependents, telephone).

The data file is not included in this repository. Add your own copy of the dataset to the project folder before running.

## Repository contents

```
Credit_Risk_Analysis.ipynb   full analysis, written to be read top to bottom
requirements.txt
```

## Author

**Mehdi Ali** - [GitHub](https://github.com/mehdi-dev-code) - [LinkedIn](https://linkedin.com/in/mehdi-ali-a33881296)
