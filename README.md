# Sex-Stratified Analysis of ADHD

## Overview

This section documents the sex-stratified component of the WiDS Datathon Capstone analysis. The aim was to investigate whether behavioural predictors and functional-connectivity patterns associated with ADHD differed between females and males.

**Analysis framework:** Behavioural inference → sex-stratified machine learning → stable connectivity feature selection → formal Feature × Sex interaction testing.

The final analysis showed that behavioural associations were largely shared across sexes, whereas functional connectivity showed evidence of sex-specific associations. Two of 11 stable candidate connectivity edges demonstrated statistically significant Feature × Sex interactions after Benjamini–Hochberg FDR correction.

---

## 1. Research Question

> **Are the predictors and functional-connectivity patterns associated with ADHD different between males and females?**

The analysis distinguished between:

1. Average behavioural differences by sex.
2. Differences in the association between predictors and ADHD by sex.

A difference in average symptom level does not necessarily imply that the relationship between that symptom and ADHD differs by sex.

---

## 2. Study Population

The sex-stratified analysis included **1,213 participants**.

| Subgroup | Participants |
|---|---:|
| Female | 416 |
| Male | 797 |
| **Total** | **1,213** |

Female and male analyses were performed independently. Participants from one subgroup were not used in the training folds of the other subgroup.

> **Important:** The male subgroup was larger than the female subgroup, which may have influenced model stability and predictive performance.

---

## 3. Behavioural Analysis

### 3.1 Objective

To determine whether behavioural symptoms and parenting measures were associated with ADHD differently across sexes.

Measures included:

- Strengths and Difficulties Questionnaire (SDQ)
- Alabama Parenting Questionnaire (APQ)

### 3.2 Statistical Model

For each behavioural outcome, the general model was:

$$
Y = \beta_0 + \beta_1 ADHD + \beta_2 Sex +
\beta_3(ADHD \times Sex) + \epsilon
$$

The key parameter was the **Sex × ADHD interaction**:

$$
H_0: \beta_3 = 0
$$

A significant interaction would indicate that the association between the behavioural measure and ADHD differed between females and males.

Multiple comparisons were controlled using the **Benjamini–Hochberg false discovery rate (FDR)** procedure.

### 3.3 SDQ Findings

After FDR correction:

- Internalising scores differed by sex.
- Emotional Problems differed by sex.
- Prosocial Behaviour differed by sex.
- ADHD was associated with higher symptom scores.
- **No Sex × ADHD interaction remained significant for any SDQ domain.**

The smallest interaction p-value was **0.118**.

### 3.4 APQ Findings

APQ measures showed the same overall pattern:

- Some baseline differences between sexes were observed.
- No meaningful ADHD effect was identified.
- No Sex × ADHD interaction remained significant after correction.

### 3.5 Interpretation

Although females and males differed in average behavioural profiles, the relationship between ADHD and behavioural symptoms was largely similar across sexes.

> **Take-home message:** Behavioural predictors of ADHD were largely shared between females and males.

This motivated the next question:

> **If behavioural associations are largely similar, do functional-connectivity patterns differ by sex?**

---

## 4. Sex-Stratified Machine-Learning Analysis

### 4.1 Objective

To investigate whether different functional-connectivity features were useful for ADHD prediction in females and males.

Separate models were fitted within each sex using:

- Clinical/behavioural metadata
- fMRI functional-connectivity features

### 4.2 Why Stratify by Sex?

A pooled model implicitly assumes that the relationship between predictors and ADHD is sufficiently similar across sexes.

Sex-stratified models allow the predictive structure to differ:

$$
P(ADHD \mid X, Sex=Female)
\neq
P(ADHD \mid X, Sex=Male)
$$

This provides an opportunity to identify connectivity patterns that may be obscured when females and males are analysed together.

---

## 5. Leakage-Safe Machine-Learning Pipeline

The sex-stratified models used **stratified 5-fold cross-validation independently within each sex**.

### Within each training fold

1. Separate female and male data.
2. Perform preprocessing using training data only.
3. Apply Fisher's z transformation to connectivity measures.
4. Rank connectivity edges within the training fold.
5. Retain the top 50 candidate connectivity edges.
6. Train Logistic Regression and LightGBM models.
7. Evaluate predictions on the held-out fold.

### Leakage Prevention

All data-dependent operations were performed inside the training fold.

Therefore:

> **The validation fold was never used to determine imputation, edge ranking, or feature selection.**

Stable features were defined after cross-validation as features selected in **at least 4 of 5 folds (≥80%)**.

### Figure placeholder

**Figure 1. Sex-stratified, leakage-safe machine-learning pipeline.**

`![Sex-stratified ML pipeline](figures/sex_stratified_pipeline.png)`

---

## 6. Model Performance

The models demonstrated moderate ability to distinguish ADHD status in both subgroups.

| Sex | Best-performing model | ROC-AUC |
|---|---|---:|
| Female | LightGBM | **0.698** |
| Male | Logistic Regression | **0.710** |

Logistic Regression was the most consistent performer overall, while LightGBM achieved the highest AUC in females.

### Interpretation

The purpose of the ML stage was not only to maximise prediction performance. More importantly, it provided a systematic way of identifying **reproducible candidate connectivity features** for subsequent statistical testing.

---

## 7. Stable Functional-Connectivity Features

### 7.1 Definition

A connectivity edge was classified as **stable** if it was selected in at least **4 of the 5 cross-validation folds**.

This threshold corresponds to a selection frequency of at least **80%**.

### 7.2 Female Stable Edges

**8 stable edges** were identified:

| Connectivity pattern |
|---|
| Somatomotor ↔ Dorsal Attention |
| Visual ↔ Somatomotor |
| Visual ↔ Control |
| Limbic ↔ Control |
| Limbic ↔ Somatomotor |
| Limbic ↔ Somatomotor |
| Control ↔ Visual |
| Control ↔ Limbic |

### 7.3 Male Stable Edges

**3 stable edges** were identified:

| Connectivity pattern |
|---|
| Limbic ↔ Somatomotor |
| Default Mode ↔ Limbic |
| Default Mode ↔ Control |

### 7.4 Key Observation

There was **zero overlap** between the female and male stable-edge sets.

### Interpretation

The sex-stratified ML models selected different functional-connectivity patterns for ADHD prediction.

However, different features being selected by separate ML models does **not** itself establish that their statistical associations with ADHD differ by sex.

Therefore, the stable edges were taken forward for formal interaction testing.

### Figure placeholder

**Figure 2. Stable functional-connectivity edges identified separately in females and males.**

`![Stable connectivity edges](figures/stable_edges.png)`

---

# 8. Formal Feature × Sex Interaction Analysis

## 8.1 Why Was Interaction Analysis Needed?

Machine learning was used for **candidate discovery**.

It answered:

> Which connectivity features are repeatedly useful for prediction within each sex?

It did not directly answer:

> Does the association between a particular connectivity edge and ADHD statistically differ between females and males?

To address this question, targeted logistic regression interaction analyses were performed.

## 8.2 Statistical Model

For each of the 11 candidate edges:

$$
\text{logit}\{P(ADHD=1)\}
=
\beta_0 +
\beta_1 Edge +
\beta_2 Sex +
\beta_3(Edge\times Sex) +
\beta_4 Age +
\beta_5 Site
$$

The primary parameter of interest was:

$$
\beta_3
$$

representing the **Edge × Sex interaction**.

A statistically significant interaction indicates evidence that the association between the connectivity edge and ADHD differs between females and males.

### Covariate adjustment

The interaction models adjusted for:

- Age at scan
- Site group

### Multiple testing

The 11 interaction tests were corrected using the **Benjamini–Hochberg FDR** procedure.

---

## 8.3 Model Diagnostics

All fitted interaction models successfully converged.

| Diagnostic | Result |
|---|---|
| Binary outcome | ADHD yes/no ✓ |
| Independence | One participant per row ✓ |
| Multicollinearity | Maximum VIF = 2.67 ✓ |
| Linearity of logit | Satisfied ✓ |
| Model convergence | 11/11 models converged ✓ |
| Multiple testing | Benjamini–Hochberg FDR applied ✓ |

---

# 9. Interaction Results

Of the **11 stable candidate connectivity edges**, **2 remained statistically significant after FDR correction**.

| Connectivity edge | Female OR | Male OR | Interaction OR | FDR-adjusted p |
|---|---:|---:|---:|---:|
| **Limbic A ↔ Control B** | **0.58** | 1.01 | **0.57** | **0.009** |
| **Somatomotor A ↔ Dorsal Attention B** | **1.46** | 0.96 | **1.53** | **0.044** |

## 9.1 Limbic A ↔ Control B

### Female association

$$
OR = 0.58
$$

Higher connectivity was associated with approximately **42% lower odds of ADHD** in females:

$$
(1-0.58)\times100 \approx 42\%
$$

### Male association

$$
OR = 1.01
$$

There was no meaningful association between this connectivity measure and ADHD in males.

### Interaction

$$
OR_{interaction}=0.57,\quad p_{FDR}=0.009
$$

This provides statistically significant evidence that the connectivity–ADHD association differed by sex.

## 9.2 Somatomotor A ↔ Dorsal Attention B

### Female association

$$
OR = 1.46
$$

Higher connectivity was associated with approximately **46% higher odds of ADHD** in females.

### Male association

$$
OR = 0.96
$$

There was no meaningful association between this connectivity measure and ADHD in males.

### Interaction

$$
OR_{interaction}=1.53,\quad p_{FDR}=0.044
$$

This provides statistically significant evidence that the connectivity–ADHD association differed by sex.

### Figure placeholder

**Figure 3. Sex-specific odds ratios for the two FDR-significant connectivity × sex interactions.**

`![Interaction forest plot](figures/interaction_forest_plot.png)`

---

# 10. Overall Interpretation

The analysis produced a sequential chain of evidence:

```text
Behavioural analysis
        ↓
Associations with ADHD largely shared across sexes
        ↓
Sex-stratified ML
        ↓
Different stable connectivity patterns emerged
        ↓
11 candidate edges
        ↓
Formal Edge × Sex interaction testing
        ↓
2 edges significant after FDR correction
```

The behavioural analyses did not provide strong evidence that ADHD-related behavioural associations differed by sex.

In contrast, the sex-stratified ML analysis identified different stable functional-connectivity patterns. Formal interaction testing subsequently provided statistical evidence that **two connectivity edges had sex-specific associations with ADHD**.

Thus, in this analysis, **sex-related heterogeneity was more evident at the functional-connectivity level than at the behavioural level**.

---

# 11. Exploratory Nature of the Findings

The interaction analysis should be interpreted as **exploratory and hypothesis-generating**.

The connectivity edges were identified using the same dataset in which the interaction analyses were subsequently performed. Therefore, the analysis does not provide independent external confirmation of the identified associations.

FDR correction controls the expected proportion of false discoveries among the tested interactions, but it does not replace independent replication.

> **The two identified sex-specific connectivity associations represent candidate findings that require validation in an independent cohort.**

---

# 12. Limitations

### 12.1 Same-sample feature discovery and testing

Stable edges were identified and tested within the same overall dataset.

**Implication:** The interaction findings may be optimistic and should not be interpreted as independently validated biomarkers.

### 12.2 Unequal subgroup sizes

The male subgroup was larger than the female subgroup:

- Female: 416
- Male: 797

This imbalance may affect statistical power, feature stability and predictive performance.

### 12.3 Site and acquisition variability

Site group and scan location may capture differences in acquisition or cohort characteristics.

These variables were included as adjustment factors but should **not** be interpreted as biological mechanisms.

### 12.4 Exploratory analysis

The interaction analyses were targeted follow-up analyses rather than pre-specified confirmatory tests.

---

# 13. Final Take-Home Message

> **Behavioural associations with ADHD were largely shared across females and males, whereas functional connectivity revealed evidence of sex-specific associations. Sex-stratified machine learning identified candidate connectivity patterns, and formal interaction analysis identified two edges with statistically significant sex-dependent associations after FDR correction. These findings are exploratory and require validation in independent cohorts.**

---

## 14. Presentation Takeaways

### What did we do?

> We analysed ADHD separately in females and males, identified stable connectivity features using leakage-safe cross-validation, and formally tested those features for sex interactions.

### What did we find?

> Behavioural associations were largely similar, but connectivity patterns differed. Of 11 stable candidate edges, two showed significant sex-specific associations after FDR correction.

### Why does it matter?

> It suggests that sex-specific functional connectivity may capture heterogeneity in ADHD that is not apparent from behavioural measures alone.
