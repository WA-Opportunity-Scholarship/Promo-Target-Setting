# CTS Furthest-From-Opportunity Regression: Write-Up

## What Is This?

This document summarizes a regression model that predicts the **share of selected CTS scholars who are "furthest from opportunity"** — defined as both first-generation college students AND from families earning ≤65% of the HUD Median Family Income.

The model is designed to support annual target-setting by answering: *Given a target percentage and a known number of scholarship slots, how many furthest-from-opportunity applicants need to be in the pipeline?*

---

## The Pattern

The CTS furthest-from-opportunity share has varied substantially across application years:

| Year | Furthest Share |
|------|---------------|
| 2019-20 | 71.9% |
| 2020-21 | 67.5% |
| 2021-22 | 65.3% |
| 2022-23 | 52.9% |
| 2023-24 | 52.2% |
| 2024-25 | 56.8% |
| 2025-26 | 60.9% |

**Mean: 61.0% | Range: 52.2% – 71.9%**

The share declined steadily from 2019-20 to 2023-24, then partially rebounded. This creates a challenge for target-setting: what's achievable depends on factors outside the selection committee's control.

---

## What Drives the Variation?

The exploratory analysis tested several potential drivers:

1. **Applicant pool composition** — The share of furthest-from-opportunity students in the full applicant pool tracks closely with the selected share. The decline is not purely a selection decision; it reflects who is applying.

2. **First-generation status** — The first-gen share among both applicants and selected scholars has declined over time, contributing to the downward trend.

3. **Low-income share (≤65% MFI)** — The low-income share has also declined, reflecting rising applicant incomes that outpace the MFI threshold.

4. **Selection preference** — CTS consistently selects furthest scholars at a higher rate than non-furthest scholars (lift > 1), but this preference cannot overcome a declining applicant pool.

5. **The key predictor: Furthest applicants per slot** — The ratio of furthest-from-opportunity applicants to available scholarship slots explains most of the year-to-year variation.

---

## The Model

### Specification

$$\text{Furthest Share} = 0.6556 + 0.2312 \cdot \ln\!\left(\frac{\text{N Furthest Applicants}}{\text{N Slots}}\right)$$

### Fit Statistics

| Metric | Value |
|--------|-------|
| R² | 0.895 |
| p-value (slope) | 0.0013 |
| N (observations) | 7 years |

The model explains **~90% of the variance** in the furthest share. The log transform captures the diminishing-returns relationship: going from 0.5 to 1.0 furthest applicants per slot has a much larger effect than going from 1.5 to 2.0.

### Scatter Plot

![CTS Regression: Furthest Share vs. Furthest Applicants per Slot](figures/CTS_regression_furthest_per_slot.png)

Each point represents one application academic year. The red dashed curve is the fitted log-linear model.

---

## How to Use This for Target-Setting

### The Inverse Formula

To find the number of furthest-from-opportunity applicants needed to achieve a target:

$$\text{N Furthest Needed} = \text{N Slots} \times \exp\!\left(\frac{\text{Target Share} - 0.6556}{0.2312}\right)$$

### Examples

| Slots | Target | Furthest Applicants Needed |
|-------|--------|---------------------------|
| 500 | 55% | ~330 |
| 500 | 60% | ~393 |
| 500 | 65% | ~468 |
| 750 | 55% | ~495 |
| 750 | 60% | ~589 |
| 750 | 65% | ~703 |

### Streamlit Tool

The Streamlit application will allow your colleague to:
1. Input the **number of slots** (how many CTS scholars will be selected)
2. Input the **target furthest share** (the Board's goal)
3. Receive the **number of furthest-from-opportunity applicants** needed in the pipeline

---

## Caveats

- **Small sample size** — The model is fit on 7 observations (one per year). While the fit is strong (R² = 0.90), confidence intervals are wide.
- **Assumes stable selection preference** — The model assumes the program will continue to preferentially select furthest scholars at roughly the same rate.
- **Extreme targets** — Targets below ~50% or above ~75% are outside the range of observed data and extrapolated predictions are unreliable.
- **Applicant pool is exogenous** — The model tells you how many furthest applicants you *need*, but does not control whether they actually apply. Recruitment strategy matters.

---

## Files

| File | Location |
|------|----------|
| Pickled model | `Output/CTS_furthest_share_ols_model.pkl` |
| Regression figure | `Suporting Material/figures/CTS_regression_furthest_per_slot.png` |
| Exploration notebook | `Suporting Material/Board_Indicator_Selection_Furthest_CTS_Exploration.ipynb` |
| Summary data | `Output/CTS_furthest_oscillation_summary.csv` |
