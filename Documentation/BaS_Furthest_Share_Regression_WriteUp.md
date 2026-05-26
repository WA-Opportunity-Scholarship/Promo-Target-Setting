# Regression Model: Predicting the Furthest-From-Opportunity Share

## What This Model Does

This OLS regression predicts the **share of selected BaS scholars who are "furthest from opportunity"** based on the ratio of furthest-from-opportunity applicants to available scholarship slots.

It answers the question: *Given a target percentage of furthest scholars among those selected, how many furthest-from-opportunity applicants need to apply?*

---

## Model Definition

$$
\text{Furthest Share (Selected)} = \beta_0 + \beta_1 \cdot \ln\!\left(\frac{\text{N Furthest Applicants}}{\text{N Slots}}\right) + \varepsilon
$$

| Parameter | Description |
|-----------|-------------|
| **Dependent variable (y)** | Share of selected scholars who are both first-generation *and* below 65% of Median Family Income |
| **Independent variable (x)** | Number of furthest-from-opportunity applicants divided by the number of scholarship slots |
| **Log transform** | The natural log of x is used because the relationship exhibits diminishing returns — doubling the ratio from 0.3→0.6 has a larger effect than from 0.9→1.2 |

### Why Log-Linear?

The relationship between applicant supply and selected composition is not linear. When slots are scarce relative to furthest applicants (high ratio), the program can be highly selective and the furthest share saturates. A log transform captures this diminishing-returns pattern and produces a better fit than a linear model.

---

## Fitted Model Results

![Regression Scatterplot](figures/regression_furthest_per_slot.png)

The scatter plot above shows each application academic year as a point. The red dashed curve is the fitted log-linear model. Years are labeled for reference.

---

## Why This Matters for Target Setting

The Board sets an annual target for the percentage of selected scholars who are "furthest from opportunity." Achieving that target depends on how many furthest-from-opportunity students apply relative to the number of available slots.

**The key insight:** The furthest share among selected scholars is not purely a selection decision — it is constrained by the composition of the applicant pool. If the program wants 55% of selected scholars to be furthest-from-opportunity and has 750 slots, the model tells us approximately how many furthest applicants need to be in the pipeline.

---

## Inverse Formula (for Target Setting)

To find the number of furthest-from-opportunity applicants needed:

$$
\text{N Furthest Needed} = \text{N Slots} \times \exp\!\left(\frac{\text{Target Share} - \beta_0}{\beta_1}\right)
$$

### Example

> If the target is **55% furthest share** and there are **750 slots**:
>
> $$\text{N Furthest Needed} = 750 \times \exp\!\left(\frac{0.55 - \beta_0}{\beta_1}\right)$$

The Streamlit tool automates this calculation. The user inputs:
1. **Number of slots** (how many scholars will be selected)
2. **Target furthest share** (the Board's goal, e.g., 55%)

And the tool returns:
- **Number of furthest-from-opportunity applicants needed** in the pipeline

---

## Caveats

- The model is fit on a small number of observations (one per application year). Standard errors should be interpreted cautiously.
- The model assumes the program's selection preference for furthest scholars remains approximately stable over time.
- Extreme values of the target share (near 0% or 100%) will produce unreliable predictions.
- The model file is stored at: `Output/furthest_share_ols_model.pkl`

---

## File References

| File | Description |
|------|-------------|
| `Output/furthest_share_ols_model.pkl` | Pickled OLS model object (statsmodels) |
| `Suporting Material/figures/regression_furthest_per_slot.png` | Regression scatter plot |
| `Suporting Material/Board_Indicator_Selection_Furthest_BaS_Exploration.ipynb` | Full exploratory analysis notebook |
