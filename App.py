# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from fpdf import FPDF

# --- Configuration ---
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "OLS-Models"
DATA_DIR = BASE_DIR / "Data"

MODELS = {
    "BaS": {
        "model_path": MODEL_DIR / "BaS_furthest_share_ols_model.pkl",
        "data_path": DATA_DIR / "bas_volume_and_furthest_share.csv",
        "title": "BaS: Predicted Furthest Share by Applicant-to-Slot Ratio",
    },
    "CTS": {
        "model_path": MODEL_DIR / "CTS_furthest_share_ols_model.pkl",
        "data_path": DATA_DIR / "CTS_Volume_and_Furthest_Share_Trend.csv",
        "title": "CTS: Predicted Furthest Share by Applicant-to-Slot Ratio",
    },
}


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


def compute_prediction_band(model, log_x_array, alpha=0.05):
    """Compute prediction bands for an array of log(x) values."""
    from scipy.stats import t as t_dist

    X_data = model.model.exog
    n = X_data.shape[0]
    p = X_data.shape[1]
    mse = model.mse_resid
    XtX_inv = np.linalg.inv(X_data.T @ X_data)
    t_crit = t_dist.ppf(1 - alpha / 2, df=n - p)

    y_pred = model.params["const"] + model.params["log_furthest_per_slot"] * log_x_array
    lower = np.empty_like(log_x_array)
    upper = np.empty_like(log_x_array)

    for i, lx in enumerate(log_x_array):
        x_new = np.array([1.0, lx])
        h = x_new @ XtX_inv @ x_new
        se_pred = np.sqrt(mse * (1 + h))
        lower[i] = y_pred[i] - t_crit * se_pred
        upper[i] = y_pred[i] + t_crit * se_pred

    return y_pred, lower, upper


def solve_for_ratio(model, target_share):
    """Solve for x (furthest_per_slot) from the best-fit line."""
    beta0 = model.params["const"]
    beta1 = model.params["log_furthest_per_slot"]
    return np.exp((target_share - beta0) / beta1)


def solve_for_ratio_lower_pi(model, target_share, alpha=0.05):
    """Find x such that the LOWER bound of the 95% prediction interval equals target_share."""
    from scipy.optimize import brentq
    from scipy.stats import t as t_dist

    X_data = model.model.exog
    n = X_data.shape[0]
    p = X_data.shape[1]
    mse = model.mse_resid
    XtX_inv = np.linalg.inv(X_data.T @ X_data)
    t_crit = t_dist.ppf(1 - alpha / 2, df=n - p)

    beta0 = model.params["const"]
    beta1 = model.params["log_furthest_per_slot"]

    def lower_pi_minus_target(log_x):
        x_new = np.array([1.0, log_x])
        y_hat = beta0 + beta1 * log_x
        h = x_new @ XtX_inv @ x_new
        se_pred = np.sqrt(mse * (1 + h))
        lower_bound = y_hat - t_crit * se_pred
        return lower_bound - target_share

    try:
        log_x_solution = brentq(lower_pi_minus_target, -5, 10)
        return np.exp(log_x_solution)
    except ValueError:
        return None


def generate_technical_pdf(model, scholarship, volume):
    """Generate a PDF with technical model details."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"{scholarship} Furthest-From-Opportunity Model: Technical Details", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6,
        "This document describes the OLS regression model used to predict the share of "
        f"selected {scholarship} scholars who are 'furthest from opportunity' (both first-generation "
        "college students AND from families earning at or below 65% of the HUD Median Family Income)."
    )
    pdf.ln(5)

    # Model specification
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Model Specification", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6,
        "Furthest Share (Selected) = B0 + B1 * ln(N Furthest Applicants / N Slots) + error\n\n"
        "The natural log transform is used because the relationship exhibits diminishing returns: "
        "when there are already many furthest applicants per slot, adding more has a smaller "
        "marginal effect on the selected share."
    )
    pdf.ln(5)

    # Fitted parameters
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Fitted Parameters", ln=True)
    pdf.set_font("Helvetica", "", 11)
    beta0 = model.params['const']
    beta1 = model.params['log_furthest_per_slot']
    pdf.cell(0, 6, f"  Intercept (B0): {beta0:.6f}", ln=True)
    pdf.cell(0, 6, f"  Slope (B1): {beta1:.6f}", ln=True)
    pdf.cell(0, 6, f"  R-squared: {model.rsquared:.4f}", ln=True)
    pdf.cell(0, 6, f"  p-value (slope): {model.pvalues['log_furthest_per_slot']:.6f}", ln=True)
    pdf.cell(0, 6, f"  Number of observations: {int(model.nobs)}", ln=True)
    pdf.cell(0, 6, f"  Residual Std Error: {np.sqrt(model.mse_resid):.6f}", ln=True)
    pdf.ln(5)

    # Inverse formula
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Inverse Formula (Target Setting)", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6,
        "To find the number of furthest-from-opportunity applicants needed:\n\n"
        "  N Furthest Needed = N Slots x exp((Target Share - B0) / B1)\n\n"
        f"  N Furthest Needed = N Slots x exp((Target Share - {beta0:.4f}) / {beta1:.4f})"
    )
    pdf.ln(5)

    # Good / Better / Best explanation
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Good / Better / Best Estimates", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6,
        "Good: The number of furthest-from-opportunity applicants needed so that the "
        "best-fit regression line predicts the target share. This is the minimum needed "
        "if the historical relationship holds exactly.\n\n"
        "Best: The number of furthest-from-opportunity applicants needed so that the "
        "lower bound of the 95% prediction interval equals the target share. This provides "
        "a buffer against year-to-year variability. Even in an unfavorable year, you would "
        "still expect to meet the target.\n\n"
        "Better: The midpoint between Good and Best."
    )
    pdf.ln(5)

    # Prediction intervals
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Prediction Intervals", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6,
        "The 95% prediction interval accounts for both estimation uncertainty (how well we "
        "know the regression line) and residual variance (natural year-to-year scatter). "
        "The interval width varies with the predictor value, being narrowest near the center "
        "of the observed data and wider at the extremes."
    )
    pdf.ln(5)

    # Historical data
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Historical Data", ln=True)
    pdf.set_font("Courier", "", 9)
    pdf.cell(0, 5, f"{'Year':<12}{'Applicants':>11}{'Selected':>10}{'Furthest':>10}{'Share':>9}{'Ratio':>8}", ln=True)
    pdf.cell(0, 5, "-" * 60, ln=True)
    for _, row in volume.iterrows():
        yr = str(row['Application_Academic_Year'])
        pdf.cell(0, 5,
            f"{yr:<12}{int(row['n_applicants']):>11,}{int(row['n_selected']):>10,}"
            f"{int(row['n_furthest']):>10,}{row['furthest_share']:>8.1%}{row['furthest_per_slot']:>8.2f}",
            ln=True
        )
    pdf.ln(5)

    # Caveats
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Caveats", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6,
        f"- The model is fit on {int(model.nobs)} observations (one per application year). "
        "Standard errors should be interpreted cautiously.\n"
        "- The model assumes the program's selection preference for furthest scholars "
        "remains approximately stable over time.\n"
        "- Extreme target values (near 0% or 100%) produce unreliable predictions.\n"
        "- The applicant pool composition is exogenous. The model indicates how many "
        "furthest applicants are needed but does not control whether they apply."
    )

    pdf_path = BASE_DIR / "Documentation" / f"{scholarship}_Technical_Details.pdf"
    pdf.output(str(pdf_path))
    return pdf_path


# --- Streamlit App ---
st.set_page_config(page_title="Furthest-From-Opportunity Target Setting", layout="wide")
st.title("Furthest-From-Opportunity Target Setting Tool")
st.markdown(
    """
    This tool helps determine **how many eligible furthest-from-opportunity applicants** 
    are needed in the pipeline to achieve a target share among selected scholars.
    """
)

# --- Inputs ---
col1, col2, col3 = st.columns(3)

with col1:
    scholarship = st.selectbox("Scholarship", options=["BaS", "CTS"])

with col2:
    slots = st.number_input("Slots", min_value=1, step=1, value=750)

with col3:
    target_pct = st.number_input(
        "Furthest From Opportunity Share (%)", min_value=1, max_value=100, step=1, value=55
    )

target_share = target_pct / 100.0

# --- Load model and data ---
config = MODELS[scholarship]
model = load_model(config["model_path"])
volume = load_data(config["data_path"])

# --- Compute Good / Better / Best ---
ratio_good = solve_for_ratio(model, target_share)
n_good = ratio_good * slots

ratio_best = solve_for_ratio_lower_pi(model, target_share)
if ratio_best is not None:
    n_best = ratio_best * slots
else:
    n_best = None

if n_best is not None:
    n_better = (n_good + n_best) / 2.0
    ratio_better = n_better / slots
else:
    n_better = None
    ratio_better = None

# --- Display Results ---
st.markdown("---")
st.subheader("Number of Eligible Furthest-From-Opportunity Applicants Needed")

res_col1, res_col2, res_col3 = st.columns(3)
with res_col1:
    st.metric("Good", f"{int(round(n_good)):,}")
    st.caption("Expected outcome matches target")
with res_col2:
    if n_better is not None:
        st.metric("Better", f"{int(round(n_better)):,}")
        st.caption("Midpoint between Good and Best")
    else:
        st.metric("Better", "N/A")
with res_col3:
    if n_best is not None:
        st.metric("Best", f"{int(round(n_best)):,}")
        st.caption("Target met even in an unfavorable year")
    else:
        st.metric("Best", "N/A")

# --- Plot ---
st.markdown("---")
st.subheader("Historical Fit and Prediction")

fig, ax = plt.subplots(figsize=(10, 6))

# Historical scatter
ax.scatter(
    volume["furthest_per_slot"],
    volume["furthest_share"],
    s=80,
    zorder=5,
    color="#005C5D",
    label="Historical Years",
)

# Prediction range
x_min = volume["furthest_per_slot"].min() * 0.7
x_max = volume["furthest_per_slot"].max() * 1.3
if ratio_best is not None:
    x_max = max(x_max, ratio_best * 1.1)
x_max = max(x_max, ratio_good * 1.1)

x_range = np.linspace(x_min, x_max, 200)
log_x_range = np.log(x_range)

# Compute prediction band
y_pred, y_lower, y_upper = compute_prediction_band(model, log_x_range)

# Plot fitted curve
ax.plot(
    x_range,
    y_pred,
    "r--",
    linewidth=2,
    label=f"Model Fit (R\u00b2 = {model.rsquared:.3f})",
)

# Plot prediction interval band
ax.fill_between(
    x_range, y_lower, y_upper, alpha=0.12, color="red", label="95% Prediction Range"
)

# Horizontal reference line at target
ax.axhline(target_share, color="gray", linestyle="--", alpha=0.6, label=f"Target: {target_pct}%")

# Highlight the Good estimate (triangle)
ax.scatter(
    [ratio_good], [target_share], s=140, color="#005C5D", zorder=10,
    marker="^", edgecolors="black", linewidths=0.8
)
ax.annotate(
    f"Good: {int(round(n_good)):,}",
    (ratio_good, target_share),
    textcoords="offset points",
    xytext=(10, 8),
    fontsize=9,
    fontweight="bold",
    color="#005C5D",
)

# Highlight the Better estimate (square)
if ratio_better is not None:
    y_better_pred = model.params["const"] + model.params["log_furthest_per_slot"] * np.log(ratio_better)
    ax.scatter(
        [ratio_better], [y_better_pred], s=140, color="#005C5D", zorder=10,
        marker="s", edgecolors="black", linewidths=0.8
    )
    ax.annotate(
        f"Better: {int(round(n_better)):,}",
        (ratio_better, y_better_pred),
        textcoords="offset points",
        xytext=(10, -15),
        fontsize=9,
        fontweight="bold",
        color="#005C5D",
    )

# Highlight the Best estimate (diamond) with vertical line to target
if ratio_best is not None:
    y_best_pred = model.params["const"] + model.params["log_furthest_per_slot"] * np.log(ratio_best)
    ax.scatter(
        [ratio_best], [y_best_pred], s=140, color="#005C5D", zorder=10,
        marker="D", edgecolors="black", linewidths=0.8
    )
    ax.annotate(
        f"Best: {int(round(n_best)):,}",
        (ratio_best, y_best_pred),
        textcoords="offset points",
        xytext=(10, 8),
        fontsize=9,
        fontweight="bold",
        color="#005C5D",
    )
    # Vertical dashed line from Best point down to the target line
    ax.plot(
        [ratio_best, ratio_best], [target_share, y_best_pred],
        linestyle="--", color="#005C5D", linewidth=1.2, alpha=0.7
    )

ax.set_xlabel("# Furthest From Opportunity Applicants / # Slots")
ax.set_ylabel("Share of Selected Scholars Who Are Furthest")
ax.set_title(config["title"])
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

st.pyplot(fig)

# --- Explanation ---
with st.expander("How are Good / Better / Best calculated?"):
    st.markdown(
        f"""
    **Good** \u2014 The number of furthest-from-opportunity applicants needed so that the 
    model's expected outcome matches the target share ({target_pct}%). This is the 
    minimum needed if the historical pattern holds exactly.
    
    **Best** \u2014 The number of furthest-from-opportunity applicants needed so that 
    even in an unfavorable year (the bottom of the 95% prediction range), you would 
    still expect to hit the target ({target_pct}%).
    
    **Better** \u2014 The midpoint between Good and Best.
    """
    )

# --- Technical PDF ---
st.markdown("---")
pdf_path = generate_technical_pdf(model, scholarship, volume)
with open(pdf_path, "rb") as pdf_file:
    st.download_button(
        label="Download Technical Details (PDF)",
        data=pdf_file,
        file_name=f"{scholarship}_Technical_Details.pdf",
        mime="application/pdf",
    )
