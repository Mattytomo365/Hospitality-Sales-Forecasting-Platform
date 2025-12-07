import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.dataset.load_save import load_csv # avoiding fragile relative imports
'''
Data visualisations supporting points raised in report
'''
# helper function to improve visibility of values on x-axis
def monthly_labels(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.figure.autofmt_xdate()


# plots basis waves at varying harmonic levels visualising fourier features against time
def fourier_basis_wave(df, sin_col, cos_col, k, title):
    d = df["date"]
    s = df[sin_col].astype(float)
    c = df[cos_col].astype(float)
    fig, ax = plt.subplots()

    ax.plot(d, s, label=f"sin k={k}")
    ax.plot(d, c, label=f"cos k={k}")

    ax.set_xlabel("date")
    ax.set_ylabel("basis value")
    if title: ax.set_title(title)
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)
    monthly_labels(ax)

    return fig, ax

# plots linear combination of baseline waves, visualises what linear models perform on yearly fourier features to understand seasonal shape
def seasonal_curve(df, target_col_name, k, title):
    d = df["date"]
    y = df[target_col_name].astype("float").values # the series which the seasonal shape is being visualised from
    if k == 2:
        s = df["doy_sin_2"].astype(float).values # Fourier features previously generated
        c = df["doy_cos_2"].astype(float).values
    else:
        s = df["doy_sin"].astype(float).values
        c = df["doy_cos"].astype(float).values

    # build a 2-D array of the features to make a small design matrix
    X = np.column_stack([np.ones_like(s), s, c]) # [1, sin, cos] so X has shape (num_days, 3)

    # find coefficients that minimise total squared error
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)    # beta = [β₀, a, b]

    # make a seasonal curve from calculated weights/coefficients
    y_fitted = X @ beta     # β₀ + a*sin + b*cos

    # plot actual vs fitted baseline
    fig, ax = plt.subplots()
    ax.plot(d, y, label="actual")
    ax.plot(d, y_fitted, "--", label="seasonal curve (from sin/cos)")
    if title: ax.set_title(title)
    ax.set_ylabel(target_col_name)
    ax.set_xlabel("date")
    ax.legend()
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)
    monthly_labels(ax)

    return fig, ax

# partial dependence plot to show what tree-based models perofrm on fourier features?

# generates unit-circle plots of fourier features, helping visualise the circular pattern
def fourier_unit_cirle(df, cos_col, sin_col, title):
    fig, ax = plt.subplots()
    df.plot.scatter(sin_col, cos_col, ax=ax)
    ax.set_aspect("equal", adjustable="box") # forms circle
    t = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(t), np.sin(t), linewidth=1) # unit-circle line
    if title: ax.set_title(title)
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)
    # group and highlight weekday/monthly means?
    return fig, ax


# power spectral density

# centralised function
def plot_all(df):

    # fourier baseline waves
    fourier_basis_wave(df, "dow_sin", "dow_cos", 1, "Weekly Fourier basis")
    fourier_basis_wave(df, "doy_sin", "doy_cos", 1, "Yearly Fourier basis")
    fourier_basis_wave(df, "doy_sin_2", "doy_cos_2", 2, "Yearly Fourier basis") # k=2 adds an extra Fourier harmonic so the model can capture more complex seasonality

    # seasonal curves
    seasonal_curve(df, "sales", 2, "Sales seasonal curve") # plot k=1?
    seasonal_curve(df, "covers", 2, "Covers seasonal curve")

    # fourier unit-circle plots
    fourier_unit_cirle(df, "dow_cos", "dow_sin", "Cyclical day-of-week scatter plot")
    fourier_unit_cirle(df, "doy_cos", "doy_sin", "Cyclical day-of-year scatter plot")
    plt.show()


if __name__ == "__main__":
    df = load_csv("data/df_feature.csv")
    plot_all(df)