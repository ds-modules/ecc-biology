"""Backend data loading and cleaning utilities for wastewater notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

DATA_FILENAME = "biobot_kits_quant_data_duvalletetal2022.csv"
CLEANED_FILENAME = "cleaned_wastewater_covid_data.csv"


def _resolve_data_path(filename: str = DATA_FILENAME) -> Path:
    """Find a data file from common notebook working directories."""
    candidates = [
        Path(filename),
        Path("SARS-CoV2") / filename,
        Path(__file__).resolve().parent / filename,
        Path.cwd() / filename,
        Path.cwd() / "SARS-CoV2" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename}. Checked: {', '.join(str(p) for p in candidates)}"
    )


def load_and_clean_wastewater_data(
    save_cleaned: bool = True,
    cleaned_filename: str = CLEANED_FILENAME,
) -> tuple[pd.DataFrame, Path, Path | None]:
    """Load raw kit data and return cleaned dataframe used in the notebook."""
    raw_path = _resolve_data_path(DATA_FILENAME)
    raw = pd.read_csv(raw_path)

    keep_cols = [
        "sampling_date",
        "normalized_conc_copies_per_L",
        "pmmv_copies_per_mL_kit_raw",
        "sarscov2_copies_per_mL_kit_raw",
        "rolling_average_new_cases_centered",
        "fig1_label",
    ]

    df = raw[keep_cols].copy()
    df["sampling_date"] = pd.to_datetime(df["sampling_date"], errors="coerce")
    df = (
        df.dropna(subset=["sampling_date"])
        .sort_values(["fig1_label", "sampling_date"])
        .reset_index(drop=True)
    )

    df["sars_to_pmmv_ratio"] = (
        df["sarscov2_copies_per_mL_kit_raw"] / df["pmmv_copies_per_mL_kit_raw"]
    )
    df = df.replace([np.inf, -np.inf], np.nan)

    cleaned_path = None
    if save_cleaned:
        cleaned_path = raw_path.parent / cleaned_filename
        df.to_csv(cleaned_path, index=False)

    return df, raw_path, cleaned_path


def show_column_blurb_widget() -> None:
    """Render a column-description dropdown widget for students."""
    column_blurbs = {
        "sampling_date": "The date when the wastewater sample was collected.",
        "normalized_conc_copies_per_L": "A normalized estimate of SARS-CoV-2 concentration in wastewater, scaled per liter. This is one of the main wastewater signals used in this notebook.",
        "pmmv_copies_per_mL_kit_raw": "A PMMoV control measurement. PMMoV is often used as a background/control signal because it is commonly found in wastewater.",
        "sarscov2_copies_per_mL_kit_raw": "The raw SARS-CoV-2 signal measured in the wastewater sample before normalization.",
        "rolling_average_new_cases_centered": "A centered rolling average of new reported COVID-19 cases. This smooths out short-term spikes so the overall pattern is easier to see.",
        "fig1_label": "A site or group label used in the dataset to separate different wastewater systems or locations.",
        "sars_to_pmmv_ratio": "An optional helper feature created in this notebook: raw SARS-CoV-2 signal divided by the PMMoV control signal.",
    }

    column_dropdown = widgets.Dropdown(
        options=list(column_blurbs.keys()),
        value="sampling_date",
        description="Column:",
        layout=widgets.Layout(width="60%"),
    )

    column_output = widgets.Output()

    def show_column_blurb(change=None):
        with column_output:
            column_output.clear_output()
            key = column_dropdown.value
            display(Markdown(f"**{key}**  \n{column_blurbs[key]}"))

    column_dropdown.observe(show_column_blurb, names="value")
    show_column_blurb()
    display(column_dropdown, column_output)


def plot_overlay_for_label(df: pd.DataFrame, label: str) -> None:
    """Plot normalized wastewater and case trend on twin y-axes for one label."""
    site_df = df[df["fig1_label"] == label].copy().sort_values("sampling_date")
    if site_df.empty:
        print(f"No rows found for label {label}.")
        return

    fig, ax1 = plt.subplots(figsize=(11, 5))
    line1 = ax1.plot(
        site_df["sampling_date"],
        site_df["normalized_conc_copies_per_L"],
        color="steelblue",
        label="Normalized wastewater concentration",
    )
    ax1.set_xlabel("Sampling date")
    ax1.set_ylabel("Normalized concentration (copies/L)")

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        site_df["sampling_date"],
        site_df["rolling_average_new_cases_centered"],
        color="darkorange",
        label="Rolling average of new cases",
    )
    ax2.set_ylabel("Rolling average of new cases")

    # Merge handles from both axes so legend includes both lines.
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.suptitle(f"Overlay view: wastewater signal and reported cases for {label}")
    fig.autofmt_xdate()
    plt.show()


def prepare_regression_model(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, float, float]:
    """Fit a simple linear model: cases ~ normalized wastewater concentration."""
    model_df = df[
        ["normalized_conc_copies_per_L", "rolling_average_new_cases_centered"]
    ].dropna().copy()

    x = model_df["normalized_conc_copies_per_L"].to_numpy()
    y = model_df["rolling_average_new_cases_centered"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return model_df, x, y, slope, intercept, r2


def explain_regression_for_students(slope: float, intercept: float, r2: float) -> None:
    """Print equation and plain-language interpretation for non-data-science students."""
    explanation = f"""
### Regression Summary

**Model equation**

`predicted_cases = {slope:.6f} * normalized_conc_copies_per_L + {intercept:.3f}`

**Model fit**

`R^2 = {r2:.3f}`

### How to Read This

- **Best-fit line:** This line summarizes the overall trend between wastewater and case counts.
- **Slope ({slope:.6f}):** For each 1-unit increase in normalized wastewater concentration, predicted cases change by about `{slope:.6f}` on average.
- **Intercept ({intercept:.3f}):** This is the baseline model prediction when wastewater concentration is 0.
- **R^2 ({r2:.3f}):** This is how much variation in case counts the line explains (`0` = very weak, `1` = very strong).
"""
    display(Markdown(explanation))


def plot_regression_fit(model_df: pd.DataFrame, slope: float, intercept: float) -> None:
    """Plot scatter points and best-fit regression line."""
    plt.figure(figsize=(10, 5))
    plt.scatter(
        model_df["normalized_conc_copies_per_L"],
        model_df["rolling_average_new_cases_centered"],
        alpha=0.6,
        color="steelblue",
        label="Observed data points",
    )

    x_line = np.linspace(
        model_df["normalized_conc_copies_per_L"].min(),
        model_df["normalized_conc_copies_per_L"].max(),
        200,
    )
    y_line = slope * x_line + intercept

    plt.plot(x_line, y_line, color="darkorange", linewidth=2, label="Best-fit regression line")
    plt.xlabel("Normalized concentration (copies/L)")
    plt.ylabel("Rolling average of new cases")
    plt.title("Regression: wastewater concentration vs. reported cases")
    plt.legend()
    plt.tight_layout()
    plt.show()


def explain_regression_plot_for_students() -> None:
    """Show plain-language explanation of the regression graph."""
    text = """
### How to Read This Graph

- **Blue dots:** each dot is one real observation from the dataset (wastewater level + reported cases).
- **Orange line:** this is the model's best-fit line showing the overall trend.
- **Upward slope:** if the line slopes up, higher wastewater values tend to go with higher reported case counts.
- **Distance from line:** dots far from the line show real-world noise and other factors not captured by this simple model.
"""
    display(Markdown(text))


def show_interactive_prediction_widget(
    df: pd.DataFrame, model_df: pd.DataFrame, slope: float, intercept: float
) -> None:
    """Render interactive sliders and dynamic regression prediction plot."""
    norm_min = float(np.nanpercentile(df["normalized_conc_copies_per_L"], 5))
    norm_max = float(np.nanpercentile(df["normalized_conc_copies_per_L"], 95))
    raw_min = float(np.nanpercentile(df["sarscov2_copies_per_mL_kit_raw"], 5))
    raw_max = float(np.nanpercentile(df["sarscov2_copies_per_mL_kit_raw"], 95))

    median_conversion = np.nanmedian(
        df["normalized_conc_copies_per_L"] / df["sarscov2_copies_per_mL_kit_raw"]
    )

    norm_slider = widgets.FloatSlider(
        value=float(np.nanmedian(df["normalized_conc_copies_per_L"])),
        min=norm_min,
        max=norm_max,
        step=(norm_max - norm_min) / 100,
        description="Normalized:",
        readout_format=".1f",
        layout=widgets.Layout(width="85%"),
    )

    raw_slider = widgets.FloatSlider(
        value=float(np.nanmedian(df["sarscov2_copies_per_mL_kit_raw"])),
        min=raw_min,
        max=raw_max,
        step=(raw_max - raw_min) / 100,
        description="Raw SARS:",
        readout_format=".1f",
        layout=widgets.Layout(width="85%"),
    )

    use_raw_toggle = widgets.Checkbox(
        value=False,
        description="Use raw SARS slider instead of normalized slider",
    )

    prediction_output = widgets.Output()

    def update_prediction(change=None):
        with prediction_output:
            prediction_output.clear_output()

            if use_raw_toggle.value:
                x_input = raw_slider.value * median_conversion
                source_text = (
                    "raw SARS value converted to an approximate normalized concentration"
                )
            else:
                x_input = norm_slider.value
                source_text = "normalized concentration slider"

            y_pred = slope * x_input + intercept

            print(f"Input source: {source_text}")
            print(f"Model input (x): {x_input:.2f}")
            print(f"Predicted rolling-average cases (y): {y_pred:.2f}")

            plt.figure(figsize=(10, 5))
            plt.scatter(
                model_df["normalized_conc_copies_per_L"],
                model_df["rolling_average_new_cases_centered"],
                alpha=0.35,
                color="steelblue",
                label="Observed points",
            )
            x_line = np.linspace(
                model_df["normalized_conc_copies_per_L"].min(),
                model_df["normalized_conc_copies_per_L"].max(),
                200,
            )
            y_line = slope * x_line + intercept
            plt.plot(
                x_line,
                y_line,
                color="darkorange",
                linewidth=2,
                label="Best-fit line",
            )
            plt.scatter(
                [x_input],
                [y_pred],
                color="crimson",
                s=110,
                zorder=5,
                label="Your selected prediction",
            )

            plt.axvline(
                x=x_input,
                color="purple",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )
            plt.axhline(
                y=y_pred,
                color="green",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )
            plt.annotate(
                f"(x={x_input:.1f}, y={y_pred:.1f})",
                (x_input, y_pred),
                textcoords="offset points",
                xytext=(10, 10),
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8},
            )

            x_span = x_line.max() - x_line.min()
            local_half_window = max(0.08 * x_span, 1.0)
            x_left = max(x_line.min(), x_input - local_half_window)
            x_right = min(x_line.max(), x_input + local_half_window)
            plt.xlim(x_left, x_right)

            plt.xlabel("Normalized concentration (copies/L)")
            plt.ylabel("Rolling average of new cases")
            plt.title("Interactive prediction on the regression line")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()

    for widget in [norm_slider, raw_slider, use_raw_toggle]:
        widget.observe(update_prediction, names="value")

    display(norm_slider, raw_slider, use_raw_toggle, prediction_output)
    update_prediction()
