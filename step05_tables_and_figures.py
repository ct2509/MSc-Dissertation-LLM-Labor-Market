"""Step 05: Dissertation tables, figures and final reporting outputs.

This script DOES NOT estimate new regressions.

It reads the balanced occupation-year panel plus final outputs from Steps 2, 3 and 4,
then produces:

1. Descriptive statistics for Exposure, Complementarity and M.
2. The preferred LE / HELC / HEHC classification.
3. An Exposure–Complementarity occupation map.
4. Representative occupations in each category.
5. A figure showing why M is closely related to Exposure.
6. Publication-ready main-result and joint-test tables.
7. Appendix-ready pre-trend and M-robustness tables.
8. Alternative-measure robustness tables from Step 4.
9. A basic-FE versus SOC-year specification-sensitivity table from Step 2.
10. A manifest explaining where each output belongs in the dissertation.

Keep this script in the same folder as the existing Step-2, Step-3 and
Step-4 outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 0. Settings
# -----------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent

PANEL_FILE = BASE / "out_occupation_year_panel_balanced.csv"
OCCUPATION_FILE = BASE / "occupation_master_acs.csv"

MAIN_RESULTS_FILE = BASE / "out_step3_dissertation_main_table.csv"
JOINT_TESTS_FILE = BASE / "out_step3_joint_significance_tests.csv"
M_RESULTS_FILE = BASE / "out_step3_M_robustness_results.csv"
PRETREND_FILE = BASE / "out_step2_pretrend_tests.csv"
SPEC_SENSITIVITY_FILE = BASE / "out_step2_continuous_models.csv"
ALT_ROBUST_POOLED_FILE = BASE / "out_step4_robustness_pooled.csv"
ALT_ROBUST_DYNAMIC_FILE = BASE / "out_step4_robustness_dynamic.csv"
ALT_ROBUST_JOINT_FILE = BASE / "out_step4_robustness_joint_tests.csv"
ALT_ROBUST_SUMMARY_FILE = BASE / "out_step4_robustness_summary.csv"

REFERENCE_YEAR = 2022
TOP_OCCUPATIONS_PER_CATEGORY = 8

CATEGORY_ORDER = ["LE", "HELC", "HEHC"]

CATEGORY_LABELS = {
    "LE": "Low exposure",
    "HELC": "High exposure, low complementarity",
    "HEHC": "High exposure, high complementarity",
}

# Exact occupation titles used as labels in the main descriptive figure.
# These are chosen by employment size within the three categories.
FIGURE_LABEL_TITLES = [
    "Driver/sales workers and truck drivers",
    "Laborers and freight, stock, and material movers, hand",
    "Customer service representatives",
    "Cashiers",
    "Registered nurses",
    "Other managers",
]

# Shorter, line-broken labels for the figure.
FIGURE_LABEL_TEXT = {
    "Driver/sales workers and truck drivers":
        "Driver/sales workers\nand truck drivers",
    "Laborers and freight, stock, and material movers, hand":
        "Freight and stock\nmaterial movers",
    "Customer service representatives":
        "Customer service\nrepresentatives",
    "Cashiers":
        "Cashiers",
    "Registered nurses":
        "Registered nurses",
    "Other managers":
        "Other managers",
}

# Manual label positions prevent the large-occupation labels from overlapping.
FIGURE_LABEL_OFFSETS = {
    "Driver/sales workers and truck drivers": (12, 18),
    "Laborers and freight, stock, and material movers, hand": (-145, -18),
    "Customer service representatives": (10, 0),
    "Cashiers": (10, 8),
    "Registered nurses": (10, 10),
    "Other managers": (10, -8),
}

SPECIFICATION_LABELS = {
    "exposure_z_post": "Exposure × Post",
    "complementarity_z_post": "Complementarity × Post",
    "ec_interaction_post": "Exposure × Complementarity × Post",
    "exposure_z_y2023": "Exposure × 2023",
    "exposure_z_y2024": "Exposure × 2024",
    "complementarity_z_y2023": "Complementarity × 2023",
    "complementarity_z_y2024": "Complementarity × 2024",
    "ec_interaction_y2023": "Exposure × Complementarity × 2023",
    "ec_interaction_y2024": "Exposure × Complementarity × 2024",
    "high_c_post": "HEHC − HELC: Post",
    "high_c_y2023": "HEHC − HELC: 2023",
    "high_c_y2024": "HEHC − HELC: 2024",
}

ANALYSIS_LABELS = {
    "continuous_pooled": "Pooled 2023–2024",
    "continuous_dynamic": "Year-specific estimates",
    "exposure_adjusted_HEHC_vs_HELC":
        "High-exposure occupation comparison",
}

ROBUSTNESS_DISPLAY_ORDER = [
    "model_rated_exposure",
    "no_jobzone_complementarity",
    "oews_weighted_mapping",
]


def out(name: str) -> Path:
    """Return a Step-5 output path in the same folder as this script."""

    return BASE / f"out_step5_{name}"


def require_files(paths: Iterable[Path]) -> None:
    """Raise a clear error if any required file is missing."""

    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files in the same folder:\n  - "
            + "\n  - ".join(missing)
        )


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """Calculate a weighted mean after removing invalid observations."""

    valid = (
        values.notna()
        & weights.notna()
        & (pd.to_numeric(weights, errors="coerce") > 0)
    )
    if not valid.any():
        return np.nan

    return float(
        np.average(
            pd.to_numeric(values.loc[valid], errors="coerce"),
            weights=pd.to_numeric(
                weights.loc[valid],
                errors="coerce",
            ),
        )
    )


def significance_stars(p_value: float) -> str:
    """Return conventional significance stars."""

    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def format_estimate(
    coefficient: float,
    standard_error: float,
    p_value: float,
) -> str:
    """Format a coefficient with significance stars and standard error."""

    stars = significance_stars(p_value)
    return f"{coefficient:.4f}{stars} ({standard_error:.4f})"


# -----------------------------------------------------------------------------
# 1. Load data and reproduce the preferred category definition
# -----------------------------------------------------------------------------

def load_panel() -> pd.DataFrame:
    """Load and validate the balanced occupation-year panel."""

    require_files([PANEL_FILE])

    panel = pd.read_csv(
        PANEL_FILE,
        dtype={"acs_occ_code": "string"},
        low_memory=False,
    )

    required = {
        "acs_occ_code",
        "year",
        "soc_major",
        "llm_exposure_main",
        "complementarity_theta",
        "m_index",
        "employment_weight",
        "employment_share",
        "mean_real_wage_2024",
        "wage_weight",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise KeyError(
            "The balanced occupation-year panel is missing: "
            + str(missing)
        )

    panel = panel.copy()
    panel["year"] = pd.to_numeric(
        panel["year"],
        errors="raise",
    ).astype(int)

    # Add occupation titles if an older panel does not contain them.
    if "acs_occ_title" not in panel.columns:
        require_files([OCCUPATION_FILE])

        occupation_titles = pd.read_csv(
            OCCUPATION_FILE,
            usecols=["acs_occ_code", "acs_occ_title"],
            dtype={"acs_occ_code": "string"},
            low_memory=False,
        ).drop_duplicates("acs_occ_code")

        panel = panel.merge(
            occupation_titles,
            on="acs_occ_code",
            how="left",
            validate="many_to_one",
        )

    if panel["acs_occ_title"].isna().any():
        missing_titles = int(panel["acs_occ_title"].isna().sum())
        raise ValueError(
            f"{missing_titles} panel rows are missing occupation titles."
        )

    return panel


def construct_categories(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the preferred LE, HELC and HEHC categories.

    Rule:
    1. Occupations below the overall exposure median are LE.
    2. Among high-exposure occupations, those below the within-group
       complementarity median are HELC.
    3. The remaining high-exposure occupations are HEHC.
    """

    baseline = (
        panel.loc[panel["year"].eq(REFERENCE_YEAR)]
        .copy()
        .drop_duplicates("acs_occ_code")
    )

    if baseline.empty:
        raise ValueError(
            f"No observations were found for reference year {REFERENCE_YEAR}."
        )

    exposure_cutoff = float(
        baseline["llm_exposure_main"].median()
    )
    high_exposure = (
        baseline["llm_exposure_main"] >= exposure_cutoff
    )

    complementarity_cutoff = float(
        baseline.loc[
            high_exposure,
            "complementarity_theta",
        ].median()
    )

    baseline["ai_category"] = np.select(
        [
            ~high_exposure,
            high_exposure
            & (
                baseline["complementarity_theta"]
                < complementarity_cutoff
            ),
            high_exposure
            & (
                baseline["complementarity_theta"]
                >= complementarity_cutoff
            ),
        ],
        CATEGORY_ORDER,
        default="Unclassified",
    )

    if baseline["ai_category"].eq("Unclassified").any():
        raise ValueError(
            "At least one occupation could not be classified."
        )

    baseline["ai_category"] = pd.Categorical(
        baseline["ai_category"],
        categories=CATEGORY_ORDER,
        ordered=True,
    )
    baseline["category_label"] = (
        baseline["ai_category"]
        .astype("string")
        .map(CATEGORY_LABELS)
    )

    cutoffs = pd.DataFrame(
        {
            "classification_rule": [
                "Overall occupation median",
                "Median within high-exposure occupations",
            ],
            "measure": [
                "llm_exposure_main",
                "complementarity_theta",
            ],
            "cutoff": [
                exposure_cutoff,
                complementarity_cutoff,
            ],
            "reference_year": [
                REFERENCE_YEAR,
                REFERENCE_YEAR,
            ],
        }
    )

    return baseline, cutoffs


# -----------------------------------------------------------------------------
# 2. Descriptive statistics and representative occupations
# -----------------------------------------------------------------------------

def create_descriptive_tables(
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create score, category and representative-occupation tables."""

    score_columns = [
        "llm_exposure_main",
        "complementarity_theta",
        "m_index",
    ]

    score_summary = (
        baseline[score_columns]
        .describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
        )
        .T
        .reset_index()
        .rename(columns={"index": "measure"})
    )

    rows: list[dict[str, object]] = []
    total_employment = float(
        baseline["employment_weight"].sum()
    )

    for category in CATEGORY_ORDER:
        data = baseline.loc[
            baseline["ai_category"].astype("string").eq(category)
        ].copy()

        rows.append(
            {
                "category": category,
                "category_label": CATEGORY_LABELS[category],
                "occupations": data["acs_occ_code"].nunique(),
                "employment_2022": data["employment_weight"].sum(),
                "employment_share_2022": (
                    data["employment_weight"].sum()
                    / total_employment
                ),
                "mean_exposure": data[
                    "llm_exposure_main"
                ].mean(),
                "median_exposure": data[
                    "llm_exposure_main"
                ].median(),
                "mean_complementarity": data[
                    "complementarity_theta"
                ].mean(),
                "median_complementarity": data[
                    "complementarity_theta"
                ].median(),
                "mean_M": data["m_index"].mean(),
                "median_M": data["m_index"].median(),
                # The observation year is 2022, while wages are expressed
                # in constant 2024 dollars.
                "employment_weighted_mean_real_annual_wage_income_2022_in_2024_usd":
                    weighted_mean(
                        data["mean_real_wage_2024"],
                        data["wage_weight"],
                    ),
            }
        )

    category_summary = pd.DataFrame(rows)
    category_summary["category"] = pd.Categorical(
        category_summary["category"],
        categories=CATEGORY_ORDER,
        ordered=True,
    )
    category_summary = category_summary.sort_values(
        "category"
    ).reset_index(drop=True)

    representative_columns = [
        "ai_category",
        "category_label",
        "acs_occ_code",
        "acs_occ_title",
        "soc_major",
        "llm_exposure_main",
        "complementarity_theta",
        "m_index",
        "employment_weight",
        "employment_share",
        "mean_real_wage_2024",
    ]

    representative = (
        baseline[representative_columns]
        .copy()
        .sort_values(
            ["ai_category", "employment_weight"],
            ascending=[True, False],
        )
        .groupby(
            "ai_category",
            sort=False,
            observed=True,
        )
        .head(TOP_OCCUPATIONS_PER_CATEGORY)
        .copy()
    )

    representative["rank_within_category"] = (
        representative.groupby(
            "ai_category",
            observed=True,
        )["employment_weight"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    representative["selection_rule"] = (
        f"Top {TOP_OCCUPATIONS_PER_CATEGORY} occupations by "
        f"{REFERENCE_YEAR} ACS employment weight within category"
    )

    representative = representative.rename(
        columns={
            "mean_real_wage_2024":
                "mean_real_annual_wage_income_2022_in_2024_usd",
        }
    )

    representative["ai_category"] = pd.Categorical(
        representative["ai_category"],
        categories=CATEGORY_ORDER,
        ordered=True,
    )
    representative = representative.sort_values(
        ["ai_category", "rank_within_category"]
    ).reset_index(drop=True)

    return score_summary, category_summary, representative


def create_score_correlations(
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Create an occupation-level correlation matrix."""

    columns = [
        "llm_exposure_main",
        "complementarity_theta",
        "m_index",
    ]

    return (
        baseline[columns]
        .corr()
        .reset_index()
        .rename(columns={"index": "measure"})
    )


# -----------------------------------------------------------------------------
# 3. Descriptive figures
# -----------------------------------------------------------------------------

def exposure_complementarity_map(
    baseline: pd.DataFrame,
    cutoffs: pd.DataFrame,
) -> None:
    """Plot occupation-level exposure and complementarity.

    The horizontal complementarity threshold is drawn only in the
    high-exposure region because complementarity is used to split only
    high-exposure occupations.
    """

    exposure_cutoff = float(
        cutoffs.loc[
            cutoffs["measure"].eq("llm_exposure_main"),
            "cutoff",
        ].iloc[0]
    )
    complementarity_cutoff = float(
        cutoffs.loc[
            cutoffs["measure"].eq("complementarity_theta"),
            "cutoff",
        ].iloc[0]
    )

    plot_data = baseline.copy()
    maximum_employment = float(
        plot_data["employment_weight"].max()
    )

    if maximum_employment <= 0:
        raise ValueError(
            "Maximum employment weight must be positive."
        )

    plot_data["plot_size"] = (
        20
        + 180
        * np.sqrt(
            plot_data["employment_weight"].clip(lower=0)
            / maximum_employment
        )
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    for category in CATEGORY_ORDER:
        data = plot_data.loc[
            plot_data["ai_category"]
            .astype("string")
            .eq(category)
        ]

        ax.scatter(
            data["llm_exposure_main"],
            data["complementarity_theta"],
            s=data["plot_size"],
            alpha=0.65,
            label=f"{category}: {CATEGORY_LABELS[category]}",
        )

    ax.axvline(
        exposure_cutoff,
        linestyle="--",
        linewidth=1,
        label="Exposure cutoff",
    )

    # Draw this threshold only in the high-exposure region.
    current_x_min, current_x_max = ax.get_xlim()
    ax.hlines(
        y=complementarity_cutoff,
        xmin=exposure_cutoff,
        xmax=current_x_max,
        linestyle=":",
        linewidth=1,
        label="Complementarity cutoff within high exposure",
    )
    ax.set_xlim(current_x_min, current_x_max)

    # Label six pre-selected, large occupations. This avoids automatic
    # truncation and prevents labels from overlapping.
    labelled = plot_data.loc[
        plot_data["acs_occ_title"].isin(FIGURE_LABEL_TITLES)
    ].copy()

    for occupation_title in FIGURE_LABEL_TITLES:
        row_data = labelled.loc[
            labelled["acs_occ_title"].eq(occupation_title)
        ]

        # Continue safely if a title differs in a future occupation file.
        if row_data.empty:
            continue

        row = row_data.iloc[0]
        plot_label = FIGURE_LABEL_TEXT.get(
            occupation_title,
            occupation_title,
        )
        offset = FIGURE_LABEL_OFFSETS.get(
            occupation_title,
            (6, 6),
        )

        ax.annotate(
            plot_label,
            (
                row["llm_exposure_main"],
                row["complementarity_theta"],
            ),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            annotation_clip=False,
        )

    ax.set_xlabel("LLM exposure")
    ax.set_ylabel("Occupational complementarity")
    ax.set_title(
        "Occupational LLM exposure and complementarity"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig.savefig(
        out("exposure_complementarity_map.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def m_vs_exposure_figure(
    baseline: pd.DataFrame,
) -> None:
    """Plot M against Exposure and display their correlation."""

    correlation = float(
        baseline[
            ["llm_exposure_main", "m_index"]
        ].corr().iloc[0, 1]
    )

    plot_data = baseline.copy()
    maximum_employment = float(
        plot_data["employment_weight"].max()
    )

    if maximum_employment <= 0:
        raise ValueError(
            "Maximum employment weight must be positive."
        )

    plot_data["plot_size"] = (
        20
        + 180
        * np.sqrt(
            plot_data["employment_weight"].clip(lower=0)
            / maximum_employment
        )
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    for category in CATEGORY_ORDER:
        data = plot_data.loc[
            plot_data["ai_category"]
            .astype("string")
            .eq(category)
        ]

        ax.scatter(
            data["llm_exposure_main"],
            data["m_index"],
            s=data["plot_size"],
            alpha=0.65,
            label=category,
        )

    ax.text(
        0.03,
        0.96,
        f"Occupation-level correlation = {correlation:.3f}",
        transform=ax.transAxes,
        va="top",
    )
    ax.set_xlabel("LLM exposure")
    ax.set_ylabel(
        "Composite index M = Exposure × Complementarity"
    )
    ax.set_title("Composite index M and LLM exposure")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        out("M_vs_exposure.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. Format final regression and joint-test tables
# -----------------------------------------------------------------------------

def format_main_results() -> pd.DataFrame:
    """Create a compact dissertation table for wage and employment."""

    require_files([MAIN_RESULTS_FILE])
    results = pd.read_csv(MAIN_RESULTS_FILE)

    required = {
        "analysis",
        "outcome_name",
        "specification",
        "coefficient",
        "standard_error",
        "p_value",
        "q_value_bh",
        "ci_low",
        "ci_high",
        "occupation_clusters",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise KeyError(
            "The Step-3 dissertation main table is missing: "
            + str(missing)
        )

    results = results.loc[
        results["outcome_name"].isin(
            ["wage", "employment"]
        )
    ].copy()

    order = list(SPECIFICATION_LABELS)
    order_lookup = {
        specification: index
        for index, specification in enumerate(order)
    }

    rows: list[dict[str, object]] = []

    for (
        analysis,
        specification,
    ), data in results.groupby(
        ["analysis", "specification"],
        sort=False,
    ):
        wage = data.loc[
            data["outcome_name"].eq("wage")
        ]
        employment = data.loc[
            data["outcome_name"].eq("employment")
        ]

        if wage.empty or employment.empty:
            continue

        wage_row = wage.iloc[0]
        employment_row = employment.iloc[0]

        rows.append(
            {
                "section": ANALYSIS_LABELS.get(
                    analysis,
                    analysis,
                ),
                "term": SPECIFICATION_LABELS.get(
                    specification,
                    specification,
                ),
                "wage_estimate_se": format_estimate(
                    wage_row["coefficient"],
                    wage_row["standard_error"],
                    wage_row["p_value"],
                ),
                "wage_p_value": wage_row["p_value"],
                "wage_q_value": wage_row["q_value_bh"],
                "employment_estimate_se": format_estimate(
                    employment_row["coefficient"],
                    employment_row["standard_error"],
                    employment_row["p_value"],
                ),
                "employment_p_value":
                    employment_row["p_value"],
                "employment_q_value":
                    employment_row["q_value_bh"],
                "occupation_clusters_wage":
                    wage_row["occupation_clusters"],
                "occupation_clusters_employment":
                    employment_row["occupation_clusters"],
                "sort_order": order_lookup.get(
                    specification,
                    999,
                ),
            }
        )

    formatted = pd.DataFrame(rows)

    if formatted.empty:
        raise ValueError(
            "No matched wage and employment results were found."
        )

    formatted = (
        formatted
        .sort_values("sort_order")
        .drop(columns="sort_order")
        .reset_index(drop=True)
    )

    return formatted


def write_main_results_latex(
    table: pd.DataFrame,
) -> None:
    """Write a LaTeX version of the compact main table."""

    latex_table = table[
        [
            "section",
            "term",
            "wage_estimate_se",
            "employment_estimate_se",
        ]
    ].copy()

    latex = latex_table.to_latex(
        index=False,
        escape=True,
        column_format="llcc",
        caption=(
            "LLM exposure, occupational complementarity, "
            "and post-2022 labour-market outcomes"
        ),
        label="tab:main_results",
    )

    notes = (
        "\n% Notes: Each cell reports the coefficient followed by the "
        "occupation-clustered standard error in parentheses. "
        "The estimation sample covers 2022--2024. "
        "All models include occupation fixed effects, year fixed effects, "
        "and full-rank SOC-major-group-specific year deviations. "
        "The wage regressions use 2022 baseline wage weights. "
        "* p<0.10, ** p<0.05, *** p<0.01.\n"
    )

    out("main_results_formatted.tex").write_text(
        latex + notes,
        encoding="utf-8",
    )


def format_joint_tests() -> pd.DataFrame:
    """Create a compact table of the four joint significance tests."""

    require_files([JOINT_TESTS_FILE])
    tests = pd.read_csv(JOINT_TESTS_FILE)

    required = {
        "test_name",
        "outcome_name",
        "test_statistic",
        "p_value",
        "q_value_bh",
    }
    missing = sorted(required.difference(tests.columns))
    if missing:
        raise KeyError(
            "The Step-3 joint-test table is missing: "
            + str(missing)
        )

    tests = tests.loc[
        tests["outcome_name"].isin(
            ["wage", "employment"]
        )
    ].copy()

    test_labels = {
        "Exposure_2023_2024_joint":
            "Exposure: 2023 and 2024 jointly zero",
        "Complementarity_2023_2024_joint":
            "Complementarity: 2023 and 2024 jointly zero",
        "Interaction_2023_2024_joint":
            "Exposure × Complementarity: 2023 and 2024 jointly zero",
        "HEHC_minus_HELC_2023_2024_joint":
            "HEHC − HELC: 2023 and 2024 jointly zero",
    }

    rows: list[dict[str, object]] = []

    for test_name, data in tests.groupby(
        "test_name",
        sort=False,
    ):
        wage = data.loc[
            data["outcome_name"].eq("wage")
        ]
        employment = data.loc[
            data["outcome_name"].eq("employment")
        ]

        if wage.empty or employment.empty:
            continue

        rows.append(
            {
                "joint_test": test_labels.get(
                    test_name,
                    test_name,
                ),
                "wage_test_statistic":
                    wage.iloc[0]["test_statistic"],
                "wage_p_value":
                    wage.iloc[0]["p_value"],
                "wage_q_value":
                    wage.iloc[0]["q_value_bh"],
                "employment_test_statistic":
                    employment.iloc[0]["test_statistic"],
                "employment_p_value":
                    employment.iloc[0]["p_value"],
                "employment_q_value":
                    employment.iloc[0]["q_value_bh"],
            }
        )

    formatted = pd.DataFrame(rows)

    if formatted.empty:
        raise ValueError(
            "No matched wage and employment joint tests were found."
        )

    return formatted


# -----------------------------------------------------------------------------
# 5. Appendix-ready robustness tables
# -----------------------------------------------------------------------------

def create_pretrend_appendix() -> pd.DataFrame:
    """Create the preferred Step-2 pre-trend appendix table."""

    if not PRETREND_FILE.exists():
        return pd.DataFrame()

    pretrend = pd.read_csv(PRETREND_FILE)

    required = {
        "fe_type",
        "outcome",
        "dimension",
        "tested_pre_years",
        "reference_year",
        "joint_pretrend_p_value",
        "occupation_clusters",
    }
    missing = sorted(required.difference(pretrend.columns))
    if missing:
        raise KeyError(
            "The Step-2 pre-trend table is missing: "
            + str(missing)
        )

    preferred = pretrend.loc[
        pretrend["fe_type"].eq("soc_year")
        & pretrend["outcome"].isin(
            [
                "mean_log_real_wage_2024",
                "log_employment",
            ]
        )
    ].copy()

    outcome_labels = {
        "mean_log_real_wage_2024":
            "Real annual wage income",
        "log_employment":
            "Occupational employment",
    }
    preferred["outcome_label"] = (
        preferred["outcome"].map(outcome_labels)
    )

    keep = [
        "outcome_label",
        "dimension",
        "tested_pre_years",
        "reference_year",
        "joint_pretrend_p_value",
        "occupation_clusters",
    ]

    return preferred[keep].sort_values(
        ["outcome_label", "dimension"]
    ).reset_index(drop=True)


def create_m_appendix() -> pd.DataFrame:
    """Create the final M-index robustness appendix table."""

    if not M_RESULTS_FILE.exists():
        return pd.DataFrame()

    m_results = pd.read_csv(M_RESULTS_FILE)

    required = {
        "outcome_name",
        "model_type",
        "term",
        "coefficient",
        "standard_error",
        "p_value",
        "q_value_bh",
        "ci_low",
        "ci_high",
        "occupation_clusters",
    }
    missing = sorted(required.difference(m_results.columns))
    if missing:
        raise KeyError(
            "The Step-3 M-robustness table is missing: "
            + str(missing)
        )

    m_results = m_results.loc[
        m_results["outcome_name"].isin(
            ["wage", "employment"]
        )
    ].copy()

    term_labels = {
        "m_index_z_final_post": "M × Post",
        "m_index_z_final_y2023": "M × 2023",
        "m_index_z_final_y2024": "M × 2024",
    }
    m_results["term_label"] = (
        m_results["term"].map(term_labels)
    )

    keep = [
        "outcome_name",
        "model_type",
        "term_label",
        "coefficient",
        "standard_error",
        "p_value",
        "q_value_bh",
        "ci_low",
        "ci_high",
        "occupation_clusters",
    ]

    return m_results[keep].reset_index(drop=True)


# -----------------------------------------------------------------------------
# 6. Alternative-measure robustness and specification sensitivity
# -----------------------------------------------------------------------------

def create_alternative_measure_robustness() -> pd.DataFrame:
    """Create a compact pooled robustness table from the final Step-4 models."""

    require_files([ALT_ROBUST_POOLED_FILE, ALT_ROBUST_SUMMARY_FILE])

    results = pd.read_csv(ALT_ROBUST_POOLED_FILE)
    summary = pd.read_csv(ALT_ROBUST_SUMMARY_FILE)

    required = {
        "robustness",
        "robustness_label",
        "outcome",
        "term",
        "coefficient",
        "standard_error",
        "p_value",
        "q_value_bh",
        "occupation_clusters",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise KeyError(
            "The Step-4 pooled robustness table is missing: "
            + str(missing)
        )

    outcome_labels = {
        "mean_log_real_wage_2024": "Real annual wage income",
        "log_employment": "Occupational employment",
    }
    term_labels = {
        "robust_exposure_post": "Exposure × Post",
        "robust_complementarity_post": "Complementarity × Post",
        "robust_interaction_post": "Exposure × Complementarity × Post",
    }
    term_order = list(term_labels)

    results = results.loc[
        results["outcome"].isin(outcome_labels)
        & results["term"].isin(term_labels)
    ].copy()

    rows: list[dict[str, object]] = []

    for robustness_name, robustness_data in results.groupby(
        "robustness",
        sort=False,
    ):
        robustness_label = robustness_data["robustness_label"].iloc[0]

        summary_match = summary.loc[
            summary["robustness"].eq(robustness_name)
        ]
        exposure_variable = (
            summary_match["exposure_variable"].iloc[0]
            if not summary_match.empty
            else ""
        )
        complementarity_variable = (
            summary_match["complementarity_variable"].iloc[0]
            if not summary_match.empty
            else ""
        )

        for outcome, outcome_label in outcome_labels.items():
            data = robustness_data.loc[
                robustness_data["outcome"].eq(outcome)
            ].copy()

            if data.empty:
                continue

            row: dict[str, object] = {
                "robustness": robustness_name,
                "robustness_label": robustness_label,
                "outcome": outcome_label,
                "exposure_variable": exposure_variable,
                "complementarity_variable": complementarity_variable,
                "occupation_clusters": int(data["occupation_clusters"].iloc[0]),
            }

            for term in term_order:
                match = data.loc[data["term"].eq(term)]
                if match.empty:
                    continue

                item = match.iloc[0]
                prefix = {
                    "robust_exposure_post": "exposure",
                    "robust_complementarity_post": "complementarity",
                    "robust_interaction_post": "interaction",
                }[term]

                row[f"{prefix}_estimate_se"] = format_estimate(
                    item["coefficient"],
                    item["standard_error"],
                    item["p_value"],
                )
                row[f"{prefix}_coefficient"] = item["coefficient"]
                row[f"{prefix}_standard_error"] = item["standard_error"]
                row[f"{prefix}_p_value"] = item["p_value"]
                row[f"{prefix}_q_value"] = item["q_value_bh"]

            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No pooled alternative-measure robustness results were found.")

    robustness_order = list(ROBUSTNESS_DISPLAY_ORDER)
    outcome_order = ["Real annual wage income", "Occupational employment"]

    table["robustness_sort"] = table["robustness"].map(
        {name: i for i, name in enumerate(robustness_order)}
    ).fillna(999)
    table["outcome_sort"] = table["outcome"].map(
        {name: i for i, name in enumerate(outcome_order)}
    ).fillna(999)

    return (
        table.sort_values(["robustness_sort", "outcome_sort"])
        .drop(columns=["robustness_sort", "outcome_sort"])
        .reset_index(drop=True)
    )


def create_alternative_measure_dynamic_appendix() -> pd.DataFrame:
    """Create a detailed year-specific robustness table for the appendix."""

    require_files([ALT_ROBUST_DYNAMIC_FILE])
    data = pd.read_csv(ALT_ROBUST_DYNAMIC_FILE)

    required = {
        "robustness",
        "robustness_label",
        "outcome",
        "term",
        "coefficient",
        "standard_error",
        "p_value",
        "q_value_bh",
        "ci_low",
        "ci_high",
        "occupation_clusters",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(
            "The Step-4 dynamic robustness table is missing: "
            + str(missing)
        )

    outcome_labels = {
        "mean_log_real_wage_2024": "Real annual wage income",
        "log_employment": "Occupational employment",
    }
    term_labels = {
        "robust_exposure_y2023": "Exposure × 2023",
        "robust_exposure_y2024": "Exposure × 2024",
        "robust_complementarity_y2023": "Complementarity × 2023",
        "robust_complementarity_y2024": "Complementarity × 2024",
        "robust_interaction_y2023": "Exposure × Complementarity × 2023",
        "robust_interaction_y2024": "Exposure × Complementarity × 2024",
    }

    data = data.loc[
        data["outcome"].isin(outcome_labels)
        & data["term"].isin(term_labels)
    ].copy()

    data["outcome_label"] = data["outcome"].map(outcome_labels)
    data["term_label"] = data["term"].map(term_labels)
    data["estimate_se"] = data.apply(
        lambda row: format_estimate(
            row["coefficient"],
            row["standard_error"],
            row["p_value"],
        ),
        axis=1,
    )

    keep = [
        "robustness",
        "robustness_label",
        "outcome_label",
        "term_label",
        "estimate_se",
        "coefficient",
        "standard_error",
        "p_value",
        "q_value_bh",
        "ci_low",
        "ci_high",
        "occupation_clusters",
    ]

    return data[keep].reset_index(drop=True)


def create_alternative_measure_joint_appendix() -> pd.DataFrame:
    """Create joint 2023/2024 tests for the alternative-measure appendix."""

    require_files([ALT_ROBUST_JOINT_FILE])
    data = pd.read_csv(ALT_ROBUST_JOINT_FILE)

    required = {
        "robustness",
        "robustness_label",
        "outcome",
        "dimension",
        "test_statistic",
        "p_value",
        "q_value_bh",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(
            "The Step-4 robustness joint-test table is missing: "
            + str(missing)
        )

    outcome_labels = {
        "mean_log_real_wage_2024": "Real annual wage income",
        "log_employment": "Occupational employment",
    }
    data = data.loc[data["outcome"].isin(outcome_labels)].copy()
    data["outcome_label"] = data["outcome"].map(outcome_labels)

    keep = [
        "robustness",
        "robustness_label",
        "outcome_label",
        "dimension",
        "test_statistic",
        "p_value",
        "q_value_bh",
    ]

    return data[keep].reset_index(drop=True)


def create_specification_sensitivity() -> pd.DataFrame:
    """Compare basic FE with the preferred SOC-major-by-year specification."""

    require_files([SPEC_SENSITIVITY_FILE])
    data = pd.read_csv(SPEC_SENSITIVITY_FILE)

    required = {
        "model",
        "window_years",
        "fe_type",
        "outcome",
        "term",
        "coefficient",
        "standard_error",
        "p_value",
        "occupation_clusters",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(
            "The Step-2 continuous-model table is missing: "
            + str(missing)
        )

    outcome_labels = {
        "mean_log_real_wage_2024": "Real annual wage income",
        "log_employment": "Occupational employment",
    }
    term_labels = {
        "exposure_z_post": "Exposure × Post",
        "complementarity_z_post": "Complementarity × Post",
        "ec_interaction_post": "Exposure × Complementarity × Post",
    }

    preferred_window = "2022,2023,2024"

    subset = data.loc[
        data["window_years"].astype(str).eq(preferred_window)
        & data["model"].astype(str).str.contains("_joint", na=False)
        & data["fe_type"].isin(["basic", "soc_year"])
        & data["outcome"].isin(outcome_labels)
        & data["term"].isin(term_labels)
    ].copy()

    if subset.empty:
        raise ValueError(
            "No short-window basic-versus-SOC-year joint models were found."
        )

    rows: list[dict[str, object]] = []

    for outcome in outcome_labels:
        for term in term_labels:
            basic = subset.loc[
                subset["outcome"].eq(outcome)
                & subset["term"].eq(term)
                & subset["fe_type"].eq("basic")
            ]
            soc_year = subset.loc[
                subset["outcome"].eq(outcome)
                & subset["term"].eq(term)
                & subset["fe_type"].eq("soc_year")
            ]

            if basic.empty or soc_year.empty:
                continue

            b = basic.iloc[0]
            s = soc_year.iloc[0]

            rows.append(
                {
                    "outcome": outcome_labels[outcome],
                    "term": term_labels[term],
                    "basic_estimate_se": format_estimate(
                        b["coefficient"],
                        b["standard_error"],
                        b["p_value"],
                    ),
                    "basic_coefficient": b["coefficient"],
                    "basic_standard_error": b["standard_error"],
                    "basic_p_value": b["p_value"],
                    "soc_year_estimate_se": format_estimate(
                        s["coefficient"],
                        s["standard_error"],
                        s["p_value"],
                    ),
                    "soc_year_coefficient": s["coefficient"],
                    "soc_year_standard_error": s["standard_error"],
                    "soc_year_p_value": s["p_value"],
                    "coefficient_change_soc_minus_basic": (
                        s["coefficient"] - b["coefficient"]
                    ),
                    "occupation_clusters": int(s["occupation_clusters"]),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No specification-sensitivity rows could be created.")

    outcome_order = list(outcome_labels.values())
    term_order = list(term_labels.values())
    table["outcome_sort"] = table["outcome"].map(
        {name: i for i, name in enumerate(outcome_order)}
    )
    table["term_sort"] = table["term"].map(
        {name: i for i, name in enumerate(term_order)}
    )

    return (
        table.sort_values(["outcome_sort", "term_sort"])
        .drop(columns=["outcome_sort", "term_sort"])
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# 6. Results manifest
# -----------------------------------------------------------------------------

def write_manifest(
    category_summary: pd.DataFrame,
) -> None:
    """Write a guide explaining where each Step-5 output belongs."""

    category_text = "\n".join(
        (
            f"  - {row.category}: "
            f"{int(row.occupations)} occupations; "
            f"{row.employment_share_2022:.1%} of 2022 matched employment"
        )
        for row in category_summary.itertuples()
    )

    manifest = f"""STEP 5 OUTPUT GUIDE

MAIN TEXT

Figure 1:
  out_step5_exposure_complementarity_map.png
  Purpose: Descriptive two-dimensional map of LLM exposure and
  occupational complementarity.
  Bubble size reflects 2022 occupational employment.

Descriptive tables:
  out_step5_score_summary.csv
  out_step5_category_summary.csv
  out_step5_representative_occupations.csv
  Purpose: Occupation-score distribution, category composition and
  representative occupations.

Preferred regression results:
  out_step5_main_results_formatted.csv
  out_step5_main_results_formatted.tex
  Purpose: Final 2022-2024 preferred estimates from Step 3.

Joint tests:
  out_step5_joint_tests_formatted.csv
  Purpose: Joint 2023/2024 tests from Step 3.

Robustness to alternative measures:
  out_step5_alternative_measure_robustness.csv
  Purpose: Compact pooled results using GPT/model-rated exposure,
  complementarity excluding Job Zones, and OEWS-employment-weighted mapping.

Specification sensitivity:
  out_step5_specification_sensitivity.csv
  Purpose: Compare occupation+year FE with the preferred model that also
  allows SOC major groups to follow different annual paths.

APPENDIX / SUPPORTING OUTPUTS

Figure A1:
  out_step5_M_vs_exposure.png
  Purpose: Shows the close descriptive relationship between M and Exposure.

Pre-period trend diagnostics:
  out_step5_appendix_pretrend.csv

Original M-index robustness:
  out_step5_appendix_M_robustness.csv

Alternative-measure dynamic estimates:
  out_step5_appendix_alternative_measure_dynamic.csv

Alternative-measure joint tests:
  out_step5_appendix_alternative_measure_joint_tests.csv

Existing Step-2 event-study figures:
  Place in the appendix, not as the preferred short-window estimates.

PREFERRED EMPIRICAL SPECIFICATION

  Sample: 2022, 2023 and 2024
  Reference year: 2022
  Fixed effects:
    - occupation
    - year
    - SOC-major-group-specific year deviations, normalized relative to a
      reference SOC major group and 2022 to avoid redundant indicators
  Standard errors: clustered by occupation
  Wage weighting: fixed 2022 baseline wage weight
  Employment models: unweighted

PREFERRED CATEGORY DEFINITION

  LE:
    Exposure below the overall occupation median.

  HELC:
    Exposure at or above the overall occupation median and
    complementarity below the median among high-exposure occupations.

  HEHC:
    Exposure at or above the overall occupation median and
    complementarity at or above the median among high-exposure occupations.

PREFERRED CATEGORY COMPOSITION

{category_text}

ALTERNATIVE-MEASURE COVERAGE NOTE

  GPT/model-rated exposure robustness: 498 occupations.
  No-Job-Zone complementarity robustness: 498 occupations.
  OEWS-employment-weighted mapping robustness: 497 occupations because one
  ACS occupation lacks the required OEWS employment weights.

WAGE VARIABLE NOTE

  Category wages use 2022 observations expressed in constant 2024 USD.
  They must be described as:
  "2022 employment-weighted mean real annual wage income (2024 USD)."

INTERPRETATION RULE

  Statistical insignificance must be described as a lack of robust evidence,
  not as proof that the true effect is exactly zero.
"""

    out("results_manifest.txt").write_text(
        manifest,
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# 7. Main
# -----------------------------------------------------------------------------

def main() -> None:
    """Run the complete Step-5 reporting pipeline."""

    print("\nSTEP 5: FINAL DISSERTATION TABLES AND FIGURES\n")

    panel = load_panel()
    baseline, cutoffs = construct_categories(panel)

    cutoffs.to_csv(
        out("category_cutoffs.csv"),
        index=False,
    )

    (
        score_summary,
        category_summary,
        representative,
    ) = create_descriptive_tables(baseline)

    score_summary.to_csv(
        out("score_summary.csv"),
        index=False,
    )
    category_summary.to_csv(
        out("category_summary.csv"),
        index=False,
    )
    representative.to_csv(
        out("representative_occupations.csv"),
        index=False,
    )

    correlations = create_score_correlations(baseline)
    correlations.to_csv(
        out("score_correlations.csv"),
        index=False,
    )

    exposure_complementarity_map(
        baseline,
        cutoffs,
    )
    m_vs_exposure_figure(baseline)

    main_results = format_main_results()
    main_results.to_csv(
        out("main_results_formatted.csv"),
        index=False,
    )
    write_main_results_latex(main_results)

    joint_tests = format_joint_tests()
    joint_tests.to_csv(
        out("joint_tests_formatted.csv"),
        index=False,
    )

    pretrend_appendix = create_pretrend_appendix()
    if not pretrend_appendix.empty:
        pretrend_appendix.to_csv(
            out("appendix_pretrend.csv"),
            index=False,
        )

    m_appendix = create_m_appendix()
    if not m_appendix.empty:
        m_appendix.to_csv(
            out("appendix_M_robustness.csv"),
            index=False,
        )

    alternative_pooled = create_alternative_measure_robustness()
    alternative_pooled.to_csv(
        out("alternative_measure_robustness.csv"),
        index=False,
    )

    alternative_dynamic = create_alternative_measure_dynamic_appendix()
    alternative_dynamic.to_csv(
        out("appendix_alternative_measure_dynamic.csv"),
        index=False,
    )

    alternative_joint = create_alternative_measure_joint_appendix()
    alternative_joint.to_csv(
        out("appendix_alternative_measure_joint_tests.csv"),
        index=False,
    )

    specification_sensitivity = create_specification_sensitivity()
    specification_sensitivity.to_csv(
        out("specification_sensitivity.csv"),
        index=False,
    )

    write_manifest(category_summary)

    print("Completed. Review these files first:")
    print("  1. out_step5_exposure_complementarity_map.png")
    print("  2. out_step5_category_summary.csv")
    print("  3. out_step5_main_results_formatted.csv")
    print("  4. out_step5_joint_tests_formatted.csv")
    print("  5. out_step5_alternative_measure_robustness.csv")
    print("  6. out_step5_specification_sensitivity.csv")
    print("  7. out_step5_appendix_pretrend.csv")
    print("  8. out_step5_appendix_M_robustness.csv")
    print("  9. out_step5_appendix_alternative_measure_dynamic.csv")
    print(" 10. out_step5_appendix_alternative_measure_joint_tests.csv")
    print(" 11. out_step5_results_manifest.txt")


if __name__ == "__main__":
    main()