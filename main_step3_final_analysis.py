"""Step 3: Final dissertation analysis.

This script reads the balanced occupation-year panel created by
main_step1_revised.py. It does not reread the raw ACS files.

Final analyses:
1. Preferred short window: 2022-2024.
2. Occupation FE, year FE, and SOC-major-by-year FE.
3. Separate 2023 and 2024 coefficients.
4. Exposure, complementarity, and their interaction kept as distinct dimensions.
5. HEHC versus HELC comparison within high-exposure occupations, controlling
   for continuous exposure differences.
6. Employment-share robustness.
7. Alternative category thresholds.
8. Benjamini-Hochberg false-discovery-rate adjustments.
9. Original M retained only as a robustness measure.

Keep this file in the same folder as:
    out_occupation_year_panel_balanced.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# -----------------------------------------------------------------------------
# 0. Settings
# -----------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent
PANEL_FILE = BASE / "out_occupation_year_panel_balanced.csv"

ANALYSIS_YEARS = [2022, 2023, 2024]
REFERENCE_YEAR = 2022
POST_YEARS = [2023, 2024]

OUTCOMES = {
    "wage": {
        "column": "mean_log_real_wage_2024",
        "weight": "baseline_wage_weight",
        "label": "Real annual wage income",
    },
    "employment": {
        "column": "log_employment",
        "weight": None,
        "label": "Occupational employment",
    },
    "employment_share": {
        "column": "log_employment_share",
        "weight": None,
        "label": "Occupational employment share",
    },
}

# Pre-specified category definitions. Do not add thresholds after seeing results.
CATEGORY_SCHEMES = {
    # Preferred category comparison:
    # high exposure = top half of occupations;
    # high complementarity = above the median within the high-exposure sample.
    "baseline_within_highE_median": {
        "exposure_quantile": 0.50,
        "complementarity_rule": "within_highE",
        "complementarity_quantile": 0.50,
    },
    # Reproduces the Step-2 global-median classification.
    "global_median": {
        "exposure_quantile": 0.50,
        "complementarity_rule": "global",
        "complementarity_quantile": 0.50,
    },
    # A stricter global threshold used only as robustness.
    "global_60th": {
        "exposure_quantile": 0.60,
        "complementarity_rule": "global",
        "complementarity_quantile": 0.60,
    },
}

MIN_CATEGORY_OCCUPATIONS = 30


def out(name: str) -> Path:
    return BASE / f"out_step3_{name}"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    standard_deviation = values.std(ddof=0)
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - values.mean()) / standard_deviation


# -----------------------------------------------------------------------------
# 1. Load and validate the Step-1 panel
# -----------------------------------------------------------------------------

def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"{PANEL_FILE.name} was not found. "
            "Run main_step1_revised.py first and keep its output in this folder."
        )

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
        "mean_log_real_wage_2024",
        "log_employment",
        "employment_share",
        "baseline_wage_weight",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise KeyError(
            "The balanced panel is missing required columns: " + str(missing)
        )

    panel = panel.copy()
    panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
    panel["acs_occ_code"] = panel["acs_occ_code"].astype("string")
    panel["soc_major"] = panel["soc_major"].astype("string")

    # Construct the log employment share if an older Step-1 output lacks it.
    if "log_employment_share" not in panel.columns:
        panel["log_employment_share"] = np.log(
            pd.to_numeric(panel["employment_share"], errors="coerce")
            .clip(lower=1e-12)
        )

    panel = panel.loc[panel["year"].isin(ANALYSIS_YEARS)].copy()

    # Standardise scores across unique occupations, not repeated occupation-years.
    occupation_scores = (
        panel[
            [
                "acs_occ_code",
                "llm_exposure_main",
                "complementarity_theta",
                "m_index",
            ]
        ]
        .drop_duplicates("acs_occ_code")
        .copy()
    )
    occupation_scores["exposure_z"] = zscore(
        occupation_scores["llm_exposure_main"]
    )
    occupation_scores["complementarity_z"] = zscore(
        occupation_scores["complementarity_theta"]
    )
    occupation_scores["ec_interaction"] = (
        occupation_scores["exposure_z"]
        * occupation_scores["complementarity_z"]
    )
    occupation_scores["m_index_z_final"] = zscore(
        occupation_scores["m_index"]
    )

    new_score_cols = [
        "exposure_z",
        "complementarity_z",
        "ec_interaction",
        "m_index_z_final",
    ]
    panel = panel.drop(
        columns=[column for column in new_score_cols if column in panel.columns],
        errors="ignore",
    )
    panel = panel.merge(
        occupation_scores[
            ["acs_occ_code", *new_score_cols]
        ],
        on="acs_occ_code",
        how="left",
        validate="many_to_one",
    )

    # Post and year-specific interactions.
    panel["post"] = panel["year"].isin(POST_YEARS).astype(int)

    for score in new_score_cols:
        panel[f"{score}_post"] = panel[score] * panel["post"]
        for year in POST_YEARS:
            panel[f"{score}_y{year}"] = (
                panel[score] * panel["year"].eq(year).astype(int)
            )

    # Basic data checks saved for the dissertation audit trail.
    check_rows = []
    for year, data in panel.groupby("year"):
        check_rows.append(
            {
                "year": int(year),
                "occupation_year_cells": len(data),
                "unique_occupations": data["acs_occ_code"].nunique(),
                "valid_wage_cells": data["mean_log_real_wage_2024"].notna().sum(),
                "valid_employment_cells": data["log_employment"].notna().sum(),
                "valid_employment_share_cells": (
                    data["log_employment_share"].notna().sum()
                ),
            }
        )
    data_check = pd.DataFrame(check_rows)

    score_summary = (
        occupation_scores[
            [
                "llm_exposure_main",
                "complementarity_theta",
                "m_index",
                "exposure_z",
                "complementarity_z",
                "ec_interaction",
                "m_index_z_final",
            ]
        ]
        .describe()
        .T
    )
    score_summary.to_csv(out("score_summary.csv"))

    return panel, data_check


# -----------------------------------------------------------------------------
# 2. Regression helpers
# -----------------------------------------------------------------------------

def fixed_effects() -> str:
    return (
        "C(acs_occ_code) + C(year) "
        "+ C(soc_major):C(year)"
    )


def fit_fe_model(
    data: pd.DataFrame,
    outcome: str,
    terms: Sequence[str],
    model_name: str,
    weight_col: Optional[str] = None,
) -> tuple[object, list[dict[str, object]]]:
    required_cols = [
        outcome,
        "acs_occ_code",
        "soc_major",
        "year",
        *terms,
    ]
    if weight_col:
        required_cols.append(weight_col)

    model_data = (
        data[required_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    if weight_col:
        model_data = model_data.loc[model_data[weight_col] > 0].copy()

    formula = (
        f"{outcome} ~ "
        + " + ".join(terms)
        + " + "
        + fixed_effects()
    )

    if weight_col:
        model = smf.wls(
            formula,
            data=model_data,
            weights=model_data[weight_col],
        )
    else:
        model = smf.ols(formula, data=model_data)

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": model_data["acs_occ_code"]},
        use_t=True,
    )

    out(f"{safe_name(model_name)}.txt").write_text(
        result.summary().as_text(),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for term in terms:
        if term not in result.params.index:
            continue
        confidence_interval = result.conf_int().loc[term]
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "term": term,
                "coefficient": result.params[term],
                "standard_error": result.bse[term],
                "p_value": result.pvalues[term],
                "ci_low": confidence_interval.iloc[0],
                "ci_high": confidence_interval.iloc[1],
                "occupation_year_cells": len(model_data),
                "occupation_clusters": model_data["acs_occ_code"].nunique(),
                "r_squared": result.rsquared,
                "weight_col": weight_col or "none",
            }
        )

    return result, rows


def add_fdr(
    table: pd.DataFrame,
    group_cols: Sequence[str],
    p_col: str = "p_value",
) -> pd.DataFrame:
    table = table.copy()
    table["q_value_bh"] = np.nan
    table["reject_fdr_5pct"] = False

    for _, index in table.groupby(list(group_cols), dropna=False).groups.items():
        index = list(index)
        valid_index = [
            row_index
            for row_index in index
            if pd.notna(table.loc[row_index, p_col])
        ]
        if not valid_index:
            continue

        reject, q_values, _, _ = multipletests(
            table.loc[valid_index, p_col].astype(float),
            alpha=0.05,
            method="fdr_bh",
        )
        table.loc[valid_index, "q_value_bh"] = q_values
        table.loc[valid_index, "reject_fdr_5pct"] = reject

    return table


# -----------------------------------------------------------------------------
# 3. Preferred continuous specifications
# -----------------------------------------------------------------------------

def run_continuous_models(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_terms = [
        "exposure_z_post",
        "complementarity_z_post",
        "ec_interaction_post",
    ]
    dynamic_terms = [
        "exposure_z_y2023",
        "exposure_z_y2024",
        "complementarity_z_y2023",
        "complementarity_z_y2024",
        "ec_interaction_y2023",
        "ec_interaction_y2024",
    ]

    pooled_rows: list[dict[str, object]] = []
    dynamic_rows: list[dict[str, object]] = []

    for outcome_name, settings in OUTCOMES.items():
        outcome = settings["column"]
        weight = settings["weight"]

        _, rows = fit_fe_model(
            data=panel,
            outcome=outcome,
            terms=pooled_terms,
            model_name=f"{outcome_name}_preferred_pooled",
            weight_col=weight,
        )
        for row in rows:
            row["outcome_name"] = outcome_name
        pooled_rows.extend(rows)

        _, rows = fit_fe_model(
            data=panel,
            outcome=outcome,
            terms=dynamic_terms,
            model_name=f"{outcome_name}_preferred_dynamic",
            weight_col=weight,
        )
        for row in rows:
            row["outcome_name"] = outcome_name
        dynamic_rows.extend(rows)

    pooled_table = add_fdr(
        pd.DataFrame(pooled_rows),
        group_cols=["outcome_name"],
    )
    dynamic_table = add_fdr(
        pd.DataFrame(dynamic_rows),
        group_cols=["outcome_name"],
    )
    return pooled_table, dynamic_table


# -----------------------------------------------------------------------------
# 4. Exposure-adjusted HEHC versus HELC comparisons
# -----------------------------------------------------------------------------

def construct_category_scheme(
    panel: pd.DataFrame,
    scheme_name: str,
    settings: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    occupations = (
        panel[
            [
                "acs_occ_code",
                "llm_exposure_main",
                "complementarity_theta",
                "exposure_z",
            ]
        ]
        .drop_duplicates("acs_occ_code")
        .copy()
    )

    exposure_cutoff = float(
        occupations["llm_exposure_main"].quantile(
            float(settings["exposure_quantile"])
        )
    )
    occupations["high_exposure"] = (
        occupations["llm_exposure_main"] >= exposure_cutoff
    )

    high_exposure_occupations = occupations.loc[
        occupations["high_exposure"]
    ].copy()

    if settings["complementarity_rule"] == "within_highE":
        complementarity_cutoff = float(
            high_exposure_occupations["complementarity_theta"].quantile(
                float(settings["complementarity_quantile"])
            )
        )
    elif settings["complementarity_rule"] == "global":
        complementarity_cutoff = float(
            occupations["complementarity_theta"].quantile(
                float(settings["complementarity_quantile"])
            )
        )
    else:
        raise ValueError(
            f"Unknown complementarity rule for {scheme_name}: "
            f"{settings['complementarity_rule']}"
        )

    high_exposure_occupations["high_complementarity"] = (
        high_exposure_occupations["complementarity_theta"]
        >= complementarity_cutoff
    ).astype(int)

    group_counts = (
        high_exposure_occupations.groupby(
            "high_complementarity",
            as_index=False,
        )
        .agg(
            occupations=("acs_occ_code", "nunique"),
            mean_exposure=("llm_exposure_main", "mean"),
            mean_complementarity=("complementarity_theta", "mean"),
        )
    )

    low_count = int(
        group_counts.loc[
            group_counts["high_complementarity"].eq(0),
            "occupations",
        ].sum()
    )
    high_count = int(
        group_counts.loc[
            group_counts["high_complementarity"].eq(1),
            "occupations",
        ].sum()
    )

    scheme_info = {
        "scheme": scheme_name,
        "exposure_cutoff": exposure_cutoff,
        "complementarity_cutoff": complementarity_cutoff,
        "high_exposure_occupations": len(high_exposure_occupations),
        "HELC_occupations": low_count,
        "HEHC_occupations": high_count,
        "minimum_group_size_ok": (
            low_count >= MIN_CATEGORY_OCCUPATIONS
            and high_count >= MIN_CATEGORY_OCCUPATIONS
        ),
        "mean_exposure_HELC": float(
            high_exposure_occupations.loc[
                high_exposure_occupations["high_complementarity"].eq(0),
                "llm_exposure_main",
            ].mean()
        ),
        "mean_exposure_HEHC": float(
            high_exposure_occupations.loc[
                high_exposure_occupations["high_complementarity"].eq(1),
                "llm_exposure_main",
            ].mean()
        ),
        "mean_complementarity_HELC": float(
            high_exposure_occupations.loc[
                high_exposure_occupations["high_complementarity"].eq(0),
                "complementarity_theta",
            ].mean()
        ),
        "mean_complementarity_HEHC": float(
            high_exposure_occupations.loc[
                high_exposure_occupations["high_complementarity"].eq(1),
                "complementarity_theta",
            ].mean()
        ),
    }

    category_map = high_exposure_occupations[
        [
            "acs_occ_code",
            "high_complementarity",
        ]
    ]

    category_panel = panel.merge(
        category_map,
        on="acs_occ_code",
        how="inner",
        validate="many_to_one",
    )

    category_panel["high_c_post"] = (
        category_panel["high_complementarity"]
        * category_panel["post"]
    )
    for year in POST_YEARS:
        category_panel[f"high_c_y{year}"] = (
            category_panel["high_complementarity"]
            * category_panel["year"].eq(year).astype(int)
        )

    return category_panel, scheme_info


def run_adjusted_category_models(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []

    for scheme_name, settings in CATEGORY_SCHEMES.items():
        category_panel, scheme_info = construct_category_scheme(
            panel=panel,
            scheme_name=scheme_name,
            settings=settings,
        )
        count_rows.append(scheme_info)

        if not scheme_info["minimum_group_size_ok"]:
            continue

        pooled_terms = [
            "exposure_z_post",
            "high_c_post",
        ]
        dynamic_terms = [
            "exposure_z_y2023",
            "exposure_z_y2024",
            "high_c_y2023",
            "high_c_y2024",
        ]

        for outcome_name, outcome_settings in OUTCOMES.items():
            outcome = outcome_settings["column"]
            weight = outcome_settings["weight"]

            _, rows = fit_fe_model(
                data=category_panel,
                outcome=outcome,
                terms=pooled_terms,
                model_name=(
                    f"{outcome_name}_{scheme_name}_"
                    "adjusted_category_pooled"
                ),
                weight_col=weight,
            )
            for row in rows:
                row["scheme"] = scheme_name
                row["outcome_name"] = outcome_name
                row["model_type"] = "pooled"
            result_rows.extend(rows)

            _, rows = fit_fe_model(
                data=category_panel,
                outcome=outcome,
                terms=dynamic_terms,
                model_name=(
                    f"{outcome_name}_{scheme_name}_"
                    "adjusted_category_dynamic"
                ),
                weight_col=weight,
            )
            for row in rows:
                row["scheme"] = scheme_name
                row["outcome_name"] = outcome_name
                row["model_type"] = "dynamic"
            result_rows.extend(rows)

    results = pd.DataFrame(result_rows)

    # Primary family: the preferred baseline category's two year-specific
    # HEHC-versus-HELC terms, separately for each outcome.
    results["fdr_family"] = "other"
    primary_mask = (
        results["scheme"].eq("baseline_within_highE_median")
        & results["model_type"].eq("dynamic")
        & results["term"].isin(["high_c_y2023", "high_c_y2024"])
    )
    results.loc[primary_mask, "fdr_family"] = "primary_adjusted_category"

    # Robustness family: all year-specific high-C terms across thresholds.
    robustness_mask = (
        results["model_type"].eq("dynamic")
        & results["term"].isin(["high_c_y2023", "high_c_y2024"])
    )
    results.loc[
        robustness_mask & ~primary_mask,
        "fdr_family",
    ] = "threshold_robustness"

    results = add_fdr(
        results,
        group_cols=["outcome_name", "fdr_family"],
    )

    return pd.DataFrame(count_rows), results


# -----------------------------------------------------------------------------
# 5. Original M robustness
# -----------------------------------------------------------------------------

def run_m_robustness(panel: pd.DataFrame) -> pd.DataFrame:
    pooled_terms = ["m_index_z_final_post"]
    dynamic_terms = [
        "m_index_z_final_y2023",
        "m_index_z_final_y2024",
    ]

    rows_all: list[dict[str, object]] = []

    for outcome_name, settings in OUTCOMES.items():
        outcome = settings["column"]
        weight = settings["weight"]

        _, rows = fit_fe_model(
            data=panel,
            outcome=outcome,
            terms=pooled_terms,
            model_name=f"{outcome_name}_M_pooled_robustness",
            weight_col=weight,
        )
        for row in rows:
            row["outcome_name"] = outcome_name
            row["model_type"] = "pooled"
        rows_all.extend(rows)

        _, rows = fit_fe_model(
            data=panel,
            outcome=outcome,
            terms=dynamic_terms,
            model_name=f"{outcome_name}_M_dynamic_robustness",
            weight_col=weight,
        )
        for row in rows:
            row["outcome_name"] = outcome_name
            row["model_type"] = "dynamic"
        rows_all.extend(rows)

    table = pd.DataFrame(rows_all)
    table = add_fdr(
        table,
        group_cols=["outcome_name", "model_type"],
    )
    return table


# -----------------------------------------------------------------------------
# 6. Figures and compact dissertation tables
# -----------------------------------------------------------------------------

def dynamic_plot(
    result_table: pd.DataFrame,
    outcome_name: str,
    filename: str,
    title: str,
) -> None:
    plot_data = result_table.loc[
        result_table["outcome_name"].eq(outcome_name)
    ].copy()

    term_details = {
        "exposure_z_y2023": ("Exposure", 2023),
        "exposure_z_y2024": ("Exposure", 2024),
        "complementarity_z_y2023": ("Complementarity", 2023),
        "complementarity_z_y2024": ("Complementarity", 2024),
        "ec_interaction_y2023": ("Interaction", 2023),
        "ec_interaction_y2024": ("Interaction", 2024),
    }

    plot_data = plot_data.loc[
        plot_data["term"].isin(term_details)
    ].copy()
    plot_data["dimension"] = plot_data["term"].map(
        lambda term: term_details[term][0]
    )
    plot_data["year_plot"] = plot_data["term"].map(
        lambda term: term_details[term][1]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for dimension, data in plot_data.groupby("dimension"):
        data = data.sort_values("year_plot")
        lower = data["coefficient"] - data["ci_low"]
        upper = data["ci_high"] - data["coefficient"]
        ax.errorbar(
            data["year_plot"],
            data["coefficient"],
            yerr=[lower, upper],
            fmt="o-",
            capsize=4,
            label=dimension,
        )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(POST_YEARS)
    ax.set_xlabel("Year")
    ax.set_ylabel("Coefficient relative to 2022")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out(filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def adjusted_category_plot(
    category_results: pd.DataFrame,
    outcome_name: str,
    filename: str,
    title: str,
) -> None:
    selection = category_results.loc[
        category_results["scheme"].eq("baseline_within_highE_median")
        & category_results["outcome_name"].eq(outcome_name)
        & category_results["model_type"].eq("dynamic")
        & category_results["term"].isin(["high_c_y2023", "high_c_y2024"])
    ].copy()

    year_lookup = {
        "high_c_y2023": 2023,
        "high_c_y2024": 2024,
    }
    selection["year_plot"] = selection["term"].map(year_lookup)
    selection = selection.sort_values("year_plot")

    lower = selection["coefficient"] - selection["ci_low"]
    upper = selection["ci_high"] - selection["coefficient"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        selection["year_plot"],
        selection["coefficient"],
        yerr=[lower, upper],
        fmt="o-",
        capsize=4,
    )
    ax.axhline(0, linewidth=1)
    ax.set_xticks(POST_YEARS)
    ax.set_xlabel("Year")
    ax.set_ylabel("Adjusted HEHC minus HELC coefficient")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out(filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_compact_main_table(
    pooled_results: pd.DataFrame,
    dynamic_results: pd.DataFrame,
    category_results: pd.DataFrame,
) -> pd.DataFrame:
    pooled = pooled_results.copy()
    pooled["analysis"] = "continuous_pooled"
    pooled["specification"] = pooled["term"]

    dynamic = dynamic_results.copy()
    dynamic["analysis"] = "continuous_dynamic"
    dynamic["specification"] = dynamic["term"]

    category = category_results.loc[
        category_results["scheme"].eq("baseline_within_highE_median")
        & category_results["term"].isin(
            ["high_c_post", "high_c_y2023", "high_c_y2024"]
        )
    ].copy()
    category["analysis"] = "exposure_adjusted_HEHC_vs_HELC"
    category["specification"] = category["term"]

    keep = [
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
    ]
    compact = pd.concat(
        [
            pooled[keep],
            dynamic[keep],
            category[keep],
        ],
        ignore_index=True,
    )
    return compact


# -----------------------------------------------------------------------------
# 7. Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("\nSTEP 3: FINAL DISSERTATION ANALYSIS\n")

    panel, data_check = load_panel()
    data_check.to_csv(out("data_check.csv"), index=False)

    pooled_results, dynamic_results = run_continuous_models(panel)
    pooled_results.to_csv(
        out("preferred_pooled_results.csv"),
        index=False,
    )
    dynamic_results.to_csv(
        out("preferred_dynamic_results.csv"),
        index=False,
    )

    category_counts, category_results = run_adjusted_category_models(panel)
    category_counts.to_csv(
        out("adjusted_category_counts.csv"),
        index=False,
    )
    category_results.to_csv(
        out("adjusted_category_results.csv"),
        index=False,
    )

    m_results = run_m_robustness(panel)
    m_results.to_csv(
        out("M_robustness_results.csv"),
        index=False,
    )

    compact_table = build_compact_main_table(
        pooled_results=pooled_results,
        dynamic_results=dynamic_results,
        category_results=category_results,
    )
    compact_table.to_csv(
        out("dissertation_main_table.csv"),
        index=False,
    )

    for outcome_name, settings in OUTCOMES.items():
        dynamic_plot(
            result_table=dynamic_results,
            outcome_name=outcome_name,
            filename=f"{outcome_name}_preferred_dynamic.png",
            title=(
                "Exposure, complementarity and interaction: "
                + settings["label"].lower()
            ),
        )
        adjusted_category_plot(
            category_results=category_results,
            outcome_name=outcome_name,
            filename=f"{outcome_name}_adjusted_HEHC_HELC.png",
            title=(
                "Exposure-adjusted HEHC versus HELC: "
                + settings["label"].lower()
            ),
        )

    print("Completed. Review these files first:")
    print("  1. out_step3_data_check.csv")
    print("  2. out_step3_preferred_pooled_results.csv")
    print("  3. out_step3_preferred_dynamic_results.csv")
    print("  4. out_step3_adjusted_category_counts.csv")
    print("  5. out_step3_adjusted_category_results.csv")
    print("  6. out_step3_M_robustness_results.csv")
    print("  7. out_step3_dissertation_main_table.csv")
    print("  8. out_step3_*_preferred_dynamic.png")
    print("  9. out_step3_*_adjusted_HEHC_HELC.png")


if __name__ == "__main__":
    main()