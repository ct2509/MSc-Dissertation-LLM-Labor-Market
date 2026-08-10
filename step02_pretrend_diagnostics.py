"""Step 2: Exposure–complementarity framework.

Purpose
-------
Use the balanced occupation-year panel produced by step01_build_panel.py
to examine post-2022 wage-income and employment changes along two distinct
occupation-level dimensions:

    E = LLM exposure
    C = occupational complementarity

The script treats the original product M = E x C as a secondary robustness
measure rather than the sole main variable.

Main analyses
-------------
1. Continuous decomposition:
       E x Post, C x Post, and (standardised E x standardised C) x Post.
2. Pizzinelli-style occupation categories:
       LE   = low exposure
       HELC = high exposure, low complementarity
       HEHC = high exposure, high complementarity
3. Extended-window and short-window specifications.
4. Occupation and year fixed effects, with SOC-major-by-year fixed effects
   as a stricter specification.
5. Full-window event studies and pre-trend tests.
6. M-only models retained as robustness checks.

All inputs and outputs remain in the same folder as this script.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# 0. Settings
# -----------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent
PANEL_FILE = BASE / "out_occupation_year_panel_balanced.csv"

FULL_YEARS = [2019, 2021, 2022, 2023, 2024]
SHORT_YEARS = [2022, 2023, 2024]
REFERENCE_YEAR = 2022
PRE_YEARS = [2019, 2021]

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
}


def out(name: str) -> Path:
    return BASE / f"out_step2_{name}"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    sd = values.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - values.mean()) / sd


# -----------------------------------------------------------------------------
# 1. Load the Step-1 panel and construct the new framework
# -----------------------------------------------------------------------------

def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"{PANEL_FILE.name} was not found. Run step01_build_panel.py first."
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
        "baseline_wage_weight",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise KeyError(
            "The Step-1 panel is missing required columns: " + str(missing)
        )

    panel = panel.copy()
    panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
    panel["acs_occ_code"] = panel["acs_occ_code"].astype("string")
    panel["soc_major"] = panel["soc_major"].astype("string")
    panel["post"] = (panel["year"] >= 2023).astype(int)

    # Standardise scores across unique occupations, not occupation-year rows.
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
    occupation_scores["m_index_z_step2"] = zscore(
        occupation_scores["m_index"]
    )

    # Pizzinelli-style classification using occupation medians.
    exposure_cutoff = float(
        occupation_scores["llm_exposure_main"].median()
    )
    complementarity_cutoff = float(
        occupation_scores["complementarity_theta"].median()
    )

    occupation_scores["high_exposure"] = (
        occupation_scores["llm_exposure_main"] >= exposure_cutoff
    ).astype(int)
    occupation_scores["high_complementarity"] = (
        occupation_scores["complementarity_theta"]
        >= complementarity_cutoff
    ).astype(int)

    occupation_scores["ai_category"] = np.select(
        [
            occupation_scores["high_exposure"].eq(0),
            occupation_scores["high_exposure"].eq(1)
            & occupation_scores["high_complementarity"].eq(0),
            occupation_scores["high_exposure"].eq(1)
            & occupation_scores["high_complementarity"].eq(1),
        ],
        ["LE", "HELC", "HEHC"],
        default="unclassified",
    )

    occupation_scores["helc"] = (
        occupation_scores["ai_category"] == "HELC"
    ).astype(int)
    occupation_scores["hehc"] = (
        occupation_scores["ai_category"] == "HEHC"
    ).astype(int)

    # Remove old versions if the panel already contains similarly named fields.
    replace_cols = [
        "exposure_z",
        "complementarity_z",
        "ec_interaction",
        "m_index_z_step2",
        "high_exposure",
        "high_complementarity",
        "ai_category",
        "helc",
        "hehc",
    ]
    panel = panel.drop(
        columns=[c for c in replace_cols if c in panel.columns],
        errors="ignore",
    )
    panel = panel.merge(
        occupation_scores[
            ["acs_occ_code", *replace_cols]
        ],
        on="acs_occ_code",
        how="left",
        validate="many_to_one",
    )

    for score in [
        "exposure_z",
        "complementarity_z",
        "ec_interaction",
        "m_index_z_step2",
        "helc",
        "hehc",
    ]:
        panel[f"{score}_post"] = panel[score] * panel["post"]

    cutoffs = pd.DataFrame(
        {
            "measure": [
                "exposure_median",
                "complementarity_median",
            ],
            "cutoff": [
                exposure_cutoff,
                complementarity_cutoff,
            ],
        }
    )

    category_counts = (
        occupation_scores.groupby("ai_category", as_index=False)
        .agg(
            occupations=("acs_occ_code", "nunique"),
            mean_exposure=("llm_exposure_main", "mean"),
            mean_complementarity=("complementarity_theta", "mean"),
            mean_M=("m_index", "mean"),
        )
        .sort_values("ai_category")
    )

    return panel, cutoffs, category_counts


# -----------------------------------------------------------------------------
# 2. Fixed-effects models
# -----------------------------------------------------------------------------

def add_soc_year_deviations(
    data: pd.DataFrame,
    fe_type: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add a full-rank set of SOC-major-group-by-year deviations.

    Occupation fixed effects absorb permanent occupation differences.
    Year fixed effects absorb aggregate annual shocks.

    For the stricter SOC-year specification, one SOC major group and
    the reference year 2022 are omitted to avoid redundant indicators.
    """
    data = data.copy()

    if fe_type == "basic":
        return data, []

    if fe_type != "soc_year":
        raise ValueError(f"Unknown fe_type: {fe_type}")

    data["soc_major"] = data["soc_major"].astype("string")

    groups = sorted(
        data["soc_major"].dropna().unique().tolist()
    )

    years = sorted(
        int(year)
        for year in data["year"].dropna().unique()
        if int(year) != REFERENCE_YEAR
    )

    if len(groups) < 2:
        raise ValueError(
            "At least two SOC major groups are required."
        )

    reference_group = groups[0]

    fe_cols: list[str] = []

    for group in groups[1:]:
        safe_group = (
            str(group)
            .replace("-", "_")
            .replace(".", "_")
            .replace(" ", "_")
        )

        for year in years:
            col = f"socdev_{safe_group}_{year}"

            data[col] = (
                (data["soc_major"] == group)
                & (data["year"] == year)
            ).astype(int)

            fe_cols.append(col)

    return data, fe_cols


def fixed_effects(
    soc_year_cols: list[str],
) -> str:
    base = "C(acs_occ_code) + C(year)"

    if not soc_year_cols:
        return base

    return base + " + " + " + ".join(soc_year_cols)


def fit_model(
    panel: pd.DataFrame,
    outcome: str,
    terms: Sequence[str],
    model_name: str,
    years: Sequence[int],
    fe_type: str,
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

    data = (
        panel.loc[panel["year"].isin(years), required_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )



    if weight_col:
        data = data.loc[data[weight_col] > 0].copy()

    data, soc_year_cols = add_soc_year_deviations(
        data,
        fe_type,
    )

    formula = (
        f"{outcome} ~ "
        + " + ".join(terms)
        + " + "
        + fixed_effects(soc_year_cols)
    )

    if weight_col:
        model = smf.wls(
            formula,
            data=data,
            weights=data[weight_col],
        )
    else:
        model = smf.ols(formula, data=data)

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["acs_occ_code"]},
        use_t=True,
    )

    out(f"{safe_name(model_name)}.txt").write_text(
        result.summary().as_text(),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for term in terms:
        ci = result.conf_int().loc[term]
        rows.append(
            {
                "model": model_name,
                "window_years": ",".join(map(str, years)),
                "fe_type": fe_type,
                "outcome": outcome,
                "term": term,
                "coefficient": result.params[term],
                "standard_error": result.bse[term],
                "p_value": result.pvalues[term],
                "ci_low": ci.iloc[0],
                "ci_high": ci.iloc[1],
                "occupation_year_cells": len(data),
                "occupation_clusters": data["acs_occ_code"].nunique(),
                "r_squared": result.rsquared,
                "weight_col": weight_col or "none",
            }
        )

    return result, rows


def linear_contrast(
    result: object,
    positive_term: str,
    negative_term: str,
    label: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    params = result.params
    covariance = result.cov_params()

    estimate = params[positive_term] - params[negative_term]
    variance = (
        covariance.loc[positive_term, positive_term]
        + covariance.loc[negative_term, negative_term]
        - 2 * covariance.loc[positive_term, negative_term]
    )
    standard_error = float(np.sqrt(max(variance, 0.0)))
    if standard_error > 0:
        t_value = float(estimate / standard_error)
        # Use the fitted model's t distribution through t_test.
        test = result.t_test(
            f"{positive_term} - {negative_term} = 0"
        )
        p_value = float(np.asarray(test.pvalue).squeeze())
        ci = np.asarray(test.conf_int()).reshape(-1)
        ci_low, ci_high = float(ci[0]), float(ci[1])
    else:
        t_value = np.nan
        p_value = np.nan
        ci_low = np.nan
        ci_high = np.nan

    return {
        **metadata,
        "contrast": label,
        "coefficient": float(estimate),
        "standard_error": standard_error,
        "t_value": t_value,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# -----------------------------------------------------------------------------
# 3. Event studies
# -----------------------------------------------------------------------------

def fit_joint_continuous_event_study(
    panel: pd.DataFrame,
    outcome: str,
    model_name: str,
    fe_type: str,
    weight_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = panel.copy()
    dimensions = {
        "exposure": "exposure_z",
        "complementarity": "complementarity_z",
        "interaction": "ec_interaction",
    }

    event_terms: list[str] = []
    term_lookup: dict[tuple[str, int], str] = {}

    for dimension, score_col in dimensions.items():
        for year in FULL_YEARS:
            if year == REFERENCE_YEAR:
                continue
            term = f"event_{dimension}_{year}"
            data[term] = (
                data[score_col]
                * (data["year"] == year).astype(int)
            )
            event_terms.append(term)
            term_lookup[(dimension, year)] = term

    required_cols = [
        outcome,
        "acs_occ_code",
        "soc_major",
        "year",
        *event_terms,
    ]
    if weight_col:
        required_cols.append(weight_col)

    data = (
        data.loc[data["year"].isin(FULL_YEARS), required_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    if weight_col:
        data = data.loc[data[weight_col] > 0].copy()

    data, soc_year_cols = add_soc_year_deviations(
        data,
        fe_type,
    )

    formula = (
        f"{outcome} ~ "
        + " + ".join(event_terms)
        + " + "
        + fixed_effects(soc_year_cols)
    )

    if weight_col:
        model = smf.wls(
            formula,
            data=data,
            weights=data[weight_col],
        )
    else:
        model = smf.ols(formula, data=data)

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["acs_occ_code"]},
        use_t=True,
    )
    out(f"{safe_name(model_name)}.txt").write_text(
        result.summary().as_text(),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    pretrend_rows: list[dict[str, object]] = []

    for dimension in dimensions:
        for year in FULL_YEARS:
            if year == REFERENCE_YEAR:
                rows.append(
                    {
                        "model": model_name,
                        "fe_type": fe_type,
                        "outcome": outcome,
                        "dimension": dimension,
                        "year": year,
                        "reference_year": REFERENCE_YEAR,
                        "coefficient": 0.0,
                        "standard_error": 0.0,
                        "p_value": np.nan,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                    }
                )
                continue

            term = term_lookup[(dimension, year)]
            ci = result.conf_int().loc[term]
            rows.append(
                {
                    "model": model_name,
                    "fe_type": fe_type,
                    "outcome": outcome,
                    "dimension": dimension,
                    "year": year,
                    "reference_year": REFERENCE_YEAR,
                    "coefficient": result.params[term],
                    "standard_error": result.bse[term],
                    "p_value": result.pvalues[term],
                    "ci_low": ci.iloc[0],
                    "ci_high": ci.iloc[1],
                }
            )

        pre_terms = [
            term_lookup[(dimension, year)]
            for year in PRE_YEARS
        ]
        restriction = ", ".join(f"{term} = 0" for term in pre_terms)
        test = result.wald_test(restriction, scalar=True)
        pretrend_rows.append(
            {
                "model": model_name,
                "fe_type": fe_type,
                "outcome": outcome,
                "dimension": dimension,
                "tested_pre_years": ",".join(map(str, PRE_YEARS)),
                "reference_year": REFERENCE_YEAR,
                "joint_pretrend_p_value": float(
                    np.asarray(test.pvalue).squeeze()
                ),
                "occupation_clusters": data["acs_occ_code"].nunique(),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(pretrend_rows)


def fit_category_event_study(
    panel: pd.DataFrame,
    outcome: str,
    model_name: str,
    fe_type: str,
    weight_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = panel.copy()
    event_terms: list[str] = []
    lookup: dict[tuple[str, int], str] = {}

    for group, dummy in {"HELC": "helc", "HEHC": "hehc"}.items():
        for year in FULL_YEARS:
            if year == REFERENCE_YEAR:
                continue
            term = f"event_{group.lower()}_{year}"
            data[term] = (
                data[dummy]
                * (data["year"] == year).astype(int)
            )
            event_terms.append(term)
            lookup[(group, year)] = term

    required_cols = [
        outcome,
        "acs_occ_code",
        "soc_major",
        "year",
        *event_terms,
    ]
    if weight_col:
        required_cols.append(weight_col)

    data = (
        data.loc[data["year"].isin(FULL_YEARS), required_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    if weight_col:
        data = data.loc[data[weight_col] > 0].copy()

    data, soc_year_cols = add_soc_year_deviations(
        data,
        fe_type,
    )

    formula = (
        f"{outcome} ~ "
        + " + ".join(event_terms)
        + " + "
        + fixed_effects(soc_year_cols)
    )

    if weight_col:
        model = smf.wls(
            formula,
            data=data,
            weights=data[weight_col],
        )
    else:
        model = smf.ols(formula, data=data)

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["acs_occ_code"]},
        use_t=True,
    )
    out(f"{safe_name(model_name)}.txt").write_text(
        result.summary().as_text(),
        encoding="utf-8",
    )

    covariance = result.cov_params()
    rows: list[dict[str, object]] = []
    pretrend_rows: list[dict[str, object]] = []

    for year in FULL_YEARS:
        if year == REFERENCE_YEAR:
            rows.append(
                {
                    "model": model_name,
                    "fe_type": fe_type,
                    "outcome": outcome,
                    "year": year,
                    "reference_year": REFERENCE_YEAR,
                    "contrast": "HEHC_minus_HELC",
                    "coefficient": 0.0,
                    "standard_error": 0.0,
                    "p_value": np.nan,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                }
            )
            continue

        positive = lookup[("HEHC", year)]
        negative = lookup[("HELC", year)]
        estimate = result.params[positive] - result.params[negative]
        variance = (
            covariance.loc[positive, positive]
            + covariance.loc[negative, negative]
            - 2 * covariance.loc[positive, negative]
        )
        se = float(np.sqrt(max(variance, 0.0)))
        test = result.t_test(f"{positive} - {negative} = 0")
        ci = np.asarray(test.conf_int()).reshape(-1)

        rows.append(
            {
                "model": model_name,
                "fe_type": fe_type,
                "outcome": outcome,
                "year": year,
                "reference_year": REFERENCE_YEAR,
                "contrast": "HEHC_minus_HELC",
                "coefficient": float(estimate),
                "standard_error": se,
                "p_value": float(np.asarray(test.pvalue).squeeze()),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
            }
        )

    # Joint pre-trend test for the HEHC minus HELC contrast.
    restrictions = [
        (
            f"{lookup[('HEHC', year)]} "
            f"- {lookup[('HELC', year)]} = 0"
        )
        for year in PRE_YEARS
    ]
    test = result.wald_test(", ".join(restrictions), scalar=True)
    pretrend_rows.append(
        {
            "model": model_name,
            "fe_type": fe_type,
            "outcome": outcome,
            "dimension": "HEHC_minus_HELC",
            "tested_pre_years": ",".join(map(str, PRE_YEARS)),
            "reference_year": REFERENCE_YEAR,
            "joint_pretrend_p_value": float(
                np.asarray(test.pvalue).squeeze()
            ),
            "occupation_clusters": data["acs_occ_code"].nunique(),
        }
    )

    return pd.DataFrame(rows), pd.DataFrame(pretrend_rows)


# -----------------------------------------------------------------------------
# 4. Figures
# -----------------------------------------------------------------------------

def event_figure(
    event_table: pd.DataFrame,
    filename: str,
    title: str,
    ylabel: str,
) -> None:
    plot = event_table.sort_values("year").copy()
    lower = plot["coefficient"] - plot["ci_low"]
    upper = plot["ci_high"] - plot["coefficient"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        plot["year"],
        plot["coefficient"],
        yerr=[lower, upper],
        fmt="o-",
        capsize=4,
    )
    ax.axhline(0, linewidth=1)
    ax.axvline(2022.5, linestyle="--", linewidth=1)
    ax.set_xticks(FULL_YEARS)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out(filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Run all specifications
# -----------------------------------------------------------------------------

def main() -> None:
    print("\nSTEP 2: EXPOSURE–COMPLEMENTARITY FRAMEWORK\n")

    panel, cutoffs, category_counts = load_panel()
    cutoffs.to_csv(out("category_cutoffs.csv"), index=False)
    category_counts.to_csv(out("category_counts.csv"), index=False)

    windows = {
        "extended": FULL_YEARS,
        "short": SHORT_YEARS,
    }
    fe_types = ["basic", "soc_year"]

    continuous_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    m_rows: list[dict[str, object]] = []

    # A. Continuous E, C, and E x C decomposition.
    continuous_specs = {
        "joint": [
            "exposure_z_post",
            "complementarity_z_post",
            "ec_interaction_post",
        ],
        "exposure_only": ["exposure_z_post"],
        "complementarity_only": ["complementarity_z_post"],
    }

    for window_name, years in windows.items():
        for fe_type in fe_types:
            for outcome_name, settings in OUTCOMES.items():
                outcome = settings["column"]
                weight = settings["weight"]

                for spec_name, terms in continuous_specs.items():
                    name = (
                        f"{outcome_name}_{window_name}_"
                        f"{fe_type}_{spec_name}"
                    )
                    _, rows = fit_model(
                        panel=panel,
                        outcome=outcome,
                        terms=terms,
                        model_name=name,
                        years=years,
                        fe_type=fe_type,
                        weight_col=weight,
                    )
                    continuous_rows.extend(rows)

                # B. Pizzinelli-style categories, with LE as reference.
                category_name = (
                    f"{outcome_name}_{window_name}_"
                    f"{fe_type}_categories"
                )
                category_result, rows = fit_model(
                    panel=panel,
                    outcome=outcome,
                    terms=["helc_post", "hehc_post"],
                    model_name=category_name,
                    years=years,
                    fe_type=fe_type,
                    weight_col=weight,
                )
                category_rows.extend(rows)

                contrast_rows.append(
                    linear_contrast(
                        result=category_result,
                        positive_term="hehc_post",
                        negative_term="helc_post",
                        label="HEHC_minus_HELC",
                        metadata={
                            "model": category_name,
                            "window_years": ",".join(map(str, years)),
                            "fe_type": fe_type,
                            "outcome": outcome,
                        },
                    )
                )

                # C. Original M retained only as robustness.
                m_name = (
                    f"{outcome_name}_{window_name}_"
                    f"{fe_type}_M_robustness"
                )
                _, rows = fit_model(
                    panel=panel,
                    outcome=outcome,
                    terms=["m_index_z_step2_post"],
                    model_name=m_name,
                    years=years,
                    fe_type=fe_type,
                    weight_col=weight,
                )
                m_rows.extend(rows)

    pd.DataFrame(continuous_rows).to_csv(
        out("continuous_models.csv"),
        index=False,
    )
    pd.DataFrame(category_rows).to_csv(
        out("category_models.csv"),
        index=False,
    )
    pd.DataFrame(contrast_rows).to_csv(
        out("category_contrasts.csv"),
        index=False,
    )
    pd.DataFrame(m_rows).to_csv(
        out("M_robustness_models.csv"),
        index=False,
    )

    # D. Full-window event studies. Run both FE structures.
    continuous_events = []
    category_events = []
    pretrend_tables = []

    for fe_type in fe_types:
        for outcome_name, settings in OUTCOMES.items():
            outcome = settings["column"]
            weight = settings["weight"]

            continuous_event, continuous_pre = (
                fit_joint_continuous_event_study(
                    panel=panel,
                    outcome=outcome,
                    model_name=(
                        f"{outcome_name}_{fe_type}_"
                        "continuous_event_study"
                    ),
                    fe_type=fe_type,
                    weight_col=weight,
                )
            )
            continuous_events.append(continuous_event)
            pretrend_tables.append(continuous_pre)

            category_event, category_pre = fit_category_event_study(
                panel=panel,
                outcome=outcome,
                model_name=(
                    f"{outcome_name}_{fe_type}_"
                    "category_event_study"
                ),
                fe_type=fe_type,
                weight_col=weight,
            )
            category_events.append(category_event)
            pretrend_tables.append(category_pre)

    continuous_event_table = pd.concat(
        continuous_events,
        ignore_index=True,
    )
    category_event_table = pd.concat(
        category_events,
        ignore_index=True,
    )
    pretrend_table = pd.concat(
        pretrend_tables,
        ignore_index=True,
    )

    continuous_event_table.to_csv(
        out("continuous_event_study.csv"),
        index=False,
    )
    category_event_table.to_csv(
        out("category_event_study.csv"),
        index=False,
    )
    pretrend_table.to_csv(
        out("pretrend_tests.csv"),
        index=False,
    )

    # Main figures use the stricter SOC-major-by-year event specification.
    for outcome_name, settings in OUTCOMES.items():
        outcome = settings["column"]
        label = settings["label"]

        for dimension in [
            "exposure",
            "complementarity",
            "interaction",
        ]:
            selection = continuous_event_table.loc[
                (continuous_event_table["fe_type"] == "soc_year")
                & (continuous_event_table["outcome"] == outcome)
                & (continuous_event_table["dimension"] == dimension)
            ].copy()
            event_figure(
                selection,
                (
                    f"{outcome_name}_{dimension}_"
                    "event_study.png"
                ),
                (
                    f"{dimension.title()} and changes in "
                    f"{label.lower()}"
                ),
                f"Coefficient on {dimension} × year",
            )

        category_selection = category_event_table.loc[
            (category_event_table["fe_type"] == "soc_year")
            & (category_event_table["outcome"] == outcome)
        ].copy()
        event_figure(
            category_selection,
            f"{outcome_name}_HEHC_minus_HELC_event_study.png",
            f"HEHC versus HELC: changes in {label.lower()}",
            "HEHC minus HELC coefficient",
        )

    print("Completed. Review these outputs first:")
    print("  1. out_step2_category_counts.csv")
    print("  2. out_step2_pretrend_tests.csv")
    print("  3. out_step2_continuous_models.csv")
    print("  4. out_step2_category_contrasts.csv")
    print("  5. out_step2_M_robustness_models.csv")
    print("  6. out_step2_*_event_study.png")


if __name__ == "__main__":
    main()