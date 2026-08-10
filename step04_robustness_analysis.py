"""
Step 04: Alternative-measure robustness analysis.

This script re-estimates the preferred 2022-2024 specifications using:

1. GPT/model-rated LLM exposure
2. Complementarity excluding Job Zones
3. OEWS-employment-weighted occupational mapping

All robustness models use the same empirical design as the preferred
Step-03 analysis:

- 2022 reference year
- 2023 and 2024 post period
- occupation fixed effects
- year fixed effects
- full-rank SOC-major-group-by-year deviations
- fixed 2022 wage weights
- unweighted employment regressions
- occupation-clustered standard errors

Inputs:
    out_occupation_year_panel_balanced.csv
    occupation_master_acs.csv

Outputs:
    out_step4_robustness_pooled.csv
    out_step4_robustness_dynamic.csv
    out_step4_robustness_joint_tests.csv
    out_step4_robustness_summary.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# =============================================================================
# 0. SETTINGS
# =============================================================================

BASE = Path(__file__).resolve().parent

PANEL_FILE = BASE / "out_occupation_year_panel_balanced.csv"
MASTER_FILE = BASE / "occupation_master_acs.csv"

ANALYSIS_YEARS = [2022, 2023, 2024]
REFERENCE_YEAR = 2022
POST_YEARS = [2023, 2024]

WAGE_OUTCOME = "mean_log_real_wage_2024"
EMPLOYMENT_OUTCOME = "log_employment"
WAGE_WEIGHT = "baseline_wage_weight"


ROBUSTNESS_SPECS = {
    "model_rated_exposure": {
        "label": "GPT/model-rated exposure",
        "exposure": "llm_exposure_model_robustness",
        "complementarity": "complementarity_theta",
    },
    "no_jobzone_complementarity": {
        "label": "Complementarity excluding Job Zones",
        "exposure": "llm_exposure_main",
        "complementarity": "complementarity_no_jobzone",
    },
    "oews_weighted_mapping": {
        "label": "OEWS-employment-weighted mapping",
        "exposure": "llm_exposure_main_oews_weighted",
        "complementarity": "complementarity_theta_oews_weighted",
    },
}


# =============================================================================
# 1. HELPERS
# =============================================================================

def out(name: str) -> Path:
    return BASE / name


def normalise_occ(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "n/a"}:
        return None

    try:
        return str(int(float(text)))
    except ValueError:
        digits = re.sub(r"\D", "", text)
        return str(int(digits)) if digits else None


def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    sd = series.std(ddof=0)

    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(
            "Cannot standardise a variable with zero or invalid "
            "standard deviation."
        )

    return (series - mean) / sd


def add_soc_year_deviations(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add full-rank SOC-major-group-by-year deviations.

    Occupation fixed effects absorb permanent occupation differences.
    Year fixed effects absorb aggregate year shocks.

    Therefore only non-reference SOC groups interacted with post years
    are required.

    2022 is the reference year.
    The first SOC major group is the reference group.
    """
    data = data.copy()

    data["soc_major"] = data["soc_major"].astype("string")

    groups = sorted(
        data["soc_major"].dropna().unique().tolist()
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

        for year in POST_YEARS:
            col = f"socdev_{safe_group}_{year}"

            data[col] = (
                (data["soc_major"] == group)
                & (data["year"] == year)
            ).astype(int)

            fe_cols.append(col)

    return data, fe_cols


def fixed_effects(soc_year_cols: list[str]) -> str:
    base = "C(acs_occ_code) + C(year)"

    if not soc_year_cols:
        return base

    return base + " + " + " + ".join(soc_year_cols)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

def load_data() -> pd.DataFrame:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"{PANEL_FILE.name} was not found."
        )

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"{MASTER_FILE.name} was not found."
        )

    panel = pd.read_csv(
        PANEL_FILE,
        dtype={"acs_occ_code": "string"},
        low_memory=False,
    )

    master = pd.read_csv(
        MASTER_FILE,
        dtype={"acs_occ_code": "string"},
        low_memory=False,
    )

    panel["acs_occ_code"] = panel["acs_occ_code"].map(
        normalise_occ
    )
    master["acs_occ_code"] = master["acs_occ_code"].map(
        normalise_occ
    )

    master = master.dropna(
        subset=["acs_occ_code"]
    ).copy()

    if master["acs_occ_code"].duplicated().any():
        duplicates = master.loc[
            master["acs_occ_code"].duplicated(False),
            "acs_occ_code",
        ].unique()

        raise ValueError(
            "occupation_master_acs.csv contains duplicate "
            f"ACS occupation codes: {duplicates[:10]}"
        )

    required_master_cols = {
        "acs_occ_code",
        "llm_exposure_main",
        "complementarity_theta",
        "llm_exposure_model_robustness",
        "complementarity_no_jobzone",
        "llm_exposure_main_oews_weighted",
        "complementarity_theta_oews_weighted",
    }

    missing = required_master_cols.difference(
        master.columns
    )

    if missing:
        raise KeyError(
            "occupation_master_acs.csv is missing: "
            + str(sorted(missing))
        )

    alt_cols = sorted(
        required_master_cols.difference(
            {"acs_occ_code"}
        )
    )

    # Avoid duplicate baseline columns already present in the panel.
    drop_from_panel = [
        col
        for col in alt_cols
        if col in panel.columns
    ]

    if drop_from_panel:
        panel = panel.drop(
            columns=drop_from_panel
        )

    panel = panel.merge(
        master[
            ["acs_occ_code", *alt_cols]
        ],
        on="acs_occ_code",
        how="left",
        validate="many_to_one",
    )

    panel = panel.loc[
        panel["year"].isin(ANALYSIS_YEARS)
    ].copy()

    panel["post"] = panel["year"].isin(
        POST_YEARS
    ).astype(int)

    print("=" * 72)
    print("ROBUSTNESS DATA CHECK")
    print("=" * 72)
    print("Occupation-year cells:", len(panel))
    print(
        "Occupations:",
        panel["acs_occ_code"].nunique(),
    )
    print(
        "Years:",
        sorted(panel["year"].unique()),
    )

    return panel


# =============================================================================
# 3. PREPARE ALTERNATIVE SCORES
# =============================================================================

def prepare_specification(
    panel: pd.DataFrame,
    exposure_col: str,
    complementarity_col: str,
) -> pd.DataFrame:
    """
    Standardise exposure and complementarity once across unique
    occupations, then merge the standardised scores back to all years.
    """
    data = panel.copy()

    occupation_scores = (
        data[
            [
                "acs_occ_code",
                exposure_col,
                complementarity_col,
            ]
        ]
        .drop_duplicates(
            subset=["acs_occ_code"]
        )
        .copy()
    )

    occupation_scores[exposure_col] = pd.to_numeric(
        occupation_scores[exposure_col],
        errors="coerce",
    )

    occupation_scores[complementarity_col] = pd.to_numeric(
        occupation_scores[complementarity_col],
        errors="coerce",
    )

    occupation_scores = occupation_scores.dropna(
        subset=[
            exposure_col,
            complementarity_col,
        ]
    ).copy()

    occupation_scores["robust_exposure_z"] = zscore(
        occupation_scores[exposure_col]
    )

    occupation_scores["robust_complementarity_z"] = zscore(
        occupation_scores[complementarity_col]
    )

    occupation_scores["robust_interaction"] = (
        occupation_scores["robust_exposure_z"]
        * occupation_scores["robust_complementarity_z"]
    )

    keep = occupation_scores[
        [
            "acs_occ_code",
            "robust_exposure_z",
            "robust_complementarity_z",
            "robust_interaction",
        ]
    ]

    data = data.merge(
        keep,
        on="acs_occ_code",
        how="inner",
        validate="many_to_one",
    )

    # Pooled terms
    data["robust_exposure_post"] = (
        data["robust_exposure_z"]
        * data["post"]
    )

    data["robust_complementarity_post"] = (
        data["robust_complementarity_z"]
        * data["post"]
    )

    data["robust_interaction_post"] = (
        data["robust_interaction"]
        * data["post"]
    )

    # Dynamic terms
    for year in POST_YEARS:
        year_indicator = (
            data["year"] == year
        ).astype(int)

        data[f"robust_exposure_y{year}"] = (
            data["robust_exposure_z"]
            * year_indicator
        )

        data[f"robust_complementarity_y{year}"] = (
            data["robust_complementarity_z"]
            * year_indicator
        )

        data[f"robust_interaction_y{year}"] = (
            data["robust_interaction"]
            * year_indicator
        )

    return data


# =============================================================================
# 4. ESTIMATION
# =============================================================================

def fit_model(
    data: pd.DataFrame,
    outcome: str,
    terms: Sequence[str],
    robustness_name: str,
    robustness_label: str,
    model_type: str,
    weight_col: Optional[str] = None,
) -> tuple[object, pd.DataFrame]:

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
        model_data = model_data.loc[
            model_data[weight_col] > 0
        ].copy()

    model_data, soc_year_cols = (
        add_soc_year_deviations(model_data)
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
            data=model_data,
            weights=model_data[weight_col],
        )
    else:
        model = smf.ols(
            formula,
            data=model_data,
        )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": model_data["acs_occ_code"]
        },
        use_t=True,
    )

    rows = []

    for term in terms:
        if term not in result.params.index:
            continue

        ci = result.conf_int().loc[term]

        rows.append(
            {
                "robustness": robustness_name,
                "robustness_label": robustness_label,
                "model_type": model_type,
                "outcome": outcome,
                "term": term,
                "coefficient": result.params[term],
                "standard_error": result.bse[term],
                "p_value": result.pvalues[term],
                "ci_low": ci.iloc[0],
                "ci_high": ci.iloc[1],
                "occupation_year_cells": len(
                    model_data
                ),
                "occupation_clusters": (
                    model_data[
                        "acs_occ_code"
                    ].nunique()
                ),
                "r_squared": result.rsquared,
                "weight_col": (
                    weight_col
                    if weight_col
                    else "none"
                ),
            }
        )

    return result, pd.DataFrame(rows)


# =============================================================================
# 5. FDR ADJUSTMENT
# =============================================================================

def add_fdr(
    table: pd.DataFrame,
) -> pd.DataFrame:
    table = table.copy()

    table["q_value_bh"] = np.nan
    table["reject_fdr_5pct"] = False

    group_cols = [
        "robustness",
        "outcome",
        "model_type",
    ]

    for _, index in table.groupby(
        group_cols,
        dropna=False,
    ).groups.items():

        index = list(index)

        pvalues = table.loc[
            index,
            "p_value",
        ].astype(float)

        reject, qvalues, _, _ = multipletests(
            pvalues,
            alpha=0.05,
            method="fdr_bh",
        )

        table.loc[
            index,
            "q_value_bh",
        ] = qvalues

        table.loc[
            index,
            "reject_fdr_5pct",
        ] = reject

    return table


# =============================================================================
# 6. JOINT TESTS
# =============================================================================

def joint_test(
    result: object,
    terms: Sequence[str],
    robustness_name: str,
    robustness_label: str,
    outcome: str,
    dimension: str,
) -> dict[str, object]:

    available = [
        term
        for term in terms
        if term in result.params.index
    ]

    if len(available) != len(terms):
        raise ValueError(
            "Not all requested terms are available "
            "for the joint test."
        )

    hypothesis = ", ".join(
        f"{term} = 0"
        for term in available
    )

    test = result.f_test(hypothesis)

    return {
        "robustness": robustness_name,
        "robustness_label": robustness_label,
        "outcome": outcome,
        "dimension": dimension,
        "tested_terms": "; ".join(available),
        "number_of_restrictions": len(available),
        "test_statistic": float(
            np.asarray(test.fvalue).squeeze()
        ),
        "p_value": float(
            np.asarray(test.pvalue).squeeze()
        ),
    }


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def main() -> None:

    panel = load_data()

    pooled_tables = []
    dynamic_tables = []
    joint_rows = []
    summary_rows = []

    for robustness_name, spec in ROBUSTNESS_SPECS.items():

        print("\n" + "=" * 72)
        print(spec["label"].upper())
        print("=" * 72)

        data = prepare_specification(
            panel=panel,
            exposure_col=spec["exposure"],
            complementarity_col=spec[
                "complementarity"
            ],
        )

        occupation_count = (
            data["acs_occ_code"].nunique()
        )

        print(
            "Occupations with complete scores:",
            occupation_count,
        )

        if occupation_count != 498:
            print(
                "WARNING: robustness sample does not "
                "contain all 498 occupations."
            )

        pooled_terms = [
            "robust_exposure_post",
            "robust_complementarity_post",
            "robust_interaction_post",
        ]

        dynamic_terms = [
            "robust_exposure_y2023",
            "robust_exposure_y2024",
            "robust_complementarity_y2023",
            "robust_complementarity_y2024",
            "robust_interaction_y2023",
            "robust_interaction_y2024",
        ]

        for outcome, weight_col in [
            (WAGE_OUTCOME, WAGE_WEIGHT),
            (EMPLOYMENT_OUTCOME, None),
        ]:

            # -------------------------------------------------------------
            # Pooled
            # -------------------------------------------------------------

            pooled_result, pooled_table = fit_model(
                data=data,
                outcome=outcome,
                terms=pooled_terms,
                robustness_name=robustness_name,
                robustness_label=spec["label"],
                model_type="pooled",
                weight_col=weight_col,
            )

            pooled_tables.append(
                pooled_table
            )

            # -------------------------------------------------------------
            # Dynamic
            # -------------------------------------------------------------

            dynamic_result, dynamic_table = fit_model(
                data=data,
                outcome=outcome,
                terms=dynamic_terms,
                robustness_name=robustness_name,
                robustness_label=spec["label"],
                model_type="dynamic",
                weight_col=weight_col,
            )

            dynamic_tables.append(
                dynamic_table
            )

            # -------------------------------------------------------------
            # Joint dynamic tests
            # -------------------------------------------------------------

            joint_rows.append(
                joint_test(
                    dynamic_result,
                    [
                        "robust_exposure_y2023",
                        "robust_exposure_y2024",
                    ],
                    robustness_name,
                    spec["label"],
                    outcome,
                    "Exposure 2023 and 2024",
                )
            )

            joint_rows.append(
                joint_test(
                    dynamic_result,
                    [
                        "robust_complementarity_y2023",
                        "robust_complementarity_y2024",
                    ],
                    robustness_name,
                    spec["label"],
                    outcome,
                    "Complementarity 2023 and 2024",
                )
            )

            joint_rows.append(
                joint_test(
                    dynamic_result,
                    [
                        "robust_interaction_y2023",
                        "robust_interaction_y2024",
                    ],
                    robustness_name,
                    spec["label"],
                    outcome,
                    "Interaction 2023 and 2024",
                )
            )

        summary_rows.append(
            {
                "robustness": robustness_name,
                "robustness_label": spec["label"],
                "exposure_variable": spec["exposure"],
                "complementarity_variable": spec[
                    "complementarity"
                ],
                "occupation_clusters": occupation_count,
                "analysis_years": "2022, 2023, 2024",
                "reference_year": REFERENCE_YEAR,
            }
        )

    # =============================================================================
    # 8. COMBINE AND SAVE
    # =============================================================================

    pooled = pd.concat(
        pooled_tables,
        ignore_index=True,
    )

    dynamic = pd.concat(
        dynamic_tables,
        ignore_index=True,
    )

    pooled = add_fdr(pooled)
    dynamic = add_fdr(dynamic)

    joint = pd.DataFrame(joint_rows)

    if not joint.empty:
        joint["q_value_bh"] = np.nan
        joint["reject_fdr_5pct"] = False

        for _, index in joint.groupby(
            ["robustness", "outcome"]
        ).groups.items():

            index = list(index)

            reject, qvalues, _, _ = multipletests(
                joint.loc[index, "p_value"],
                alpha=0.05,
                method="fdr_bh",
            )

            joint.loc[
                index,
                "q_value_bh",
            ] = qvalues

            joint.loc[
                index,
                "reject_fdr_5pct",
            ] = reject

    summary = pd.DataFrame(
        summary_rows
    )

    pooled.to_csv(
        out(
            "out_step4_robustness_pooled.csv"
        ),
        index=False,
    )

    dynamic.to_csv(
        out(
            "out_step4_robustness_dynamic.csv"
        ),
        index=False,
    )

    joint.to_csv(
        out(
            "out_step4_robustness_joint_tests.csv"
        ),
        index=False,
    )

    summary.to_csv(
        out(
            "out_step4_robustness_summary.csv"
        ),
        index=False,
    )

    # =============================================================================
    # 9. TERMINAL SUMMARY
    # =============================================================================

    print("\n" + "=" * 72)
    print("POOLED ROBUSTNESS RESULTS")
    print("=" * 72)

    display_cols = [
        "robustness_label",
        "outcome",
        "term",
        "coefficient",
        "standard_error",
        "p_value",
        "q_value_bh",
        "occupation_clusters",
    ]

    print(
        pooled[display_cols].to_string(
            index=False
        )
    )

    print("\n" + "=" * 72)
    print("DYNAMIC ROBUSTNESS RESULTS")
    print("=" * 72)

    print(
        dynamic[display_cols].to_string(
            index=False
        )
    )

    print("\n" + "=" * 72)
    print("JOINT TESTS")
    print("=" * 72)

    print(
        joint[
            [
                "robustness_label",
                "outcome",
                "dimension",
                "test_statistic",
                "p_value",
                "q_value_bh",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(
        "  out_step4_robustness_pooled.csv"
    )
    print(
        "  out_step4_robustness_dynamic.csv"
    )
    print(
        "  out_step4_robustness_joint_tests.csv"
    )
    print(
        "  out_step4_robustness_summary.csv"
    )


if __name__ == "__main__":
    main()