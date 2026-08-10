"""Second-direction analysis: revised Step 1.

Research question
-----------------
Did occupations with higher ex-ante LLM augmentation potential
M = LLM exposure x complementarity experience different changes in
real annual wage income and employment after the public release of ChatGPT?

Design
------
1. Build a 2019/2021-2024 ACS occupation-year panel.
2. Use 2022 as the event-study reference year and 2023-2024 as post periods.
3. Estimate continuous-treatment difference-in-differences models with
   occupation and year fixed effects.
4. Report M-only models, exposure/complementarity decomposition models,
   alternative-M robustness checks, and event studies.

All inputs and outputs stay in the same folder as this script.
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
YEARS = [2019, 2021, 2022, 2023, 2024]
POST_START = 2023
EVENT_REFERENCE = 2022
PRETREND_YEARS = [2019, 2021]
CHUNK_SIZE = 250_000
MIN_AGE, MAX_AGE = 16, 64
EMPLOYED_ESR = {1, 2}
MIN_MATCH_RATE = 0.85

ACS_FILES = {
    2019: [BASE / "psam_pusa.csv", BASE / "psam_pusb.csv"],
    2021: [BASE / "psam_pusa_2021.csv", BASE / "psam_pusb_2021.csv"],
    2022: [BASE / "psam_pusa_2022.csv", BASE / "psam_pusb_2022.csv"],
    2023: [BASE / "psam_pusa_2023.csv", BASE / "psam_pusb_2023.csv"],
    2024: [BASE / "psam_pusa_2024.csv", BASE / "psam_pusb_2024.csv"],
}
OCCUPATION_FILE = BASE / "occupation_master_acs.csv"

# CPI-U annual averages. ACS wage/salary income is converted to 2024 dollars.
CPI = {
    2019: 255.657,
    2021: 270.970,
    2022: 292.655,
    2023: 304.702,
    2024: 313.689,
}

ACS_COLUMNS = ["OCCP", "AGEP", "ESR", "PWGTP", "WAGP", "ADJINC"]


def out(name: str) -> Path:
    return BASE / f"out_{name}"


def normalise_occ(value: object) -> Optional[str]:
    """Convert occupation codes such as '0010', 10.0, or '10' to '10'."""
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
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (values - values.mean()) / sd


def check_inputs() -> None:
    expected = [OCCUPATION_FILE]
    for paths in ACS_FILES.values():
        expected.extend(paths)
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n  - " + "\n  - ".join(missing)
        )


def load_occupation_map() -> pd.DataFrame:
    occ = pd.read_csv(OCCUPATION_FILE, low_memory=False)
    required = {
        "acs_occ_code",
        "m_index",
        "m_no_jobzone",
        "llm_exposure_main",
        "complementarity_theta",
    }
    missing = required.difference(occ.columns)
    if missing:
        raise KeyError(
            "occupation_master_acs.csv is missing columns: "
            + str(sorted(missing))
        )

    occ = occ.copy()
    occ["acs_occ_code"] = occ["acs_occ_code"].map(normalise_occ)
    occ = occ.dropna(subset=["acs_occ_code"])

    if occ["acs_occ_code"].duplicated().any():
        duplicate_codes = (
            occ.loc[occ["acs_occ_code"].duplicated(False), "acs_occ_code"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "Duplicate acs_occ_code values in occupation_master_acs.csv: "
            + ", ".join(duplicate_codes[:20])
        )

    keep = [
        "acs_occ_code",
        "acs_occ_title",
        "soc_major",
        "llm_exposure_main",
        "complementarity_theta",
        "complementarity_no_jobzone",
        "m_index",
        "m_no_jobzone",
        "llm_exposure_model_robustness",
        "m_index_oews_weighted",
    ]
    occ = occ[[c for c in keep if c in occ.columns]].copy()

    # Optional robustness index using model-rated exposure.
    if {
        "llm_exposure_model_robustness",
        "complementarity_theta",
    }.issubset(occ.columns):
        occ["m_model_rated"] = (
            occ["llm_exposure_model_robustness"]
            * occ["complementarity_theta"]
        )

    # Standardise time-invariant occupation measures before merging to the panel.
    score_cols = [
        "llm_exposure_main",
        "complementarity_theta",
        "complementarity_no_jobzone",
        "m_index",
        "m_no_jobzone",
        "m_model_rated",
        "m_index_oews_weighted",
    ]
    for col in score_cols:
        if col in occ.columns:
            occ[f"{col}_z"] = zscore(occ[col])

    # This is the conventional centred interaction representation of E x C.
    # In the joint model it tests whether the E-C combination adds information
    # beyond the separate linear E and C components.
    if {
        "llm_exposure_main_z",
        "complementarity_theta_z",
    }.issubset(occ.columns):
        occ["ec_interaction"] = (
            occ["llm_exposure_main_z"]
            * occ["complementarity_theta_z"]
        )

    return occ


def process_acs(occ: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ACS files in chunks and aggregate workers to occupation-year cells."""
    lookup = occ[["acs_occ_code", "m_index"]]
    grouped_parts: list[pd.DataFrame] = []
    quality_rows: list[dict[str, float]] = []

    for year in YEARS:
        print("=" * 72)
        print(f"Processing ACS {year}")
        print("=" * 72)

        year_parts: list[pd.DataFrame] = []
        filtered_n = filtered_weight = 0.0
        matched_n = matched_weight = 0.0

        for path in ACS_FILES[year]:
            print(f"Reading {path.name}")
            header = pd.read_csv(path, nrows=0)
            missing = set(ACS_COLUMNS).difference(header.columns)
            if missing:
                raise KeyError(f"{path.name} is missing {sorted(missing)}")

            reader = pd.read_csv(
                path,
                usecols=ACS_COLUMNS,
                dtype={"OCCP": "string"},
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )

            for chunk_number, chunk in enumerate(reader, start=1):
                for col in ["AGEP", "ESR", "PWGTP", "WAGP", "ADJINC"]:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

                chunk = chunk.loc[
                    chunk["AGEP"].between(MIN_AGE, MAX_AGE)
                    & chunk["ESR"].isin(EMPLOYED_ESR)
                    & chunk["OCCP"].notna()
                    & (chunk["PWGTP"] > 0)
                ].copy()

                if chunk.empty:
                    continue

                filtered_n += len(chunk)
                filtered_weight += float(chunk["PWGTP"].sum())

                chunk["acs_occ_code"] = chunk["OCCP"].map(normalise_occ)
                chunk = chunk.merge(
                    lookup,
                    on="acs_occ_code",
                    how="left",
                    validate="many_to_one",
                )
                chunk = chunk.loc[chunk["m_index"].notna()].copy()

                matched_n += len(chunk)
                matched_weight += float(chunk["PWGTP"].sum())

                if chunk.empty:
                    continue

                employment = (
                    chunk.groupby("acs_occ_code", as_index=False)
                    .agg(
                        employment_weight=("PWGTP", "sum"),
                        employment_n=("PWGTP", "size"),
                    )
                )

                wage = chunk.loc[
                    (chunk["WAGP"] > 0) & (chunk["ADJINC"] > 0)
                ].copy()

                if wage.empty:
                    for col in [
                        "wage_weight",
                        "wage_n",
                        "weighted_log_wage",
                        "weighted_real_wage",
                    ]:
                        employment[col] = 0.0
                else:
                    # ADJINC first harmonises income within that ACS year;
                    # CPI then converts each survey year to constant 2024 dollars.
                    adjusted_wage = wage["WAGP"] * wage["ADJINC"] / 1_000_000
                    wage["real_wage_2024"] = (
                        adjusted_wage * CPI[2024] / CPI[year]
                    )
                    wage = wage.loc[wage["real_wage_2024"] > 0].copy()
                    wage["log_real_wage_2024"] = np.log(
                        wage["real_wage_2024"]
                    )
                    wage["weighted_log_wage"] = (
                        wage["PWGTP"] * wage["log_real_wage_2024"]
                    )
                    wage["weighted_real_wage"] = (
                        wage["PWGTP"] * wage["real_wage_2024"]
                    )

                    wage_grouped = (
                        wage.groupby("acs_occ_code", as_index=False)
                        .agg(
                            wage_weight=("PWGTP", "sum"),
                            wage_n=("PWGTP", "size"),
                            weighted_log_wage=("weighted_log_wage", "sum"),
                            weighted_real_wage=("weighted_real_wage", "sum"),
                        )
                    )
                    employment = employment.merge(
                        wage_grouped,
                        on="acs_occ_code",
                        how="left",
                        validate="one_to_one",
                    )
                    for col in [
                        "wage_weight",
                        "wage_n",
                        "weighted_log_wage",
                        "weighted_real_wage",
                    ]:
                        employment[col] = employment[col].fillna(0.0)

                employment["year"] = year
                year_parts.append(employment)

                if chunk_number % 5 == 0:
                    print(f"  completed chunk {chunk_number}")

        if not year_parts:
            raise ValueError(f"No matched ACS observations were found for {year}.")

        year_data = pd.concat(year_parts, ignore_index=True)
        year_data = (
            year_data.groupby(["year", "acs_occ_code"], as_index=False)
            .agg(
                employment_weight=("employment_weight", "sum"),
                employment_n=("employment_n", "sum"),
                wage_weight=("wage_weight", "sum"),
                wage_n=("wage_n", "sum"),
                weighted_log_wage=("weighted_log_wage", "sum"),
                weighted_real_wage=("weighted_real_wage", "sum"),
            )
        )
        grouped_parts.append(year_data)

        match_rate = matched_weight / filtered_weight if filtered_weight else np.nan
        quality_rows.append(
            {
                "year": year,
                "filtered_n": filtered_n,
                "filtered_weight": filtered_weight,
                "matched_n": matched_n,
                "matched_weight": matched_weight,
                "weighted_match_rate": match_rate,
                "matched_occupations": year_data["acs_occ_code"].nunique(),
            }
        )
        print(f"Weighted occupation match rate: {match_rate:.2%}\n")

    quality = pd.DataFrame(quality_rows)
    if (quality["weighted_match_rate"] < MIN_MATCH_RATE).any():
        warnings.warn(
            "At least one year has a match rate below 85%. "
            "Do not interpret regressions until the crosswalk is checked."
        )

    panel = pd.concat(grouped_parts, ignore_index=True)
    panel["mean_log_real_wage_2024"] = np.where(
        panel["wage_weight"] > 0,
        panel["weighted_log_wage"] / panel["wage_weight"],
        np.nan,
    )
    panel["mean_real_wage_2024"] = np.where(
        panel["wage_weight"] > 0,
        panel["weighted_real_wage"] / panel["wage_weight"],
        np.nan,
    )
    return panel, quality


def prepare_panel(
    raw_panel: pd.DataFrame,
    occ: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = raw_panel.merge(
        occ,
        on="acs_occ_code",
        how="left",
        validate="many_to_one",
    )
    panel["year"] = panel["year"].astype(int)
    panel["post"] = (panel["year"] >= POST_START).astype(int)
    panel["log_employment"] = np.log(panel["employment_weight"].clip(lower=1))
    panel["employment_share"] = (
        panel["employment_weight"]
        / panel.groupby("year")["employment_weight"].transform("sum")
    )
    panel["log_employment_share"] = np.log(
        panel["employment_share"].clip(lower=1e-12)
    )

    # Keep a five-year balanced occupation panel for the main regressions.
    observed_years = panel.groupby("acs_occ_code")["year"].nunique()
    balanced_codes = observed_years.loc[observed_years == len(YEARS)].index
    panel["balanced_5year"] = panel["acs_occ_code"].isin(balanced_codes)
    balanced = panel.loc[panel["balanced_5year"]].copy()

    # Baseline weights avoid using a post-treatment cell size as a regression weight.
    baseline = panel.loc[
        panel["year"] == EVENT_REFERENCE,
        ["acs_occ_code", "employment_weight", "wage_weight"],
    ].rename(
        columns={
            "employment_weight": "baseline_employment_weight",
            "wage_weight": "baseline_wage_weight",
        }
    )
    balanced = balanced.merge(
        baseline,
        on="acs_occ_code",
        how="left",
        validate="many_to_one",
    )

    score_cols = [
        "llm_exposure_main_z",
        "complementarity_theta_z",
        "ec_interaction",
        "m_index_z",
        "m_no_jobzone_z",
        "m_model_rated_z",
        "m_index_oews_weighted_z",
    ]
    for score in score_cols:
        if score in balanced.columns:
            balanced[f"{score}_post"] = balanced[score] * balanced["post"]

    coverage_rows = []
    for year, data in panel.groupby("year"):
        coverage_rows.append(
            {
                "year": int(year),
                "occupation_cells": len(data),
                "unique_occupations": data["acs_occ_code"].nunique(),
                "cells_with_valid_wage": data["mean_log_real_wage_2024"].notna().sum(),
                "weighted_employment": data["employment_weight"].sum(),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    balance_report = pd.DataFrame(
        {
            "metric": [
                "occupations_in_any_year",
                "occupations_in_all_five_years",
                "occupations_excluded_from_balanced_panel",
                "balanced_occupation_year_cells",
            ],
            "value": [
                panel["acs_occ_code"].nunique(),
                len(balanced_codes),
                panel["acs_occ_code"].nunique() - len(balanced_codes),
                len(balanced),
            ],
        }
    )

    return balanced, coverage, balance_report


def fit_fe_model(
    panel: pd.DataFrame,
    outcome: str,
    terms: Sequence[str],
    model_name: str,
    weight_col: Optional[str] = None,
) -> tuple[object, list[dict[str, float]]]:
    cols = [outcome, "acs_occ_code", "year", *terms]
    if weight_col:
        cols.append(weight_col)
    data = panel[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if weight_col:
        data = data.loc[data[weight_col] > 0].copy()

    formula = (
        f"{outcome} ~ " + " + ".join(terms) + " + C(acs_occ_code) + C(year)"
    )
    if weight_col:
        model = smf.wls(formula, data=data, weights=data[weight_col])
    else:
        model = smf.ols(formula, data=data)

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["acs_occ_code"]},
        use_t=True,
    )
    out(f"{model_name}.txt").write_text(
        result.summary().as_text(), encoding="utf-8"
    )

    rows: list[dict[str, float]] = []
    for term in terms:
        ci = result.conf_int().loc[term]
        rows.append(
            {
                "model": model_name,
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


def fit_event_study(
    panel: pd.DataFrame,
    outcome: str,
    score_col: str,
    model_name: str,
    weight_col: Optional[str] = None,
) -> tuple[object, pd.DataFrame, dict[str, float]]:
    data = panel.copy()
    event_years = [year for year in YEARS if year != EVENT_REFERENCE]
    event_cols = []
    for year in event_years:
        col = f"event_{year}"
        data[col] = data[score_col] * (data["year"] == year).astype(int)
        event_cols.append(col)

    cols = [outcome, "acs_occ_code", "year", *event_cols]
    if weight_col:
        cols.append(weight_col)
    data = data[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if weight_col:
        data = data.loc[data[weight_col] > 0].copy()

    formula = (
        f"{outcome} ~ "
        + " + ".join(event_cols)
        + " + C(acs_occ_code) + C(year)"
    )
    if weight_col:
        model = smf.wls(formula, data=data, weights=data[weight_col])
    else:
        model = smf.ols(formula, data=data)

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["acs_occ_code"]},
        use_t=True,
    )
    out(f"{model_name}.txt").write_text(
        result.summary().as_text(), encoding="utf-8"
    )

    rows = []
    for year in event_years:
        term = f"event_{year}"
        ci = result.conf_int().loc[term]
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "score": score_col,
                "year": year,
                "reference_year": EVENT_REFERENCE,
                "coefficient": result.params[term],
                "standard_error": result.bse[term],
                "p_value": result.pvalues[term],
                "ci_low": ci.iloc[0],
                "ci_high": ci.iloc[1],
            }
        )

    available_preterms = [
        f"event_{year}" for year in PRETREND_YEARS if f"event_{year}" in result.params
    ]
    if available_preterms:
        restriction = ", ".join(f"{term} = 0" for term in available_preterms)
        test = result.wald_test(restriction, scalar=True)
        pretrend_p = float(np.asarray(test.pvalue).squeeze())
    else:
        pretrend_p = np.nan

    pretrend = {
        "model": model_name,
        "outcome": outcome,
        "tested_pre_years": ",".join(map(str, PRETREND_YEARS)),
        "reference_year": EVENT_REFERENCE,
        "joint_pretrend_p_value": pretrend_p,
        "occupation_clusters": data["acs_occ_code"].nunique(),
    }
    return result, pd.DataFrame(rows), pretrend


def event_figure(event_table: pd.DataFrame, filename: str, title: str) -> None:
    reference = pd.DataFrame(
        {
            "year": [EVENT_REFERENCE],
            "coefficient": [0.0],
            "ci_low": [0.0],
            "ci_high": [0.0],
        }
    )
    plot = pd.concat(
        [event_table[["year", "coefficient", "ci_low", "ci_high"]], reference],
        ignore_index=True,
    ).sort_values("year")

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
    ax.set_xticks(YEARS)
    ax.set_xlabel("Year")
    ax.set_ylabel("Coefficient for a one-SD higher M")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out(filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_score_diagnostics(occ: pd.DataFrame) -> None:
    score_cols = [
        "llm_exposure_main",
        "complementarity_theta",
        "m_index",
        "m_no_jobzone",
        "m_model_rated",
        "m_index_oews_weighted",
    ]
    score_cols = [col for col in score_cols if col in occ.columns]
    occ[score_cols].describe().T.to_csv(out("score_descriptive_statistics.csv"))
    occ[score_cols].corr().to_csv(out("score_correlations.csv"))


def main() -> None:
    print("\nSECOND-DIRECTION ANALYSIS: REVISED STEP 1\n")
    check_inputs()

    occ = load_occupation_map()
    save_score_diagnostics(occ)

    raw_panel, match_quality = process_acs(occ)
    panel, coverage, balance_report = prepare_panel(raw_panel, occ)

    match_quality.to_csv(out("acs_match_report.csv"), index=False)
    coverage.to_csv(out("panel_coverage_report.csv"), index=False)
    balance_report.to_csv(out("panel_balance_report.csv"), index=False)
    panel.to_csv(out("occupation_year_panel_balanced.csv"), index=False)

    coefficient_rows: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # A. Main M-only models. Since M is standardised, beta is the outcome
    #    difference after 2022 associated with a one-SD higher M.
    # ------------------------------------------------------------------
    main_specs = [
        (
            "mean_log_real_wage_2024",
            ["m_index_z_post"],
            "wage_main_M",
            "baseline_wage_weight",
        ),
        (
            "log_employment",
            ["m_index_z_post"],
            "employment_main_M",
            None,
        ),
        (
            "log_employment_share",
            ["m_index_z_post"],
            "employment_share_main_M",
            None,
        ),
    ]
    for outcome, terms, name, weight in main_specs:
        _, rows = fit_fe_model(panel, outcome, terms, name, weight)
        coefficient_rows.extend(rows)

    # ------------------------------------------------------------------
    # B. Key novelty test: does the E x C combination add information
    #    beyond exposure and complementarity considered separately?
    # ------------------------------------------------------------------
    joint_terms = [
        "llm_exposure_main_z_post",
        "complementarity_theta_z_post",
        "ec_interaction_post",
    ]
    for outcome, name, weight in [
        ("mean_log_real_wage_2024", "wage_joint_E_C_EC", "baseline_wage_weight"),
        ("log_employment", "employment_joint_E_C_EC", None),
    ]:
        _, rows = fit_fe_model(panel, outcome, joint_terms, name, weight)
        coefficient_rows.extend(rows)

    # Separate E-only and C-only models make the comparison transparent.
    for score in ["llm_exposure_main_z", "complementarity_theta_z"]:
        for outcome, prefix, weight in [
            ("mean_log_real_wage_2024", "wage", "baseline_wage_weight"),
            ("log_employment", "employment", None),
        ]:
            term = f"{score}_post"
            _, rows = fit_fe_model(
                panel,
                outcome,
                [term],
                f"{prefix}_{score}",
                weight,
            )
            coefficient_rows.extend(rows)

    # ------------------------------------------------------------------
    # C. Alternative M definitions.
    # ------------------------------------------------------------------
    robustness_scores = [
        "m_no_jobzone_z",
        "m_model_rated_z",
        "m_index_oews_weighted_z",
    ]
    for score in robustness_scores:
        term = f"{score}_post"
        if term not in panel.columns or panel[term].notna().sum() == 0:
            continue
        for outcome, prefix, weight in [
            ("mean_log_real_wage_2024", "wage", "baseline_wage_weight"),
            ("log_employment", "employment", None),
        ]:
            _, rows = fit_fe_model(
                panel,
                outcome,
                [term],
                f"{prefix}_{score}",
                weight,
            )
            coefficient_rows.extend(rows)

    # Employment weighted by 2022 occupation size is reported as robustness,
    # not as the main specification.
    _, rows = fit_fe_model(
        panel,
        "log_employment",
        ["m_index_z_post"],
        "employment_main_M_baseline_weighted",
        "baseline_employment_weight",
    )
    coefficient_rows.extend(rows)

    pd.DataFrame(coefficient_rows).to_csv(
        out("all_model_coefficients.csv"), index=False
    )

    # ------------------------------------------------------------------
    # D. Event studies and pre-trend tests.
    # ------------------------------------------------------------------
    _, wage_event, wage_pretrend = fit_event_study(
        panel,
        outcome="mean_log_real_wage_2024",
        score_col="m_index_z",
        model_name="wage_event_study",
        weight_col="baseline_wage_weight",
    )
    _, employment_event, employment_pretrend = fit_event_study(
        panel,
        outcome="log_employment",
        score_col="m_index_z",
        model_name="employment_event_study",
    )

    pd.concat([wage_event, employment_event], ignore_index=True).to_csv(
        out("event_study_coefficients.csv"), index=False
    )
    pd.DataFrame([wage_pretrend, employment_pretrend]).to_csv(
        out("pretrend_tests.csv"), index=False
    )

    event_figure(
        wage_event,
        "wage_event_study.png",
        "M and changes in real annual wage income",
    )
    event_figure(
        employment_event,
        "employment_event_study.png",
        "M and changes in occupational employment",
    )

    print("\nCompleted. Review these files in this order:")
    print("  1. out_acs_match_report.csv")
    print("  2. out_panel_coverage_report.csv")
    print("  3. out_panel_balance_report.csv")
    print("  4. out_score_correlations.csv")
    print("  5. out_pretrend_tests.csv")
    print("  6. out_all_model_coefficients.csv")
    print("  7. out_wage_event_study.png")
    print("  8. out_employment_event_study.png")


if __name__ == "__main__":
    main()