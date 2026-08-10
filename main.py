"""Second-direction analysis: Step 1.

Build a 2019/2021-2024 ACS occupation-year panel and estimate whether
occupations with higher M = LLM exposure x complementarity experienced
different wage and employment changes after 2022.

All inputs and outputs stay in the same folder as this script.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

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

# CPI-U annual averages. Wages are converted to 2024 dollars.
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


def check_inputs() -> None:
    expected = [OCCUPATION_FILE]
    for paths in ACS_FILES.values():
        expected.extend(paths)
    missing = [p.name for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files in the same folder as main.py:\n  - "
            + "\n  - ".join(missing)
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
            "occupation_master_acs.csv is missing: " + str(sorted(missing))
        )

    occ = occ.copy()
    occ["acs_occ_code"] = occ["acs_occ_code"].map(normalise_occ)
    occ = occ.dropna(subset=["acs_occ_code"])

    if occ["acs_occ_code"].duplicated().any():
        raise ValueError("Duplicate acs_occ_code values in occupation_master_acs.csv")

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
    keep = [c for c in keep if c in occ.columns]
    occ = occ[keep].copy()

    if {
        "llm_exposure_model_robustness",
        "complementarity_theta",
    }.issubset(occ.columns):
        occ["m_model_rated"] = (
            occ["llm_exposure_model_robustness"]
            * occ["complementarity_theta"]
        )

    return occ


def process_acs(occ: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

            for chunk_number, chunk in enumerate(
                pd.read_csv(
                    path,
                    usecols=ACS_COLUMNS,
                    dtype={"OCCP": "string"},
                    chunksize=CHUNK_SIZE,
                    low_memory=False,
                ),
                start=1,
            ):
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

                # Employment aggregation: all matched civilian employed workers.
                employment = (
                    chunk.groupby("acs_occ_code", as_index=False)
                    .agg(
                        employment_weight=("PWGTP", "sum"),
                        employment_n=("PWGTP", "size"),
                    )
                )

                # Wage aggregation: positive wage/salary income only.
                wage = chunk.loc[
                    (chunk["WAGP"] > 0) & (chunk["ADJINC"] > 0)
                ].copy()

                if not wage.empty:
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

                employment["year"] = year
                year_parts.append(employment)

                if chunk_number % 5 == 0:
                    print(f"  completed chunk {chunk_number}")

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

        match_rate = matched_weight / filtered_weight
        quality_rows.append(
            {
                "year": year,
                "filtered_n": filtered_n,
                "filtered_weight": filtered_weight,
                "matched_n": matched_n,
                "matched_weight": matched_weight,
                "weighted_match_rate": match_rate,
            }
        )
        print(f"Weighted occupation match rate: {match_rate:.2%}\n")

    quality = pd.DataFrame(quality_rows)
    if (quality["weighted_match_rate"] < MIN_MATCH_RATE).any():
        warnings.warn(
            "At least one year has a match rate below 85%. Check the report."
        )

    panel = pd.concat(grouped_parts, ignore_index=True)
    panel["mean_log_real_wage_2024"] = (
        panel["weighted_log_wage"] / panel["wage_weight"]
    )
    panel["mean_real_wage_2024"] = (
        panel["weighted_real_wage"] / panel["wage_weight"]
    )
    return panel, quality


def prepare_panel(panel: pd.DataFrame, occ: pd.DataFrame) -> pd.DataFrame:
    panel = panel.merge(
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

    for m_col in [
        "m_index",
        "m_no_jobzone",
        "m_model_rated",
        "m_index_oews_weighted",
    ]:
        if m_col in panel.columns:
            panel[f"{m_col}_post"] = panel[m_col] * panel["post"]

    return panel


def fit_did(
    panel: pd.DataFrame,
    outcome: str,
    m_col: str,
    model_name: str,
    weight_col: Optional[str] = None,
):
    treatment = f"{m_col}_post"
    cols = [outcome, treatment, "acs_occ_code", "year"]
    if weight_col:
        cols.append(weight_col)
    data = panel[cols].dropna().copy()

    formula = f"{outcome} ~ {treatment} + C(acs_occ_code) + C(year)"
    if weight_col:
        model = smf.wls(formula, data=data, weights=data[weight_col])
    else:
        model = smf.ols(formula, data=data)

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["acs_occ_code"]},
    )
    out(f"{model_name}.txt").write_text(
        result.summary().as_text(), encoding="utf-8"
    )

    ci = result.conf_int().loc[treatment]
    return result, {
        "model": model_name,
        "outcome": outcome,
        "M_measure": m_col,
        "term": treatment,
        "coefficient": result.params[treatment],
        "standard_error": result.bse[treatment],
        "p_value": result.pvalues[treatment],
        "ci_low": ci.iloc[0],
        "ci_high": ci.iloc[1],
        "occupation_year_cells": len(data),
        "occupation_clusters": data["acs_occ_code"].nunique(),
        "r_squared": result.rsquared,
    }


def fit_event_study(
    panel: pd.DataFrame,
    outcome: str,
    model_name: str,
    weight_col: Optional[str] = None,
):
    data = panel.copy()
    event_years = [y for y in YEARS if y != EVENT_REFERENCE]
    event_cols = []
    for year in event_years:
        col = f"event_{year}"
        data[col] = data["m_index"] * (data["year"] == year).astype(int)
        event_cols.append(col)

    cols = [outcome, "acs_occ_code", "year", *event_cols]
    if weight_col:
        cols.append(weight_col)
    data = data[cols].dropna().copy()

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
                "year": year,
                "reference_year": EVENT_REFERENCE,
                "coefficient": result.params[term],
                "standard_error": result.bse[term],
                "p_value": result.pvalues[term],
                "ci_low": ci.iloc[0],
                "ci_high": ci.iloc[1],
            }
        )
    return result, pd.DataFrame(rows)


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
    ax.set_ylabel("Coefficient on M × year")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out(filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("\nSECOND-DIRECTION ANALYSIS: STEP 1\n")
    check_inputs()
    occ = load_occupation_map()
    raw_panel, quality = process_acs(occ)
    panel = prepare_panel(raw_panel, occ)

    quality.to_csv(out("acs_match_report.csv"), index=False)
    panel.to_csv(out("occupation_year_panel.csv"), index=False)

    results = []
    _, row = fit_did(
        panel,
        outcome="mean_log_real_wage_2024",
        m_col="m_index",
        model_name="wage_main_M",
        weight_col="wage_weight",
    )
    results.append(row)

    _, row = fit_did(
        panel,
        outcome="log_employment",
        m_col="m_index",
        model_name="employment_main_M",
    )
    results.append(row)

    _, row = fit_did(
        panel,
        outcome="mean_log_real_wage_2024",
        m_col="m_no_jobzone",
        model_name="wage_no_jobzone",
        weight_col="wage_weight",
    )
    results.append(row)

    _, row = fit_did(
        panel,
        outcome="log_employment",
        m_col="m_no_jobzone",
        model_name="employment_no_jobzone",
    )
    results.append(row)

    wage_event_result, wage_event = fit_event_study(
        panel,
        outcome="mean_log_real_wage_2024",
        model_name="wage_event_study",
        weight_col="wage_weight",
    )
    employment_event_result, employment_event = fit_event_study(
        panel,
        outcome="log_employment",
        model_name="employment_event_study",
    )

    pd.DataFrame(results).to_csv(out("main_coefficients.csv"), index=False)
    pd.concat([wage_event, employment_event], ignore_index=True).to_csv(
        out("event_study_coefficients.csv"), index=False
    )

    event_figure(
        wage_event,
        "wage_event_study.png",
        "M and changes in real wage income",
    )
    event_figure(
        employment_event,
        "employment_event_study.png",
        "M and changes in occupational employment",
    )

    print("\nCompleted. Review these files first:")
    print("  out_acs_match_report.csv")
    print("  out_main_coefficients.csv")
    print("  out_event_study_coefficients.csv")
    print("  out_wage_event_study.png")
    print("  out_employment_event_study.png")


if __name__ == "__main__":
    main()