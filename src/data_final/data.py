"""Strict company-year contract and lagged, explicitly named predictors."""

import numpy as np
import pandas as pd

NUMERIC = ["revenue", "operating_income", "net_income", "assets", "liabilities", "employees"]
META = ["corp_code", "company", "year", "fs_div", "reprt_code", "data_kind"]
FINANCIAL_FEATURES = ["log_assets", "leverage", "operating_margin", "net_margin", "revenue_growth"]
EMPLOYMENT_FEATURES = ["log_employees", "employee_growth"]
FEATURES = FINANCIAL_FEATURES + EMPLOYMENT_FEATURES


def validate_panel(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(META + NUMERIC) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = frame.copy()
    if df[META].isna().any().any():
        raise ValueError("Metadata must not be missing")
    if not df.corp_code.astype(str).str.fullmatch(r"\d{8}").all():
        raise ValueError("corp_code must be an eight-digit string; read CSV with dtype=str")
    years = pd.to_numeric(df.year, errors="raise")
    if not (years.between(2015, 2100) & years.eq(np.floor(years))).all():
        raise ValueError("Invalid fiscal year")
    df["year"] = years.astype(int)
    if df.duplicated(["corp_code", "year"]).any():
        raise ValueError("Duplicate company-year rows; never sum them implicitly")
    if not df.fs_div.eq("OFS").all() or not df.reprt_code.astype(str).eq("11011").all():
        raise ValueError("This employment study requires annual OFS data only")
    if df.data_kind.nunique() != 1 or not df.data_kind.isin(["synthetic", "dart"]).all():
        raise ValueError("Use exactly one data kind: synthetic or dart")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if np.isinf(df[col].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite value in {col}")
    for col in ["assets", "revenue", "liabilities", "employees"]:
        if df[col].dropna().lt(0).any():
            raise ValueError(f"Negative {col}; inspect source")
    if not df.employees.dropna().eq(np.floor(df.employees.dropna())).all():
        raise ValueError("Employee counts must be integers")
    return df.sort_values(["corp_code", "year"]).reset_index(drop=True)


def quality_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "companies": int(df.corp_code.nunique()),
        "years": sorted(df.year.unique().tolist()),
        "data_kind": df.data_kind.iloc[0],
        "missing_fraction": df[NUMERIC].isna().mean().to_dict(),
        "missing_net_income": int(df.net_income.isna().sum()),
        "zero_net_income": int(df.net_income.eq(0).sum()),
        "scope": "annual OFS; amounts in source currency KRW; employee headcount",
    }


def build_examples(panel: pd.DataFrame) -> pd.DataFrame:
    """Year t features -> year t+1 loss. Missing/zero outcomes are excluded, not filled."""
    df = validate_panel(panel)
    previous = df[["corp_code", "year", "revenue", "employees"]].copy()
    previous["year"] += 1
    df = df.merge(
        previous,
        on=["corp_code", "year"],
        suffixes=("", "_previous"),
        how="left",
        validate="one_to_one",
    )
    df["log_assets"] = np.log1p(df.assets)
    df["log_employees"] = np.log1p(df.employees)
    df["leverage"] = df.liabilities / df.assets.where(df.assets.gt(0))
    df["operating_margin"] = df.operating_income / df.revenue.where(df.revenue.gt(0))
    df["net_margin"] = df.net_income / df.revenue.where(df.revenue.gt(0))
    for value, feature in [("revenue", "revenue_growth"), ("employees", "employee_growth")]:
        prev = df[f"{value}_previous"]
        df[feature] = (df[value] - prev) / prev.where(prev.gt(0))
    outcomes = panel[["corp_code", "year", "net_income"]].copy()
    outcomes["year"] = pd.to_numeric(outcomes.year) - 1
    outcomes = outcomes.rename(columns={"net_income": "next_net_income"})
    df = df.merge(outcomes, on=["corp_code", "year"], how="left", validate="one_to_one")
    df = df[df.next_net_income.notna() & df.next_net_income.ne(0)].copy()
    df["target_year"] = df.year + 1
    df["loss_next_year"] = df.next_net_income.lt(0).astype(int)
    df["previous_loss"] = df.net_income.lt(0).where(df.net_income.notna())
    # Identity and outcomes remain in the audit table, never in the allowlisted X.
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True)


def synthetic_panel(seed: int = 42, companies: int = 100) -> pd.DataFrame:
    """Generated system-test fixture, NOT a simulation calibrated to Korean companies."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(companies):
        assets = float(rng.lognormal(25, 1))
        employees = int(rng.integers(100, 5000))
        margin = float(rng.normal(0.04, 0.12))
        for year in range(2018, 2025):
            assets *= float(rng.uniform(0.85, 1.2))
            employees = max(10, int(employees * rng.uniform(0.88, 1.12)))
            margin = 0.55 * margin + rng.normal(0.015, 0.10)
            revenue = assets * rng.uniform(0.4, 1.4)
            rows.append(
                dict(
                    corp_code=f"{90000000 + i:08d}",
                    company=f"DEMO_{i:03d}",
                    year=year,
                    fs_div="OFS",
                    reprt_code="11011",
                    data_kind="synthetic",
                    revenue=revenue,
                    operating_income=revenue * (margin + 0.02),
                    net_income=revenue * margin,
                    assets=assets,
                    liabilities=assets * rng.uniform(0.1, 0.9),
                    employees=employees,
                )
            )
    df = pd.DataFrame(rows)
    for col in ["employees", "operating_income", "net_income"]:
        df.loc[rng.random(len(df)) < 0.03, col] = np.nan
    return validate_panel(df)
