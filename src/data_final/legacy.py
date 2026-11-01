"""Recover a two-year research dataset without inventing missing source metadata."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_FILES = {
    "final": "KOSPI_KOSDAQ_merged_with_employees.csv",
    "kosdaq": "kosdaq100.xlsx",
    "wide": "KOSPI_KOSDAQ_merge (1).xlsx",
    "kospi": "kospi100.xlsx",
    "merged": "KOSPI_KOSDAQ_merge (2).xlsx",
}
ALIASES = {
    "net_income": {
        "당기순이익",
        "당기순이익(손실)",
        "당기의순이익",
        "연결당기순이익",
        "연결당기순이익(손실)",
        "당기연결순이익",
        "당기연결순이익(손실)",
    },
    "operating_income": {"영업이익", "영업이익(손실)"},
    "revenue": {"매출액", "수익(매출액)"},
    "assets": {"자산총계"},
    "liabilities": {"부채총계"},
}
LEGACY_FINANCIAL = [
    "return_on_assets_2021",
    "net_margin_2021",
    "operating_margin_2021",
    "leverage_2021",
]
LEGACY_EMPLOYMENT = ["log_employees_2021"]


def normalize_account(name: str) -> str:
    name = re.sub(r"\s+", "", str(name))
    return re.sub(r"^(?:\(?\d+\)?\.?|[IVXⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.)", "", name).lstrip("-")


def unique_value(values) -> tuple[float, str]:
    numbers = pd.to_numeric(pd.Series(list(values), dtype=object), errors="raise").dropna()
    if np.isinf(numbers).any():
        raise ValueError("Infinite legacy source value")
    distinct = numbers.unique()
    if len(distinct) == 0:
        return np.nan, "missing"
    if len(distinct) > 1:
        return np.nan, "conflicting_values"
    return float(distinct[0]), "observed_unique"


def read_sources(directory: Path) -> tuple[dict, list]:
    sources, manifest = {}, []
    for role, name in SOURCE_FILES.items():
        path = directory / name
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
        frame.columns = frame.columns.map(str)
        # Exact identity within these files only; no invented DART code or market assignment.
        if frame["기업명"].isna().any():
            raise ValueError(f"Missing company name: {name}")
        frame["기업명"] = frame["기업명"].str.strip()
        sources[role] = frame
        manifest.append(
            {
                "role": role,
                "filename": name,
                "rows": len(frame),
                "companies": int(frame["기업명"].nunique()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sheet": "CSV" if path.suffix == ".csv" else "Sheet1",
            }
        )
    return sources, manifest


def audit_sources(sources: dict) -> tuple[dict, pd.DataFrame]:
    raw, final = sources["merged"].copy(), sources["final"].copy()
    combined = pd.concat([sources["kospi"], sources["kosdaq"]], ignore_index=True)
    same_merge = combined[raw.columns].equals(raw)
    key = ["기업명", "계정과목_원본"]
    left = raw.rename(columns={"계정과목": "계정과목_원본"})
    right = final[final["계정과목_원본"].ne("직원수")].copy()
    left["occurrence"] = left.groupby(key).cumcount()
    right["occurrence"] = right.groupby(key).cumcount()
    pairs = left.merge(
        right,
        on=key + ["occurrence"],
        how="outer",
        suffixes=("_raw", "_final"),
        indicator=True,
        validate="one_to_one",
    )
    counts = {}
    for year in ["2021", "2022"]:
        matched = pairs["_merge"].eq("both")
        filled = matched & pairs[f"{year}_raw"].isna() & pairs[f"{year}_final"].notna()
        changed = (
            matched & pairs[f"{year}_raw"].notna() & pairs[f"{year}_raw"].ne(pairs[f"{year}_final"])
        )
        counts[year] = {
            "filled_cells": int(filled.sum()),
            "changed_observed_cells": int(changed.sum()),
            "raw_global_median": float(raw[year].median()) if raw[year].notna().any() else None,
            "replacement_values": sorted(pairs.loc[filled, f"{year}_final"].unique().tolist()),
        }
    sizes = final.groupby(["기업명", "계정과목_표준"]).size()
    risky = final[
        final["계정과목_원본"].map(normalize_account).isin(["자본과부채총계", "부채와자본총계"])
    ]
    bad_mappings = risky.groupby(["계정과목_원본", "계정과목_표준"]).size().reset_index(name="rows")
    staff = final[final["계정과목_원본"].eq("직원수")]
    repeated = (
        staff.groupby(["2021", "2022"])
        .agg(companies=("기업명", "size"), names=("기업명", list))
        .reset_index()
    )
    repeated = repeated[repeated.companies >= 3]
    result = {
        "raw_rows": len(raw),
        "companies": int(raw["기업명"].nunique()),
        "merge_exactly_matches_source_concat": same_merge,
        "raw_exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_companies": sorted(raw.loc[raw.duplicated(keep=False), "기업명"].unique()),
        "fill_audit": counts,
        "financial_rows_unmatched": int(pairs["_merge"].ne("both").sum()),
        "standardized_duplicate_groups": int(sizes.gt(1).sum()),
        "standardized_extra_rows": int(final.duplicated(["기업명", "계정과목_표준"]).sum()),
        "unsafe_total_mappings": bad_mappings.to_dict("records"),
        "repeated_employee_pairs": repeated.to_dict("records"),
        "employee_pair_interpretation": (
            "Review flags only; repetition alone does not prove imputation."
        ),
        "missing_metadata": [
            "corp_code",
            "currency",
            "fs_div",
            "reprt_code",
            "actual_year",
            "rcept_no",
        ],
    }
    return result, pairs


def recover_panel(sources: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = sources["merged"].copy()
    raw["source_row"] = np.arange(len(raw)) + 2
    raw["normalized"] = raw["계정과목"].map(normalize_account)
    final, wide = sources["final"], sources["wide"]
    employee = final[final["계정과목_원본"].eq("직원수")].copy()
    repeated = employee.groupby(["2021", "2022"])["기업명"].transform("size").ge(3)
    flagged = set(employee.loc[repeated, "기업명"])
    rows, lineage = [], []
    for company, group in raw.groupby("기업명", sort=True):
        for year in ["2021", "2022"]:
            row = {
                "company": company,
                "year": int(year),
                "data_kind": "legacy_unverified",
                "currency": "unknown",
                "fs_div": "unknown",
                "report_period": "unknown",
                "employee_repeated_pair_flag": company in flagged,
            }
            for metric, aliases in ALIASES.items():
                matches = group[group.normalized.isin(aliases)]
                row[metric], status = unique_value(matches[year])
                lineage.append(
                    {
                        "company": company,
                        "year": int(year),
                        "metric": metric,
                        "status": status,
                        "source": SOURCE_FILES["merged"],
                        "sheet": "Sheet1",
                        "source_rows": ";".join(matches.source_row.astype(str)),
                        "accounts": ";".join(sorted(matches["계정과목"].unique())),
                    }
                )
            final_values = employee.loc[employee["기업명"].eq(company), year]
            wide_values = wide.loc[wide["기업명"].eq(company), f"{year}직원수"]
            row["employees"], status = unique_value(pd.concat([final_values, wide_values]))
            if pd.notna(row["employees"]) and (
                row["employees"] < 0 or not row["employees"].is_integer()
            ):
                raise ValueError(f"Invalid employee count: {company}, {year}")
            row["employee_status"] = status
            lineage.append(
                {
                    "company": company,
                    "year": int(year),
                    "metric": "employees",
                    "status": status,
                    "source": "final CSV + wide workbook",
                    "sheet": "CSV / Sheet1",
                    "source_rows": "",
                    "accounts": "직원수",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(lineage)


def build_legacy_examples(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "company",
        "year",
        "net_income",
        "operating_income",
        "revenue",
        "assets",
        "liabilities",
        "employees",
        "data_kind",
        "employee_repeated_pair_flag",
    }
    if required - set(panel.columns):
        raise ValueError(f"Missing legacy columns: {sorted(required - set(panel.columns))}")
    if panel.duplicated(["company", "year"]).any():
        raise ValueError("Duplicate legacy company-year")
    if not panel.data_kind.eq("legacy_unverified").all():
        raise ValueError("Legacy metadata must remain explicitly unverified")
    if set(panel.year.unique()) != {2021, 2022}:
        raise ValueError("This experiment is specifically the 2021 to 2022 transition")
    prior = panel[panel.year.eq(2021)].copy().set_index("company")
    target = panel[panel.year.eq(2022)].set_index("company").net_income
    prior["return_on_assets_2021"] = prior.net_income / prior.assets.where(prior.assets.gt(0))
    prior["net_margin_2021"] = prior.net_income / prior.revenue.where(prior.revenue.gt(0))
    prior["operating_margin_2021"] = prior.operating_income / prior.revenue.where(
        prior.revenue.gt(0)
    )
    prior["leverage_2021"] = prior.liabilities / prior.assets.where(prior.assets.gt(0))
    prior["log_employees_2021"] = np.log1p(prior.employees.where(prior.employees.ge(0)))
    prior["target_net_income_2022"] = target
    prior["target_year"] = 2022
    prior["previous_loss"] = prior.net_income.lt(0).astype(float).where(prior.net_income.notna())
    cols = LEGACY_FINANCIAL + LEGACY_EMPLOYMENT
    prior[cols] = prior[cols].replace([np.inf, -np.inf], np.nan)
    prior["observed_financial_features"] = prior[LEGACY_FINANCIAL].notna().sum(axis=1)
    prior["exclusion_reason"] = np.select(
        [
            prior.target_net_income_2022.isna(),
            prior.target_net_income_2022.eq(0),
            prior.observed_financial_features.lt(2),
        ],
        ["missing_target", "zero_target", "fewer_than_two_financial_features"],
        default="included",
    )
    eligibility = prior[["exclusion_reason", "observed_financial_features"]].reset_index()
    examples = prior[prior.exclusion_reason.eq("included")].copy()
    examples["loss_2022"] = examples.target_net_income_2022.lt(0).astype(int)
    return examples.reset_index(), eligibility


def compare_labels(sources: dict, panel: pd.DataFrame) -> pd.DataFrame:
    old = sources["final"].pivot_table(
        index="기업명", columns="계정과목_표준", values="2022", aggfunc="sum"
    )["당기순이익(손실)"]
    # Reconstruct the old code, including target median substitution, ONLY for the audit.
    result = panel[panel.year.eq(2022)][["company", "net_income"]].set_index("company")
    result["legacy_aggregated_net_income"] = old
    result["legacy_target_was_missing"] = result.legacy_aggregated_net_income.isna()
    result["legacy_model_target"] = result.legacy_aggregated_net_income.fillna(old.median())
    result["label_disagrees"] = result.net_income.notna() & (
        result.net_income.lt(0) != result.legacy_model_target.lt(0)
    )
    return result.reset_index()


def prepare_legacy(directory: Path, output: Path) -> tuple[pd.DataFrame, dict]:
    sources, manifest = read_sources(directory)
    audit, pairs = audit_sources(sources)
    if not audit["merge_exactly_matches_source_concat"] or audit["financial_rows_unmatched"]:
        raise ValueError("Source reconciliation changed; inspect before analysis")
    panel, lineage = recover_panel(sources)
    examples, eligibility = build_legacy_examples(panel)
    labels = compare_labels(sources, panel)
    audit.update(
        {
            "manifest": manifest,
            "recovered_target_count_2022": int(
                panel.loc[panel.year.eq(2022), "net_income"].notna().sum()
            ),
            "model_eligible_companies": len(examples),
            "model_loss_count": int(examples.loss_2022.sum()),
            "exclusions": eligibility.exclusion_reason.value_counts().to_dict(),
            "target_sign_disagreements": labels.loc[labels.label_disagrees, "company"].tolist(),
            "legacy_target_missing_companies": labels.loc[
                labels.legacy_target_was_missing, "company"
            ].tolist(),
            "employee_conflicts": lineage.loc[
                lineage.metric.eq("employees") & lineage.status.eq("conflicting_values"),
                ["company", "year"],
            ].to_dict("records"),
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output / "legacy_panel.csv", index=False)
    lineage.to_csv(output / "lineage.csv", index=False)
    eligibility.to_csv(output / "eligibility.csv", index=False)
    labels.to_csv(output / "target_audit.csv", index=False)
    pairs.to_csv(output / "source_comparison.csv", index=False)
    (output / "source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    return panel, audit
