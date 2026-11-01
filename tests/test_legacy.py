import numpy as np
import pandas as pd
import pytest

from data_final.data import synthetic_panel
from data_final.legacy import (
    LEGACY_EMPLOYMENT,
    LEGACY_FINANCIAL,
    audit_sources,
    build_legacy_examples,
    normalize_account,
    unique_value,
)
from data_final.legacy_analysis import evaluate_examples


def legacy_fixture():
    panel = synthetic_panel(companies=60)
    panel = panel[panel.year.isin([2021, 2022])].rename(columns={"employee": "employees"})
    panel["data_kind"] = "legacy_unverified"
    panel["employee_repeated_pair_flag"] = False
    return panel


def test_alias_normalization_does_not_merge_accounting_concepts():
    assert normalize_account("Ⅵ. 당기순이익") == "당기순이익"
    assert normalize_account("I.  매출액") == "매출액"
    assert normalize_account("자본과 부채총계") != normalize_account("부채총계")
    assert normalize_account("중단영업당기순이익") != "당기순이익"
    assert normalize_account("계속영업법인세비용차감전당기순이익") != "당기순이익"


def test_conflicting_values_are_not_summed_or_first_matched():
    value, status = unique_value([10, 10, np.nan])
    assert value == 10 and status == "observed_unique"
    value, status = unique_value([10, 20])
    assert np.isnan(value) and status == "conflicting_values"


def test_2022_input_cannot_change_2021_features():
    panel = legacy_fixture()
    original, _ = build_legacy_examples(panel)
    panel.loc[panel.year.eq(2022), ["assets", "revenue", "employees"]] *= 100
    modified, _ = build_legacy_examples(panel)
    cols = LEGACY_FINANCIAL + LEGACY_EMPLOYMENT
    pd.testing.assert_frame_equal(original[cols], modified[cols])
    assert all(c.endswith("2021") for c in cols)


def test_missing_target_is_excluded_and_unknown_metadata_cannot_be_relabelled():
    panel = legacy_fixture()
    company = panel.company.iloc[0]
    panel.loc[panel.company.eq(company) & panel.year.eq(2022), "net_income"] = np.nan
    examples, eligibility = build_legacy_examples(panel)
    assert company not in examples.company.tolist()
    assert eligibility.set_index("company").loc[company, "exclusion_reason"] == "missing_target"
    panel["data_kind"] = "dart"
    with pytest.raises(ValueError, match="unverified"):
        build_legacy_examples(panel)


def test_source_join_does_not_multiply_duplicate_rows():
    raw = pd.DataFrame(
        {
            "기업명": ["A", "A"],
            "계정과목": ["매출액", "매출액"],
            "2021": [np.nan, np.nan],
            "2022": [20, 20],
        }
    )
    final = raw.rename(columns={"계정과목": "계정과목_원본"}).copy()
    final["계정과목_표준"] = "매출액"
    final["2021"] = 5
    audit, pairs = audit_sources(
        {"merged": raw, "final": final, "kospi": raw.iloc[:1], "kosdaq": raw.iloc[1:]}
    )
    assert len(pairs) == 2
    assert audit["fill_audit"]["2021"]["filled_cells"] == 2


def test_outer_folds_hold_out_every_company_once_per_repeat():
    examples, _ = build_legacy_examples(legacy_fixture())
    metrics, predictions, folds, selection = evaluate_examples(examples, seeds=(42,))
    for _, fold in folds.groupby("fold"):
        assert set(fold[fold.split.eq("train")].company).isdisjoint(
            set(fold[fold.split.eq("test")].company)
        )
    assert folds[folds.split.eq("test")].company.value_counts().eq(1).all()
    assert predictions.groupby("model").size().eq(len(examples)).all()
    assert len(selection) == 5
    assert metrics.model.eq("nested_selection").any()
