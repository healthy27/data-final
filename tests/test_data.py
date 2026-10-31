import numpy as np
import pytest

from data_final.data import FEATURES, build_examples, synthetic_panel, validate_panel


def test_reject_duplicate_identity_and_mixed_scope():
    df = synthetic_panel(companies=3)
    import pandas as pd

    with pytest.raises(ValueError, match="Duplicate"):
        validate_panel(pd.concat([df, df.iloc[:1]]))
    df.loc[0, "fs_div"] = "CFS"
    with pytest.raises(ValueError, match="annual OFS"):
        validate_panel(df)


def test_missing_target_never_imputed_and_feature_year_is_previous():
    df = synthetic_panel(companies=3)
    code = df.corp_code.iloc[0]
    df.loc[df.corp_code.eq(code) & df.year.eq(2022), "net_income"] = np.nan
    examples = build_examples(df)
    assert not ((examples.corp_code == code) & (examples.target_year == 2022)).any()
    assert (examples.target_year == examples.year + 1).all()
    assert not set(FEATURES) & {"next_net_income", "loss_next_year", "company", "corp_code"}


def test_year_gap_is_not_treated_as_consecutive_growth():
    df = synthetic_panel(companies=3)
    code = df.corp_code.iloc[0]
    df = df[~(df.corp_code.eq(code) & df.year.eq(2020))]
    df.loc[df.corp_code.eq(code) & df.year.eq(2022), "net_income"] = 100
    row = build_examples(df).query("corp_code == @code and year == 2021").iloc[0]
    assert np.isnan(row.revenue_growth)


def test_zero_outcome_and_infinite_inputs_are_not_silently_accepted():
    df = synthetic_panel(companies=3)
    code = df.corp_code.iloc[0]
    df.loc[df.corp_code.eq(code) & df.year.eq(2022), "net_income"] = 0
    assert not build_examples(df).query("corp_code == @code and target_year == 2022").shape[0]
    df.loc[0, "assets"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        validate_panel(df)


def test_future_outcome_changes_do_not_change_previous_year_predictors():
    df = synthetic_panel(companies=3)
    original = build_examples(df).query("target_year == 2024").set_index("corp_code")
    df.loc[df.year.eq(2024), "assets"] *= 100
    changed = build_examples(df).query("target_year == 2024").set_index("corp_code")
    import pandas as pd

    pd.testing.assert_frame_equal(original[FEATURES], changed[FEATURES])
