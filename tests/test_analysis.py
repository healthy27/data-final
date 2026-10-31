import json

import numpy as np

from data_final.analysis import candidates, run_analysis, temporal_folds
from data_final.data import build_examples, synthetic_panel


def test_all_validation_folds_strictly_precede_test():
    examples = build_examples(synthetic_panel(companies=30))
    for year, train, valid in temporal_folds(examples, 2024):
        assert train.target_year.max() < year < 2024
        assert valid.target_year.eq(year).all()


def test_imputer_only_learns_training_median():
    model = candidates(42)["logistic"]
    model.fit([[1], [3], [np.nan], [5]], [0, 1, 0, 1])
    model.predict_proba([[1000000], [np.nan]])
    assert model.steps[0][1].statistics_[0] == 3


def test_end_to_end_report_is_labelled_and_auditable(tmp_path):
    result = run_analysis(synthetic_panel(companies=30), tmp_path)
    assert result["data_kind"] == "synthetic"
    assert "NOT REAL PERFORMANCE" in (tmp_path / "report.md").read_text()
    assert result["positive_class"] == "loss"
    assert result["test_target_year"] == 2024
    assert len(result["panel_sha256"]) == 64
    assert result["metrics"]["n"] <= 30
    assert (tmp_path / "evaluation.png").stat().st_size > 1000
    assert json.loads((tmp_path / "metrics.json").read_text())["metrics"] == result["metrics"]
