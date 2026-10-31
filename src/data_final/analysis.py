"""Fiscal-year validation, baselines, employment ablation, and auditable reports."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_final.data import (
    FEATURES,
    FINANCIAL_FEATURES,
    build_examples,
    quality_summary,
    validate_panel,
)


def candidates(seed: int) -> dict:
    return {
        "dummy": make_pipeline(
            SimpleImputer(keep_empty_features=True), DummyClassifier(strategy="prior")
        ),
        "logistic": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=1, max_iter=2000, random_state=seed),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            HistGradientBoostingClassifier(
                max_iter=80,
                max_leaf_nodes=7,
                min_samples_leaf=15,
                l2_regularization=2,
                early_stopping=False,
                random_state=seed,
            ),
        ),
    }


def scores(y, probability) -> dict:
    prediction = np.asarray(probability) >= 0.5
    return {
        "n": len(y),
        "loss_prevalence": float(np.mean(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "loss_precision": float(precision_score(y, prediction, zero_division=0)),
        "loss_recall": float(recall_score(y, prediction, zero_division=0)),
        "loss_f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def bootstrap_intervals(y, probability, seed: int, repetitions: int = 300) -> dict:
    # Test contains one row per firm, one fiscal year; stratification conditions on class counts.
    rng = np.random.default_rng(seed)
    y, probability = np.asarray(y), np.asarray(probability)
    groups = [np.flatnonzero(y == label) for label in [0, 1]]
    samples = {"roc_auc": [], "average_precision": [], "loss_recall": []}
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        result = scores(y[indices], probability[indices])
        for key in samples:
            samples[key].append(result[key])
    return {key: np.quantile(value, [0.025, 0.975]).tolist() for key, value in samples.items()}


def temporal_folds(examples: pd.DataFrame, test_year: int) -> list[tuple]:
    earlier = sorted(y for y in examples.target_year.unique() if y < test_year)
    if len(earlier) < 3:
        raise ValueError("Need at least three pre-test target years plus a held-out test year")
    folds = []
    for year in earlier[2:]:
        train = examples[examples.target_year < year]
        valid = examples[examples.target_year == year]
        if train.loss_next_year.nunique() == 2 and valid.loss_next_year.nunique() == 2:
            folds.append((year, train, valid))
    if not folds:
        raise ValueError("No temporal validation fold has both classes; expand the cohort")
    return folds


def employment_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    prior = panel[["corp_code", "year", "employees", "net_income", "assets"]].copy()
    prior["year"] += 1
    joined = panel.merge(prior, on=["corp_code", "year"], suffixes=("", "_prior"))
    joined["employee_change"] = (
        joined.employees - joined.employees_prior
    ) / joined.employees_prior.where(joined.employees_prior.gt(0))
    joined["income_change_scaled"] = (
        joined.net_income - joined.net_income_prior
    ) / joined.assets_prior.where(joined.assets_prior.gt(0))
    rows = []
    for year, group in joined.groupby("year"):
        complete = group[["employee_change", "income_change_scaled"]].dropna()
        enough = len(complete) >= 3 and complete.nunique().min() > 1
        rows.append(
            {
                "year": int(year),
                "n_complete_pairs": len(complete),
                "pearson_r": float(pearsonr(*complete.to_numpy().T).statistic) if enough else None,
                "spearman_r": float(spearmanr(*complete.to_numpy().T).statistic)
                if enough
                else None,
            }
        )
    return pd.DataFrame(rows)


def table(frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        + [
            "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def run_analysis(panel: pd.DataFrame, output: Path, test_year: int = 2024, seed: int = 42) -> dict:
    panel = validate_panel(panel)
    if panel.empty:
        raise ValueError("Empty panel")
    examples = build_examples(panel)
    train = examples[examples.target_year < test_year]
    test = examples[examples.target_year == test_year]
    if train.loss_next_year.nunique() != 2 or test.loss_next_year.nunique() != 2:
        raise ValueError("Training and test sets both need observed profit and loss outcomes")
    folds = temporal_folds(examples, test_year)
    feature_sets = {"financial": FINANCIAL_FEATURES, "financial_employment": FEATURES}
    models = candidates(seed)
    records = []
    for name, model in models.items():
        for feature_set, columns in feature_sets.items():
            if name == "dummy" and feature_set != "financial":
                continue
            for year, past, valid in folds:
                fitted = clone(model).fit(past[columns], past.loss_next_year)
                probability = fitted.predict_proba(valid[columns])[:, 1]
                records.append(
                    {
                        "model": name,
                        "features": feature_set,
                        "validation_year": int(year),
                        "train_max_target_year": int(past.target_year.max()),
                        **scores(valid.loss_next_year, probability),
                    }
                )
    cv = pd.DataFrame(records)
    comparison = cv.groupby(["model", "features"], as_index=False).agg(
        mean_validation_ap=("average_precision", "mean"),
        std_validation_ap=("average_precision", "std"),
        mean_validation_auc=("roc_auc", "mean"),
        folds=("validation_year", "count"),
    )
    comparison = comparison.sort_values(
        ["mean_validation_ap", "model", "features"], ascending=[False, True, True]
    )
    best = comparison.iloc[0]
    selected_features = feature_sets[best.features]
    model = clone(models[best.model]).fit(train[selected_features], train.loss_next_year)
    probability = model.predict_proba(test[selected_features])[:, 1]
    metrics = scores(test.loss_next_year, probability)
    prior = float(train.loss_next_year.mean())
    prior_probability = np.full(len(test), prior)
    persistence = test.previous_loss.fillna(prior).to_numpy(dtype=float)
    baselines = {
        "training_class_prior": scores(test.loss_next_year, prior_probability),
        "previous_year_loss": scores(test.loss_next_year, persistence),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    kind = panel.data_kind.iloc[0]
    label = (
        "SYNTHETIC DEMO — NOT REAL PERFORMANCE"
        if kind == "synthetic"
        else "RETROSPECTIVE DART SNAPSHOT"
    )
    cv.to_csv(output / "validation_folds.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    examples.to_csv(output / "examples.csv", index=False)
    predictions = test[["corp_code", "company", "year", "target_year", "loss_next_year"]].copy()
    predictions["loss_probability"] = probability
    predictions["loss_prediction"] = (probability >= 0.5).astype(int)
    predictions.to_csv(output / "predictions.csv", index=False)
    correlations = employment_correlations(panel)
    correlations.to_csv(output / "employment_correlations.csv", index=False)
    importance = permutation_importance(
        model,
        test[selected_features],
        test.loss_next_year,
        scoring="average_precision",
        n_repeats=15,
        random_state=seed,
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": selected_features,
            "mean_ap_decrease": importance.importances_mean,
            "std_ap_decrease": importance.importances_std,
        }
    )
    importance_df.to_csv(output / "permutation_importance.csv", index=False)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#f8fafc",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    RocCurveDisplay.from_predictions(test.loss_next_year, probability, ax=axes[0], name="Selected")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    PrecisionRecallDisplay.from_predictions(test.loss_next_year, probability, ax=axes[1])
    axes[1].axhline(test.loss_next_year.mean(), color="gray", linestyle="--", label="Prevalence")
    axes[1].legend()
    ConfusionMatrixDisplay(
        confusion_matrix(test.loss_next_year, probability >= 0.5, labels=[0, 1]),
        display_labels=["Profit", "Loss"],
    ).plot(ax=axes[2], colorbar=False)
    fig.suptitle(label, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panel.groupby("year")["employees"].apply(lambda s: s.isna().mean()).plot.bar(
        ax=axes[0], color="#0f766e"
    )
    axes[0].set(title="Employee data missingness", ylabel="Missing fraction", ylim=(0, 1))
    ordered = importance_df.sort_values("mean_ap_decrease")
    axes[1].barh(
        ordered.feature, ordered.mean_ap_decrease, xerr=ordered.std_ap_decrease, color="#2563eb"
    )
    axes[1].set(title="Held-out permutation importance", xlabel="AP decrease (not causal)")
    fig.suptitle(label, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "quality_importance.png", dpi=150)
    plt.close(fig)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, True
    source_hash = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(file.name.encode() + file.read_bytes())
    result = {
        "data_kind": kind,
        "label": label,
        "seed": seed,
        "test_target_year": test_year,
        "feature_year": test_year - 1,
        "positive_class": "loss",
        "threshold": 0.5,
        "selected_model": best.model,
        "selected_features": best.features,
        "selection": "highest mean rolling validation average precision; no test selection",
        "metrics": metrics,
        "baselines": baselines,
        "bootstrap_95_percent_intervals": bootstrap_intervals(
            test.loss_next_year, probability, seed
        ),
        "interval_scope": (
            "stratified firm resampling in one test year; excludes training/time uncertainty"
        ),
        "quality": quality_summary(panel),
        "eligible_examples": len(examples),
        "missing_or_zero_outcome_policy": "exclude; never impute target",
        "panel_sha256": hashlib.sha256(panel.to_csv(index=False).encode()).hexdigest(),
        "source_sha256": source_hash.hexdigest(),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]
        },
        "limitations": [
            "Fiscal-year ordering is not a point-in-time investment backtest.",
            "Current DART data may include restatements unavailable historically.",
            "Employee definitions and source coverage require manual audit.",
            "Same firms may recur over time; unseen-firm generalization is not measured.",
            "Associations and permutation importance are not causal effects.",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    )
    disclaimer = (
        "합성 데이터로 실행 경로를 검증한 예제이며 실제 기업에 대한 결론이 아닙니다."
        if kind == "synthetic"
        else "현재 DART 스냅샷에 대한 회고적 분석입니다. 과거 시점 실전 예측 성능이 아닙니다."
    )
    result_table = table(
        pd.DataFrame(
            [{"model": "selected", **metrics}] + [dict(model=k, **v) for k, v in baselines.items()]
        )
    )
    report = f"""# Financial & employment analysis

> **{label}**

{disclaimer}

## Research question
전년도 재무·고용 정보가 다음 해 적자 여부를 구분하는 데 도움이 되는가?
직원수 추가 효과는 아래 financial / financial_employment 검증 결과를 비교합니다.

## Selection on earlier years
{table(comparison)}

## Held-out year: {test_year}
선택 모델: **{best.model} / {best.features}**. 양성 클래스는 적자, 임계값은 0.5 고정.

{result_table}

![Evaluation](evaluation.png)
![Data quality and interpretation](quality_importance.png)

## Employment / income change association
직원수 변화율과 전기 자산으로 나눈 순이익 증감의 연도별 상관계수입니다.
완전한 관측 쌍만 사용합니다. 인과관계나 전체 상장사에 대한 대표성을 의미하지 않습니다.

{table(correlations)}

## Reproducibility and limits
- Missing/zero outcomes excluded; feature imputation fitted within each training fold.
- Input hash, source hash, versions, confidence intervals: `metrics.json`.
- The cohort is not necessarily representative; firms repeat across years.
- Annual reports become available after fiscal year-end; current snapshots may be restated.
- Employment scope needs source-document checks even when OFS is used.
- Bootstrap intervals condition on this fitted model and test year's class composition.
- Correlated predictors can share importance. Negative importance is retained.
- Do not tune the model after viewing this held-out result and still call it untouched.
"""
    (output / "report.md").write_text(report)
    return result
