"""Exploratory company-wise nested CV for the single observed 2021 -> 2022 transition."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold

from data_final.analysis import candidates, scores, table
from data_final.legacy import LEGACY_EMPLOYMENT, LEGACY_FINANCIAL, build_legacy_examples

DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def model_specs(seed: int) -> dict:
    models = candidates(seed)
    return {
        "logistic_financial": (models["logistic"], LEGACY_FINANCIAL),
        "logistic_employment": (models["logistic"], LEGACY_FINANCIAL + LEGACY_EMPLOYMENT),
        "boosting_financial": (models["hist_gradient_boosting"], LEGACY_FINANCIAL),
        "boosting_employment": (
            models["hist_gradient_boosting"],
            LEGACY_FINANCIAL + LEGACY_EMPLOYMENT,
        ),
    }


def evaluate_examples(examples: pd.DataFrame, seeds=DEFAULT_SEEDS, nested=True):
    if examples.company.duplicated().any():
        raise ValueError("One row per company is required for disjoint company folds")
    y = examples.loss_2022.to_numpy()
    if len(np.unique(y)) != 2 or np.bincount(y).min() < 8:
        raise ValueError("Need at least eight companies in each class for this CV design")
    repeated, predictions, assignments, selections = [], [], [], []
    for seed in seeds:
        specs = model_specs(seed)
        if not nested:
            specs = {k: v for k, v in specs.items() if k.startswith("logistic")}
        names = (
            list(specs)
            + ["class_prior", "previous_loss"]
            + (["nested_selection"] if nested else [])
        )
        probabilities = {name: np.full(len(examples), np.nan) for name in names}
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(outer.split(examples, y)):
            train, test = examples.iloc[train_idx], examples.iloc[test_idx]
            assert set(train.company).isdisjoint(test.company)
            prior = float(train.loss_2022.mean())
            for split, idx in [("train", train_idx), ("test", test_idx)]:
                assignments.extend(
                    {
                        "seed": seed,
                        "fold": fold,
                        "split": split,
                        "company": examples.iloc[i].company,
                    }
                    for i in idx
                )
            inner_scores = {}
            for name, (estimator, columns) in specs.items():
                if nested:
                    ap = []
                    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold)
                    for inner_train, inner_valid in inner.split(train, train.loss_2022):
                        fitted = clone(estimator).fit(
                            train.iloc[inner_train][columns], train.iloc[inner_train].loss_2022
                        )
                        pred = fitted.predict_proba(train.iloc[inner_valid][columns])[:, 1]
                        ap.append(
                            scores(train.iloc[inner_valid].loss_2022, pred)["average_precision"]
                        )
                    inner_scores[name] = float(np.mean(ap))
                fitted = clone(estimator).fit(train[columns], train.loss_2022)
                probabilities[name][test_idx] = fitted.predict_proba(test[columns])[:, 1]
            probabilities["class_prior"][test_idx] = prior
            probabilities["previous_loss"][test_idx] = test.previous_loss.fillna(prior)
            if nested:
                chosen = max(inner_scores, key=inner_scores.get)
                probabilities["nested_selection"][test_idx] = probabilities[chosen][test_idx]
                selections.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "selected": chosen,
                        "inner_mean_ap": inner_scores[chosen],
                        "inner_scores": json.dumps(inner_scores, sort_keys=True),
                    }
                )
        for name, probability in probabilities.items():
            assert np.isfinite(probability).all()
            repeated.append({"seed": seed, "model": name, **scores(y, probability)})
            predictions.extend(
                {
                    "seed": seed,
                    "model": name,
                    "company": company,
                    "loss_2022": int(label),
                    "loss_probability": float(prob),
                }
                for company, label, prob in zip(examples.company, y, probability, strict=True)
            )
    return (
        pd.DataFrame(repeated),
        pd.DataFrame(predictions),
        pd.DataFrame(assignments),
        pd.DataFrame(selections),
    )


def summarize(repeated: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "loss_recall",
        "loss_precision",
        "loss_f1",
        "balanced_accuracy",
        "brier",
    ]
    rows = []
    for name, group in repeated.groupby("model", sort=True):
        row = {"model": name, "repeats": len(group), "n": int(group.n.iloc[0])}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def employment_association(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    before = panel[panel.year.eq(2021)].set_index("company")
    after = panel[panel.year.eq(2022)].set_index("company")
    points = pd.DataFrame(index=before.index)
    points["employee_growth"] = (after.employees - before.employees) / before.employees.where(
        before.employees.gt(0)
    )
    points["income_change_scaled"] = (after.net_income - before.net_income) / before.assets.where(
        before.assets.gt(0)
    )
    points["repeated_employee_pair_flag"] = before.employee_repeated_pair_flag
    rows = []
    for name, subset in [
        ("all_complete_pairs", points),
        ("exclude_repeated_employee_pairs", points[~points.repeated_employee_pair_flag]),
    ]:
        complete = subset[["employee_growth", "income_change_scaled"]].dropna()
        if len(complete) < 3:
            raise ValueError("Insufficient complete observations for correlation")
        x, y = complete.to_numpy().T
        pearson, spearman = pearsonr(x, y), spearmanr(x, y)
        rows.append(
            {
                "sample": name,
                "n": len(complete),
                "pearson_r": float(pearson.statistic),
                "pearson_p_exploratory": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p_exploratory": float(spearman.pvalue),
            }
        )
    return {
        "comparisons": rows,
        "interpretation": "Exploratory association, not causation.",
    }, points.reset_index()


def analyze_legacy(panel: pd.DataFrame, output: Path, seeds=DEFAULT_SEEDS) -> dict:
    examples, eligibility = build_legacy_examples(panel)
    repeated, predictions, assignments, selections = evaluate_examples(examples, seeds)
    summary = summarize(repeated)
    sensitivity = examples[~examples.employee_repeated_pair_flag].reset_index(drop=True)
    sens_repeated, _, _, _ = evaluate_examples(sensitivity, seeds, nested=False)
    sens_summary = summarize(sens_repeated)
    association, points = employment_association(panel)
    paired = repeated.pivot(index="seed", columns="model", values="average_precision")
    paired_auc = repeated.pivot(index="seed", columns="model", values="roc_auc")
    delta = pd.DataFrame(
        {
            "seed": paired.index,
            "employment_ap_delta": (paired.logistic_employment - paired.logistic_financial),
            "employment_auc_delta": (
                paired_auc.logistic_employment - paired_auc.logistic_financial
            ),
        }
    )
    delta = delta.reset_index(drop=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "examples": examples,
        "eligibility": eligibility,
        "repeat_metrics": repeated,
        "model_summary": summary,
        "oof_predictions": predictions,
        "fold_assignments": assignments,
        "inner_selection": selections,
        "employment_ablation": delta,
        "employee_sensitivity": sens_summary,
        "association_points": points,
    }.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "associations.json").write_text(json.dumps(association, indent=2))
    result = {
        "data_kind": "user_supplied_legacy_unverified_metadata",
        "experiment": "2021 predictors -> 2022 loss; company-wise repeated nested CV",
        "n_companies": len(examples),
        "loss_count": int(examples.loss_2022.sum()),
        "loss_prevalence": float(examples.loss_2022.mean()),
        "seeds": list(seeds),
        "outer_folds": 5,
        "inner_folds": 3,
        "threshold": 0.5,
        "selection_metric": "inner average precision",
        "financial_features": LEGACY_FINANCIAL,
        "employment_features": LEGACY_EMPLOYMENT,
        "exclusions": eligibility.exclusion_reason.value_counts().to_dict(),
        "model_summary": summary.to_dict("records"),
        "employment_logistic_ap_delta_mean": float(delta.employment_ap_delta.mean()),
        "employment_logistic_ap_delta_range": [
            float(delta.employment_ap_delta.min()),
            float(delta.employment_ap_delta.max()),
        ],
        "sensitivity_n": len(sensitivity),
        "sensitivity_summary": sens_summary.to_dict("records"),
        "associations": association,
        "uncertainty": (
            "Repeat min/max describe split sensitivity; they are not confidence intervals."
        ),
        "input_sha256": hashlib.sha256(panel.to_csv(index=False).encode()).hexdigest(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["pandas", "numpy", "scikit-learn", "scipy", "matplotlib"]
        },
        "limitations": [
            "Only one fiscal-year transition; no future-year external validation.",
            "Currency, consolidation, actual source year and filing dates are unknown.",
            "Ratios assume consistent within-company units and scope; this is unverified.",
            "Employee values may already contain unrecoverable upstream substitutions.",
            "Companies are identified by names, not externally verified identifiers.",
            "Repeated folds and related firms are dependent; no causal inference.",
        ],
    }
    result["source_sha256"] = hashlib.sha256(
        b"".join(
            p.name.encode() + p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))
        )
    ).hexdigest()
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    )
    _figures(summary, predictions, delta, points, output, seeds[0])
    compact = [
        "model",
        "n",
        "roc_auc_mean",
        "average_precision_mean",
        "accuracy_mean",
        "loss_recall_mean",
        "loss_f1_mean",
    ]
    report = f"""# 제공 자료의 실데이터 재분석

**2021 → 2022년의 단일 전환을 분석한 탐색 결과입니다. 공시 원문 메타데이터는 미검증입니다.**

분석 대상 {len(examples)}개 기업, 2022년 적자 {int(examples.loss_2022.sum())}개.
2021년의 비율형 재무 변수 4개와 직원수 1개만 허용했습니다.
원본 통화를 알 수 없어 기업 간 원시 금액 크기 비교는 제외했습니다.
같은 회사 안의 분자·분모 단위와 연결 범위가 같다는 가정은 남아 있습니다.

## 평가
5-fold 기업 분할을 {len(seeds)}개 seed로 반복했습니다.
회사는 한 repeat 안에서 한 번만 외부 평가를 받습니다.
외부 학습 기업 안에서 3-fold AP로 모델·피처 구성을 선택했습니다.
외부 평가 기업에는 선택 결과만 적용했습니다.
`nested_selection`은 이 선택 절차의 평가이며, 다른 행은 사전에 고정한 비교 실험입니다.
표는 repeat별 OOF 전체 점수의 평균입니다. 반복 범위는 신뢰구간이 아닙니다.

{table(summary[compact])}

![Model comparison](comparison.png)
![First-repeat out-of-fold evaluation](evaluation.png)

## 직원수의 추가 가치
동일한 로지스틱 모델과 동일한 분할에서 직원수 추가 전후 AP 차이의 평균은
{result["employment_logistic_ap_delta_mean"]:+.4f}입니다.
다른 모델 간 차이를 직원수의 효과라고 해석하지 않습니다.

{table(delta)}

반복된 직원수 쌍이 있는 기업을 제외한 {len(sensitivity)}개 기업의 민감도 분석:

{table(sens_summary[compact])}

## 고용 변화와 수익 변화의 관련성
직원수 변화율과 순이익 변화/2021년 자산의 상관계수입니다. p값은 탐색용입니다.

{table(pd.DataFrame(association["comparisons"]))}

![Association](association.png)

## 해석 범위
한 해의 기업 간 구분 능력을 측정했습니다.
새로운 경기 국면이나 여러 미래 연도의 성능은 확인하지 않았습니다.
반복된 직원수 값은 검토 신호이며 그것만으로 대체값이라고 확정하지 않았습니다.
금액의 통화, 연결/별도, 실제 수집 연도, 공시 시점을 원문에서 검증해야 합니다.
원본 95.24%는 같은 해 380개 항목을 사용한 다른 문제의 결과라서 이 점수와 증감 비교하지 않습니다.
"""
    (output / "report.md").write_text(report)
    return result


def _figures(summary, predictions, delta, points, output, first_seed):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ordered = summary.sort_values("average_precision_mean")
    for ax, metric, title in [
        (axes[0], "average_precision", "Average precision"),
        (axes[1], "roc_auc", "ROC-AUC"),
    ]:
        ax.barh(ordered.model, ordered[f"{metric}_mean"], color="#2563eb")
        ax.errorbar(
            ordered[f"{metric}_mean"],
            ordered.model,
            xerr=np.maximum(
                0,
                np.array(
                    [
                        ordered[f"{metric}_mean"] - ordered[f"{metric}_min"],
                        ordered[f"{metric}_max"] - ordered[f"{metric}_mean"],
                    ]
                ),
            ),
            fmt="none",
            color="#172554",
            capsize=3,
        )
        ax.set(xlim=(0, 1), xlabel=title)
    fig.suptitle("2021 -> 2022 exploratory CV | bars: mean; whiskers: repeat range")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    sample = predictions[
        (predictions.seed == first_seed) & predictions.model.eq("nested_selection")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    RocCurveDisplay.from_predictions(sample.loss_2022, sample.loss_probability, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    PrecisionRecallDisplay.from_predictions(sample.loss_2022, sample.loss_probability, ax=axes[1])
    axes[1].axhline(sample.loss_2022.mean(), linestyle="--", color="gray")
    ConfusionMatrixDisplay.from_predictions(
        sample.loss_2022,
        sample.loss_probability >= 0.5,
        display_labels=["Profit", "Loss"],
        ax=axes[2],
        colorbar=False,
    )
    fig.suptitle(
        f"Nested selection OOF | seed={first_seed} | single transition, source metadata unverified"
    )
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)
    complete = points.dropna(subset=["employee_growth", "income_change_scaled"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for flagged, group in complete.groupby("repeated_employee_pair_flag"):
        ax.scatter(
            group.employee_growth,
            group.income_change_scaled,
            alpha=0.65,
            label="Repeated headcount pair" if flagged else "Other companies",
            color="#d97706" if flagged else "#0f766e",
        )
    ax.axhline(0, linewidth=0.6, color="gray")
    ax.axvline(0, linewidth=0.6, color="gray")
    ax.set(
        xlabel="Employee growth, 2021 -> 2022",
        ylabel="Net income change / 2021 assets",
        title="Exploratory association | no causal interpretation",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "association.png", dpi=160)
    plt.close(fig)
