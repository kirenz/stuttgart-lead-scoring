"""Train and evaluate the lead-scoring model, run the data checks, score every lead.

Run from the repository root:  uv run train.py
Writes artifacts/ (model, scored leads, report.json) and reports/model_report.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split

from lead_scoring.geo import nearest_customer
from lead_scoring.model import (EXCLUDED, FEATURES, TARGET, contributions, make_model, rule_score,
                                top_reasons, top_share_lift, wilson)

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
SEED = 7
MIN_REGION_LEADS = 50          # below this, region numbers are flagged as unreliable
LEAK_AUC = 0.85                # single-column AUC above this is suspicious
PREFER_SIMPLE_MARGIN = 0.01    # gradient boosting must beat logistic regression by this much
MODEL_NAMES = {"logreg": "Logistic regression", "gboost": "Gradient boosting"}


def single_column_auc(df: pd.DataFrame, col: str, cv: StratifiedKFold) -> float:
    """Out-of-fold AUC of one column alone (categories are target-encoded inside the folds)."""
    y = df[TARGET].to_numpy()
    if pd.api.types.is_numeric_dtype(df[col]):
        auc = roc_auc_score(y, df[col])
        return float(max(auc, 1 - auc))
    oof = np.zeros(len(df))
    for tr, te in cv.split(df, y):
        rates = df.iloc[tr].groupby(col)[TARGET].mean()
        oof[te] = df.iloc[te][col].map(rates).fillna(y[tr].mean()).to_numpy()
    return float(roc_auc_score(y, oof))


def main() -> dict:
    leads = pd.read_csv(ROOT / "data/leads.csv")
    customers = pd.read_csv(ROOT / "data/customers.csv")
    y = leads[TARGET].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    report: dict = {"n_leads": len(leads), "n_customers": len(customers), "base_rate": float(y.mean())}

    # ---- 0. basic data checks
    near = nearest_customer(leads, customers)
    gap = (near["nearest_customer_km_check"] - leads["distance_to_nearest_customer_km"]).abs()
    report["data_checks"] = {
        "missing_values": int(leads.isna().sum().sum()),
        "duplicate_lead_ids": int(leads["lead_id"].duplicated().sum()),
        "duplicate_companies": int(leads["company"].duplicated().sum()),
        "distance_column_max_gap_km": round(float(gap.max()), 2),
    }

    # ---- 1. leakage
    stage = (leads.groupby("deal_stage")[TARGET].agg(["size", "mean"])
             .rename(columns={"size": "leads", "mean": "conversion_rate"})
             .sort_values("conversion_rate", ascending=False))
    screen = {c: single_column_auc(leads, c, cv)
              for c in leads.columns if c not in (TARGET, "lead_id", "company", "lat", "lon")}
    with_stage = FEATURES + ["deal_stage"]
    auc_with_stage = cross_val_score(make_model("logreg", with_stage), leads[with_stage], y,
                                     cv=cv, scoring="roc_auc").mean()
    auc_without = cross_val_score(make_model("logreg"), leads[FEATURES], y, cv=cv, scoring="roc_auc").mean()
    report["leakage"] = {
        "deal_stage_table": stage.round(3).reset_index().to_dict("records"),
        "single_column_auc": {k: round(v, 3) for k, v in sorted(screen.items(), key=lambda kv: -kv[1])},
        "flagged_columns": [c for c, v in screen.items() if v > LEAK_AUC],
        "cv_auc_logreg_with_deal_stage": round(float(auc_with_stage), 3),
        "cv_auc_logreg_without_deal_stage": round(float(auc_without), 3),
        "excluded_columns": EXCLUDED,
    }

    # ---- 2. small regions
    reg = leads.groupby("region")[TARGET].agg(["size", "sum"]).rename(columns={"size": "leads", "sum": "won"})
    reg["conversion_rate"] = reg["won"] / reg["leads"]
    ci = [wilson(int(k), int(n)) for k, n in zip(reg["won"], reg["leads"])]
    reg["ci_low"], reg["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    reg["small"] = reg["leads"] < MIN_REGION_LEADS
    reg = reg.sort_values("leads", ascending=False)
    with_region = FEATURES + ["region"]
    auc_with_region = cross_val_score(make_model("logreg", with_region), leads[with_region], y,
                                      cv=cv, scoring="roc_auc").mean()
    report["regions"] = {
        "min_leads": MIN_REGION_LEADS,
        "table": reg.round(3).reset_index().to_dict("records"),
        "small_regions": reg.index[reg["small"]].tolist(),
        "cv_auc_logreg_with_region": round(float(auc_with_region), 3),
    }

    # ---- 3. model comparison on held-out leads
    X = leads[FEATURES]
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, leads.index, test_size=0.25, stratify=y, random_state=SEED)
    rule_te = rule_score(leads.loc[idx_te])
    rule_rate, rule_lift = top_share_lift(y_te, rule_te)
    # The test top 10 % is only 50 leads; out-of-fold scores on all leads give 200 and a steadier lift.
    rule_oof_rate, rule_oof_lift = top_share_lift(y, rule_score(leads))
    models, comp = {}, {}
    for kind in ("logreg", "gboost"):
        m = make_model(kind).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        rate, lift = top_share_lift(y_te, p)
        cv_auc = cross_val_score(make_model(kind), X, y, cv=cv, scoring="roc_auc")
        oof_p = cross_val_predict(make_model(kind), X, y, cv=cv, method="predict_proba")[:, 1]
        oof_rate, oof_lift = top_share_lift(y, oof_p)
        fpr, tpr, _ = roc_curve(y_te, p)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 120)).astype(int))
        comp[kind] = {
            "name": MODEL_NAMES[kind],
            "test_auc": round(float(roc_auc_score(y_te, p)), 3),
            "cv_auc_mean": round(float(cv_auc.mean()), 3),
            "cv_auc_std": round(float(cv_auc.std()), 3),
            "top10_conversion": round(rate, 3),
            "top10_lift": round(lift, 2),
            "top10_lift_vs_rule": round(rate / rule_rate, 2),
            "oof_top10_conversion": round(oof_rate, 3),
            "oof_top10_lift": round(oof_lift, 2),
            "oof_top10_lift_vs_rule": round(oof_rate / rule_oof_rate, 2),
            "roc": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
        }
        models[kind] = (m, p)
    fpr, tpr, _ = roc_curve(y_te, rule_te)
    comp["rule"] = {
        "name": "Rule: demo requested, then most website visits",
        "test_auc": round(float(roc_auc_score(y_te, rule_te)), 3),
        "top10_conversion": round(rule_rate, 3),
        "top10_lift": round(rule_lift, 2),
        "oof_top10_conversion": round(rule_oof_rate, 3),
        "oof_top10_lift": round(rule_oof_lift, 2),
        "roc": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
    }
    # cumulative gains on the test set, for the chart in the app
    share = np.linspace(0.02, 1, 50)
    gains = {"share": share.round(2).tolist()}
    for kind, score in [("logreg", models["logreg"][1]), ("gboost", models["gboost"][1]), ("rule", rule_te)]:
        gains[kind] = [round(top_share_lift(y_te, score, s)[0] * max(1, int(round(s * len(y_te)))) / y_te.sum(), 4)
                       for s in share]
    report["comparison"] = comp
    report["gains"] = gains
    report["test_size"] = int(len(y_te))

    gap_auc = comp["gboost"]["cv_auc_mean"] - comp["logreg"]["cv_auc_mean"]
    chosen = "gboost" if gap_auc > PREFER_SIMPLE_MARGIN else "logreg"
    report["chosen_model"] = chosen
    report["choice_reason"] = (
        f"{MODEL_NAMES[chosen]} chosen: cross-validated AUC gap gradient boosting minus logistic regression "
        f"is {gap_auc:+.3f}; gradient boosting is only preferred above +{PREFER_SIMPLE_MARGIN:.2f}, "
        "because the simpler model is easier to explain and more stable.")

    # ---- 4. final model on all leads, scores and reasons
    final = make_model(chosen).fit(X, y)
    # Scores shown in the app are out-of-fold: each lead is scored by a model that never saw it.
    oof = cross_val_predict(make_model(chosen), X, y, cv=cv, method="predict_proba")[:, 1]
    bg = leads.sample(150, random_state=SEED)
    contrib = contributions(final, leads, bg)
    reasons = top_reasons(leads, contrib)
    scored = pd.concat([leads.drop(columns=[TARGET, "deal_stage"]).reset_index(drop=True),
                        near.drop(columns="nearest_customer_km_check").reset_index(drop=True),
                        reasons], axis=1)
    scored["score"] = (oof * 100).round(1)
    scored["score_final_model"] = (final.predict_proba(X)[:, 1] * 100).round(1)
    scored = scored.sort_values("score", ascending=False)

    if chosen == "logreg":
        names = final["prep"].get_feature_names_out()
        coef = final["clf"].coef_[0]
        report["coefficients"] = {n.split("__", 1)[1]: round(float(c), 3)
                                  for n, c in sorted(zip(names, coef), key=lambda t: -abs(t[1]))}

    ART.mkdir(exist_ok=True)
    joblib.dump(final, ART / "model.joblib")
    scored.to_csv(ART / "scored_leads.csv", index=False)
    (ART / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(report)
    return report


def write_markdown(r: dict) -> None:
    c, lk, rg = r["comparison"], r["leakage"], r["regions"]
    lines = [
        "# Model report",
        "",
        f"{r['n_leads']} leads, base conversion rate {r['base_rate']:.1%}. "
        f"Held-out test set: {r['test_size']} leads (stratified 25 %). CV: 5-fold stratified.",
        "",
        "## Model comparison",
        "",
        "Top-10 % columns: held-out test set (50 leads) / out-of-fold on all leads (200 leads).",
        "",
        "| Model | Test ROC-AUC | CV ROC-AUC | Top-10 % conversion | Lift vs. base | Lift vs. rule |",
        "|---|---|---|---|---|---|",
    ]
    for k in ("logreg", "gboost"):
        m = c[k]
        lines.append(f"| {m['name']} | {m['test_auc']:.3f} | {m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f} | "
                     f"{m['top10_conversion']:.1%} / {m['oof_top10_conversion']:.1%} | "
                     f"{m['top10_lift']:.2f}× / {m['oof_top10_lift']:.2f}× | "
                     f"{m['top10_lift_vs_rule']:.2f}× / {m['oof_top10_lift_vs_rule']:.2f}× |")
    m = c["rule"]
    lines += [f"| {m['name']} | {m['test_auc']:.3f} | | {m['top10_conversion']:.1%} / {m['oof_top10_conversion']:.1%} | "
              f"{m['top10_lift']:.2f}× / {m['oof_top10_lift']:.2f}× | 1.00× |",
              "", r["choice_reason"], "",
              "## Leakage check", "",
              "Conversion rate by `deal_stage`:", "",
              "| deal_stage | leads | conversion rate |", "|---|---|---|"]
    lines += [f"| {s['deal_stage']} | {s['leads']} | {s['conversion_rate']:.1%} |" for s in lk["deal_stage_table"]]
    lines += ["", "Single-column ROC-AUC (out-of-fold):", "",
              "| column | AUC |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in lk["single_column_auc"].items()]
    lines += ["",
              f"Flagged (AUC > {LEAK_AUC}): {', '.join(lk['flagged_columns']) or 'none'}. "
              f"Logistic regression CV AUC with `deal_stage`: {lk['cv_auc_logreg_with_deal_stage']:.3f}, "
              f"without: {lk['cv_auc_logreg_without_deal_stage']:.3f}.",
              "",
              "`deal_stage` is only known late in the sales cycle (`Contract`, `Lost`, `Negotiation` are "
              "outcomes or near-outcomes), so a model that uses it would look excellent offline and be "
              "useless for new leads. It is excluded from training and from the app.",
              "", "Excluded columns:", ""]
    lines += [f"- `{k}`: {v}" for k, v in lk["excluded_columns"].items()]
    lines += ["", f"## Regions (flagged below {rg['min_leads']} leads)", "",
              "| region | leads | won | conversion | 95 % CI | |", "|---|---|---|---|---|---|"]
    for s in rg["table"]:
        flag = "⚠ too few leads" if s["small"] else ""
        lines.append(f"| {s['region']} | {s['leads']} | {s['won']} | {s['conversion_rate']:.1%} | "
                     f"{s['ci_low']:.0%} to {s['ci_high']:.0%} | {flag} |")
    lines += ["", f"Adding `region` as a feature: CV AUC {rg['cv_auc_logreg_with_region']:.3f} "
              f"(without: {lk['cv_auc_logreg_without_deal_stage']:.3f}). Region stays out of the model, "
              "so a lucky streak in a tiny region does not inflate its scores.",
              "", "## Data checks", ""]
    lines += [f"- {k}: {v}" for k, v in r["data_checks"].items()]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/model_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    rep = main()
    c = rep["comparison"]
    for k in ("logreg", "gboost", "rule"):
        print(f"{c[k]['name']:<48} test AUC {c[k]['test_auc']:.3f}  top10 lift test {c[k]['top10_lift']:.2f}x"
              f"  oof {c[k]['oof_top10_lift']:.2f}x")
    print("chosen:", rep["chosen_model"], "|", rep["choice_reason"])
    print("leakage flagged:", rep["leakage"]["flagged_columns"],
          "| AUC with deal_stage", rep["leakage"]["cv_auc_logreg_with_deal_stage"])
    print("small regions:", rep["regions"]["small_regions"])
    print("data checks:", rep["data_checks"])
