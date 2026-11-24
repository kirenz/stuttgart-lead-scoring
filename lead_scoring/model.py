"""Features, models, evaluation metrics and plain-language explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "converted"
CATEGORICAL = ["company_size", "industry", "source"]
NUMERIC = ["website_visits", "demo_requested", "distance_to_nearest_customer_km"]
FEATURES = CATEGORICAL + NUMERIC
NEAR_KM = 5.0  # "close to an existing customer" in the reason texts

# Not used as features, with the reason (documented in the model report).
EXCLUDED = {
    "deal_stage": "Leakage: set late in the sales process, it almost reveals the outcome.",
    "region": "Some regions are tiny (Ulm: 9 leads); geography enters via distance to the nearest customer.",
    "city": "Same as region.",
    "lat": "Raw coordinates; geography enters via distance to the nearest customer.",
    "lon": "Raw coordinates; geography enters via distance to the nearest customer.",
    "lead_id": "Identifier.",
    "company": "Identifier.",
}


def make_model(kind: str, features: list[str] | None = None) -> Pipeline:
    features = features or FEATURES
    cats = [c for c in features if c not in NUMERIC]
    nums = [c for c in features if c in NUMERIC]
    if kind == "logreg":
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
            ("num", StandardScaler(), nums),
        ])
        clf = LogisticRegression(C=1.0, max_iter=2000)
    elif kind == "gboost":
        prep = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
            ("num", "passthrough", nums),
        ])
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_depth=3, min_samples_leaf=30,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            categorical_features=list(range(len(cats))), random_state=7,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("prep", prep), ("clf", clf)])


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """The rep's rule of thumb: demo requests first, then the most website visits."""
    return df["demo_requested"].to_numpy() * 100 + df["website_visits"].to_numpy()


def top_share_lift(y: np.ndarray, score: np.ndarray, share: float = 0.10, seed: int = 0) -> tuple[float, float]:
    """Conversion rate in the top `share` of leads by score, and its lift over the base rate.

    Ties are broken at random, so a coarse rule does not profit from the row order.
    """
    y = np.asarray(y)
    jitter = np.random.default_rng(seed).random(len(y)) * 1e-9
    n = max(1, int(round(share * len(y))))
    top = np.argsort(-(np.asarray(score, float) + jitter))[:n]
    rate = y[top].mean()
    return float(rate), float(rate / y.mean())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return float(c - h), float(c + h)


# ---------- explanations ----------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def contributions(model: Pipeline, X: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """Effect of each feature on the log-odds of one lead, compared with typical leads.

    For feature j: logit f(x) minus the mean logit when x_j is replaced by the values of
    the background leads. For logistic regression this equals coef * (x - mean) exactly;
    for gradient boosting it is a model-agnostic approximation.
    """
    X = X[FEATURES].reset_index(drop=True)
    bg = background[FEATURES].reset_index(drop=True)
    base = _logit(model.predict_proba(X)[:, 1])
    out = {}
    n, m = len(X), len(bg)
    for col in FEATURES:
        rep = X.loc[np.repeat(np.arange(n), m)].reset_index(drop=True)
        rep[col] = np.tile(bg[col].to_numpy(), n)
        swapped = _logit(model.predict_proba(rep)[:, 1]).reshape(n, m).mean(axis=1)
        out[col] = base - swapped
    return pd.DataFrame(out)


def reason_text(col: str, row: pd.Series, effect: float) -> str:
    up = effect > 0
    v = row[col]
    if col == "company_size":
        return (f"Company size {v} employees converts above average" if up
                else f"Company size {v} employees converts below average")
    if col == "source":
        return (f"Came in via {v.lower()}, a strong channel for us" if up
                else f"Came in via {v.lower()}, a weak channel for us")
    if col == "industry":
        return (f"{v} is a strong industry for us" if up
                else f"{v} converts below average")
    if col == "website_visits":
        n = int(v)
        if up:
            return f"Visited our website {n} times, clear interest"
        return "No website visits yet" if n == 0 else f"Only {n} website visit{'s' if n != 1 else ''} so far"
    if col == "demo_requested":
        return "Asked for a demo" if up else "Has not asked for a demo yet"
    if col == "distance_to_nearest_customer_km":
        # The learned distance effect is close to zero; only claim what matches the value.
        if up and v <= NEAR_KM:
            return f"Existing customer only {v:.1f} km away, a ready reference"
        if not up and v > NEAR_KM:
            return f"Nearest existing customer is {v:.0f} km away"
        return f"Nearest existing customer {v:.0f} km away, plays a minor role"
    raise KeyError(col)


def top_reasons(leads: pd.DataFrame, contrib: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """The k features with the largest absolute effect, as plain-language text."""
    rows = []
    for i, (_, lead) in enumerate(leads.reset_index(drop=True).iterrows()):
        eff = contrib.iloc[i]
        order = eff.abs().sort_values(ascending=False).index[:k]
        rec = {}
        for r, col in enumerate(order, start=1):
            rec[f"reason_{r}"] = reason_text(col, lead, eff[col])
            rec[f"reason_{r}_up"] = bool(eff[col] > 0)
        rows.append(rec)
    return pd.DataFrame(rows)
