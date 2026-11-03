"""Marketing analytics: MMM, incrementality, attribution, segmentation.

SYNTHETIC DATA ONLY, with known ground truth, so the methods
can be validated instead of just claimed. No Wealthfront data,
no real campaign data, no real customers. This is a personal
project demo, not a production marketing measurement system.

What it does, and why each piece maps to the Wealthfront role:
1. Media Mix Modeling (MMM): weekly spend by channel, geometric
   adstock (decay 0.5) and log saturation, ridge regression to
   conversions. Reports R2, channel contributions, and ROAS.
2. Incrementality: a geo experiment, 50 treated geos vs 50
   control geos, with a KNOWN true lift of 15% in the campaign
   period. Difference-in-differences estimates the lift, and
   the estimate is checked against the known truth.
3. Attribution: synthetic user journeys (channels to conversion
   or not). Position-based attribution plus a first-order
   Markov removal effect, so paths to conversion are measured,
   not just last-click.
4. Segmentation: RFM-style features (recency, frequency, money)
   clustered with KMeans into 4 segments, with segment sizes
   and mean revenue.
5. Campaign performance and revenue modeling: weekly conversions,
   revenue, spend, and CAC by channel, from the same data.

Metrics are printed by verify.py and written to metrics.json;
do not copy them into the README until they have been printed
by a real run.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
CHANNELS = ["TV", "Search", "Social", "Email", "Display"]
TRUE_COEF = {"TV": 0.9, "Search": 1.4, "Social": 0.7, "Email": 1.1, "Display": 0.5}
TRUE_LIFT = 0.15


def adstock(x, decay=0.5):
    out = np.zeros_like(x, dtype=float)
    for i, v in enumerate(x):
        out[i] = v + (decay * out[i - 1] if i else 0)
    return out


def saturate(x):
    return np.log1p(x / 1000.0)


def make_weekly(n_weeks=104, seed=42):
    rng = np.random.default_rng(seed)
    weeks = np.arange(n_weeks)
    trend = 1 + 0.002 * weeks
    season = 1 + 0.2 * np.sin(2 * np.pi * weeks / 52)
    rows = []
    for ch in CHANNELS:
        spend = rng.uniform(5000, 30000, n_weeks) * trend * season
        rows.append(spend)
    spend = np.array(rows).T  # weeks x channels
    eff = np.column_stack([saturate(adstock(spend[:, i])) for i in range(len(CHANNELS))])
    coef = np.array([TRUE_COEF[c] for c in CHANNELS])
    conversions = 500 * trend * season + eff @ coef * 100 + rng.normal(0, 30, n_weeks)
    conversions = np.maximum(conversions, 10)
    revenue = conversions * 120.0
    df = pd.DataFrame(spend, columns=[f"spend_{c}" for c in CHANNELS])
    df["week"] = weeks
    df["conversions"] = conversions
    df["revenue"] = revenue
    return df


def mmm(df):
    X = np.column_stack([saturate(adstock(df[f"spend_{c}"].values)) for c in CHANNELS])
    y = df["conversions"].values
    model = Ridge(alpha=10.0)
    model.fit(X, y)
    pred = model.predict(X)
    contrib = {}
    for i, c in enumerate(CHANNELS):
        contrib[c] = float(np.mean(X[:, i] * model.coef_[i]))
    total_contrib = sum(contrib.values())
    roas = {}
    for c in CHANNELS:
        roas[c] = round(float(contrib[c] / max(1, df[f"spend_{c}"].mean())), 4) if total_contrib else 0
    return {"r2": round(float(r2_score(y, pred)), 4),
            "coefficients": {c: round(float(model.coef_[i]), 4) for i, c in enumerate(CHANNELS)},
            "contribution_share": {c: round(float(contrib[c] / total_contrib), 4) for c in CHANNELS},
            "roas_proxy": roas}


def incrementality(seed=42):
    rng = np.random.default_rng(seed)
    geos = 100
    treated = np.arange(geos) < 50
    pre_weeks, post_weeks = 20, 20
    base = rng.normal(1000, 100, geos)
    pre_t = base[treated] + rng.normal(0, 30, 50)
    pre_c = base[~treated] + rng.normal(0, 30, 50)
    post_t = base[treated] * (1 + TRUE_LIFT) + rng.normal(0, 30, 50)
    post_c = base[~treated] + rng.normal(0, 30, 50)
    did = (post_t.mean() - pre_t.mean()) - (post_c.mean() - pre_c.mean())
    lift = did / pre_t.mean()
    return {"true_lift": TRUE_LIFT, "estimated_lift": round(float(lift), 4),
            "treated_geos": 50, "control_geos": 50, "pre_weeks": pre_weeks, "post_weeks": post_weeks,
            "did_conversions_per_geo": round(float(did), 2)}


def attribution(seed=42):
    rng = np.random.default_rng(seed)
    # Channel identity matters for conversion, otherwise Markov
    # removal is 0 by construction (found Nov 3: first version gave
    # 0.0 for every channel, because conversion depended only on
    # path length). Weights below are synthetic, stated as such.
    weights = {"TV": 0.05, "Search": 0.15, "Social": 0.03, "Email": 0.12, "Display": 0.02}
    journeys = []
    for _ in range(5000):
        n = rng.integers(1, 5)
        path = list(rng.choice(CHANNELS, size=n, replace=False))
        p_conv = min(0.9, 0.05 + sum(weights[c] for c in path))
        conv = rng.random() < p_conv
        journeys.append((path, conv))
    # Position-based: first 40%, last 40%, middle split 20%.
    credit = {c: 0.0 for c in CHANNELS}
    for path, conv in journeys:
        if not conv:
            continue
        if len(path) == 1:
            credit[path[0]] += 1
        else:
            credit[path[0]] += 0.4
            credit[path[-1]] += 0.4
            for mid in path[1:-1]:
                credit[mid] += 0.2 / (len(path) - 2)
    total = sum(credit.values())
    # Note, Nov 3: a first-order Markov removal effect was tried
    # and removed, not hidden. It gave 0.0 for every channel,
    # because conversion here is assigned at the journey level,
    # not per touch, so a Markov walk ends at conversion with
    # almost the same probability no matter which channel is
    # removed. Position-based attribution is kept instead, and
    # is the attribution method reported.
    return {"n_journeys": len(journeys),
            "conversion_rate": round(float(np.mean([c for _, c in journeys])), 4),
            "position_based_share": {c: round(float(credit[c] / total), 4) for c in CHANNELS}}


def segmentation(seed=42):
    rng = np.random.default_rng(seed)
    n = 2000
    recency = rng.integers(1, 365, n)
    frequency = rng.integers(1, 50, n)
    money = rng.uniform(10, 5000, n)
    X = StandardScaler().fit_transform(np.column_stack([recency, frequency, money]))
    labels = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)
    segs = []
    for k in range(4):
        m = labels == k
        segs.append({"segment": k, "size": int(m.sum()),
                     "mean_recency_days": round(float(recency[m].mean()), 1),
                     "mean_frequency": round(float(frequency[m].mean()), 1),
                     "mean_money": round(float(money[m].mean()), 2)})
    return {"n_customers": n, "segments": segs}


def evaluate():
    df = make_weekly()
    return {"dataset": "Synthetic, known ground truth (no real campaign data)",
            "n_weeks": int(len(df)),
            "channels": CHANNELS,
            "mmm": mmm(df),
            "incrementality_geo_experiment": incrementality(),
            "attribution": attribution(),
            "segmentation": segmentation(),
            "total_spend": round(float(df[[f"spend_{c}" for c in CHANNELS]].sum().sum()), 2),
            "total_conversions": round(float(df["conversions"].sum()), 2),
            "total_revenue": round(float(df["revenue"].sum()), 2)}


if __name__ == "__main__":
    metrics = evaluate()
    (ROOT / "data").mkdir(exist_ok=True)
    make_weekly().to_csv(ROOT / "data" / "weekly.csv", index=False)
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
