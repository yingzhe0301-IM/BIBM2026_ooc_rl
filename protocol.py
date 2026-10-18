"""The three-stage certification protocol on the Schuster, Spiller, and Wells slices.

Stage 1 asks what the decision-point state supports. Stage 2 fits lower-burden rules on
the training patients of each leave-one-patient-out fold and retains one of three
prespecified classes by the patient-weighted doubly robust value of the burden-adjusted
reward Y - lambda0 C. Stage 3 certifies the rules with a paired two-stage cluster
bootstrap over patients, with the wild-cluster bound as a screen. Writes the result
tables to out/results and Fig 4 to the paper.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prep import OUT, load_checkpoint, save

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # embed TrueType, not Type 3

# Stage 2 retains one of the prespecified lower-burden classes. Audit classes are fit,
# evaluated, and certified the same way but never retained: state-free rules, the CQL
# penalty sweep (eta=0 is plain one-step Q-learning, a T-learner with linear base
# learners), a random-forest T-learner, and depth-limited policy trees.
CANDIDATES = ("always_low", "threshold", "one_step_cql")
CQL_ETA = 1.0
CQL_AUDIT_ETAS = {"cql_eta0": 0.0, "cql_eta0.1": 0.1, "cql_eta10": 10.0}
AUDIT = ("family_only", "cql_no_state", *CQL_AUDIT_ETAS, "t_learner_rf", "policy_tree_depth1", "policy_tree_depth2")
BASELINE = "always_high"
POLICIES = CANDIDATES + AUDIT + (BASELINE,)
ALPHA = 0.05
ALPHA_ADJUSTED = ALPHA / len(CANDIDATES)  # Bonferroni over the classes Stage 2 selects from
# (reading, bound, one-sided level): the paired certificate, the same certificate adjusted
# for retaining one of three classes, and the wild-cluster screen.
READINGS = (("certificate", "paired", ALPHA), ("certificate_adjusted", "paired", ALPHA_ADJUSTED),
            ("screen", "wild", ALPHA))
LAMBDA_RATIOS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
REFERENCE_REGION = (0.5, 2.0)  # the reading band Lambda in units of lambda0, a convention
# One bootstrap stream per policy. The CQL variants share one (common random numbers), so
# the eta sweep isolates the change in the learned rule from resampling noise.
BOOT_OFFSET = {"always_low": 1000, "threshold": 2000, "t_learner_rf": 3000, "one_step_cql": 4000,
               "cql_no_state": 4000, **dict.fromkeys(CQL_AUDIT_ETAS, 4000), "policy_tree_depth1": 5000,
               "policy_tree_depth2": 6000, "family_only": 7000}
# Hand-set search knobs, swept in knob_sensitivity: the number of within-family state
# quantiles (5% to 95%) scanned for cutpoints and the minimum wells per tree leaf.
THRESHOLD_GRID = 21
TREE_MIN_LEAF = 4
KNOB_SWEEP = {"threshold_grid": (11, 21, 41), "tree_min_leaf": (3, 4, 6)}
T_LEARNER_SEED = 42


def ridge(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    return Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), num_cols),
                                                 ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])),
                     ("ridge", Ridge(alpha=1.0))])


def iter_lopo(df: pd.DataFrame):
    for patient in sorted(df["patient"].unique()):
        held = df["patient"] == patient
        yield patient, df.loc[~held].reset_index(drop=True), df.loc[held].reset_index(drop=True)


# ---- Stage 1: what the decision-point state supports ------------------------------------

STAGE1_FEATURES = {"family_arm": [], "state": ["state"], "state_second_channel": ["state", "state2"]}


def stage1(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-patient-out ridge models of the endpoint on family and arm, adding the state
    and then the second channel where the slice has one. MAE measures level and rho_within,
    the Spearman correlation between prediction and endpoint inside the held-out patient,
    measures ordering. Per-patient r2 also charges the patient's unpredictable offset to the
    ordering and is kept for reference."""
    models = {name: num for name, num in STAGE1_FEATURES.items() if df[num].notna().all().all()}
    rows = []
    for _, train, test in iter_lopo(df):
        y = test["endpoint"].to_numpy(float)
        for name, num in models.items():
            cols = num + ["family", "arm"]
            pred = ridge(num, ["family", "arm"]).fit(train[cols], train["endpoint"]).predict(test[cols])
            rows.append({"model": name, "mae": mean_absolute_error(y, pred), "rho_within": spearmanr(pred, y)[0],
                         "r2": r2_score(y, pred)})
    return pd.DataFrame(rows).groupby("model", sort=False).mean().reset_index()


def within_cell_order(df: pd.DataFrame) -> dict[str, float]:
    """Model-free ordering check: Spearman correlation between state and endpoint inside each
    patient x family x arm cell, where the design is held fixed."""
    rho = np.array([spearmanr(g["state"], g["endpoint"])[0]
                    for _, g in df.groupby(["patient", "family", "arm"]) if len(g) >= 3])
    rho = rho[np.isfinite(rho)]
    return {"cell_rho_mean": float(rho.mean()), "cell_rho_frac_positive": float((rho > 0).mean()),
            "n_cells": int(len(rho))}


# ---- Stage 2: lower-burden rules, fit on training patients ------------------------------

def p_high(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Family-level propensity of the high arm: the training patients' arm frequency."""
    rates = train.groupby("family")["action_logged"].mean()
    return np.clip(target["family"].map(rates).fillna(float(train["action_logged"].mean())).to_numpy(float), 0.05, 0.95)


def e_logged(df: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    return np.where(df["action_logged"].to_numpy(int) == 1, p, 1.0 - p)


def ipw_value(actions: np.ndarray, df: pd.DataFrame, e: np.ndarray) -> float:
    return float(np.mean((actions == df["action_logged"].to_numpy(int)) * df["reward_observed"].to_numpy(float) / e))


def threshold_candidates(values: np.ndarray) -> np.ndarray:
    return np.unique(np.quantile(values, np.linspace(0.05, 0.95, THRESHOLD_GRID))).astype(float)


def better(value: float, frac: float, best: tuple[float, float]) -> bool:
    return value > best[0] + 1e-12 or (abs(value - best[0]) <= 1e-12 and frac < best[1])


def act(x: np.ndarray, direction: str, theta: float) -> np.ndarray:
    return (x < theta if direction == "lt" else x > theta).astype(int)


def fit_threshold(train: pd.DataFrame, e_train: np.ndarray, *, state: bool = True) -> tuple[dict, pd.DataFrame]:
    """Per-family cutpoint on the state by IPW value, ties toward lower burden. A rule
    (direction, theta) acts high where state < theta ("lt") or state > theta ("gt"); theta =
    -inf and +inf under "lt" are the fixed low and high arms. state=False keeps only the
    fixed arms, which gives the family-only audit rule."""
    rule, rows = {}, []
    for family, sub in train.groupby("family", sort=True):
        x, e = sub["state"].to_numpy(float), e_train[sub.index.to_numpy(int)]
        options = [("lt", -np.inf), ("lt", np.inf)]
        options += [(d, t) for d in ("lt", "gt") for t in threshold_candidates(x)] if state else []
        best = (-np.inf, 1.0, None)
        for d, t in options:
            actions = act(x, d, t)
            if np.isfinite(t) and min(actions.sum(), len(actions) - actions.sum()) < 2:
                continue
            v, frac = ipw_value(actions, sub, e), float(actions.mean())
            if better(v, frac, best[:2]):
                best = (v, frac, (d, t))
        rule[family] = best[2]
        rows.append({"family": family, "direction": best[2][0], "theta": best[2][1],
                     "train_value_ipw": best[0], "train_high_fraction": best[1]})
    return rule, pd.DataFrame(rows)


def apply_threshold(df: pd.DataFrame, rule: dict) -> np.ndarray:
    actions = np.zeros(len(df), dtype=int)
    family, x = df["family"].to_numpy(), df["state"].to_numpy(float)
    for fam, (d, t) in rule.items():
        actions[family == fam] = act(x[family == fam], d, t)
    return actions


def cql_actions(train: pd.DataFrame, test: pd.DataFrame, eta: float = CQL_ETA, *, state: bool = True) -> np.ndarray:
    """One-step linear CQL (Kumar et al. 2020) in the contextual-bandit limit: Q(x, a) =
    w_a . phi(x) fit by squared error plus eta [logsumexp_a Q(x, a) - Q(x, a_logged)], which
    pulls Q toward the logged actions. eta=0 is plain one-step Q-learning; state=False drops
    the state feature."""
    families = sorted(set(train["family"]).union(test["family"]))
    mu, sd = float(train["state"].mean()), float(train["state"].std()) + 1e-12

    def features(df: pd.DataFrame) -> np.ndarray:
        f = np.zeros((len(df), len(families) + 2))
        if state:
            f[:, 0] = (df["state"].to_numpy(float) - mu) / sd
        for j, fam in enumerate(families):
            f[df["family"].to_numpy() == fam, 1 + j] = 1.0
        f[:, -1] = 1.0
        return f

    X, X_test = features(train), features(test)
    D, a, r = X.shape[1], train["action_logged"].to_numpy(int), train["reward_observed"].to_numpy(float)

    def loss(w: np.ndarray) -> float:
        q0, q1 = X @ w[:D], X @ w[D:]
        q = np.where(a == 1, q1, q0)
        return float(np.mean((q - r) ** 2) + eta * np.mean(np.logaddexp(q0, q1) - q))

    w = minimize(loss, np.zeros(2 * D), method="L-BFGS-B").x
    return ((X_test @ w[D:]) > (X_test @ w[:D])).astype(int)


def t_learner_actions(train: pd.DataFrame, test: pd.DataFrame, seed: int = T_LEARNER_SEED) -> np.ndarray:
    """T-learner with random-forest base learners (200 trees): one model of the burden-adjusted
    reward per logged arm on state and family; act high where that arm's prediction is
    higher. eta=0 CQL is the same rule with linear learners."""
    cols, pred = ["state", "family"], []
    for a in (0, 1):
        sub = train.loc[train["action_logged"] == a]
        rf = Pipeline([("prep", ColumnTransformer([("num", "passthrough", ["state"]),
                                                   ("cat", OneHotEncoder(handle_unknown="ignore"), ["family"])])),
                       ("rf", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1))])
        pred.append(rf.fit(sub[cols], sub["reward_observed"].to_numpy(float)).predict(test[cols]))
    return (pred[1] > pred[0]).astype(int)


def dr_action_scores(train: pd.DataFrame) -> list[np.ndarray]:
    """Doubly robust scores of each action on the training rows, the policy-tree objective."""
    logged, r = train["action_logged"].to_numpy(int), train["reward_observed"].to_numpy(float)
    x = train[["state", "family"]]
    q = {a: np.asarray(ridge(["state"], ["family"]).fit(x[logged == a], r[logged == a]).predict(x)) for a in (0, 1)}
    q_logged = np.where(logged == 1, q[1], q[0])
    p = p_high(train, train)
    e = {0: 1.0 - p, 1: p}
    return [q[a] + (logged == a) / e[a] * (r - q_logged) for a in (0, 1)]


def leaf_action(idx: np.ndarray, scores: list[np.ndarray], logged: np.ndarray) -> tuple[int, float] | None:
    """Best action for a candidate leaf, or None if it lacks size or support for both actions."""
    if len(idx) < TREE_MIN_LEAF or logged[idx].min() == logged[idx].max():
        return None
    totals = [float(scores[a][idx].sum()) for a in (0, 1)]
    return (1 if totals[1] > totals[0] + 1e-12 else 0), max(totals)  # ties go to the low arm


def candidate_splits(df: pd.DataFrame) -> list[tuple[str, object, list[np.ndarray]]]:
    """Binary cutpoints on the state and a multiway split on family; a family root followed by
    per-family cuts reproduces the threshold rule, so the tree class nests it."""
    x = df["state"].to_numpy(float)
    splits = [("state", float(t), [x < t, x >= t]) for t in threshold_candidates(x)]
    fams = sorted(df["family"].unique())
    if len(fams) > 1:
        splits.append(("family", fams, [df["family"].to_numpy() == f for f in fams]))
    return splits


def fit_policy_tree(df: pd.DataFrame, scores: list[np.ndarray], logged: np.ndarray, depth: int):
    """Exact depth-limited policy tree on the doubly robust scores: the objective is additive
    over leaves, so optimizing each split's children independently gives the optimum.
    Returns (node, objective) or None, where node is ("leaf", a) or (feature, key, children)."""
    leaf = leaf_action(df.index.to_numpy(), scores, logged)
    best = (("leaf", leaf[0]), leaf[1]) if leaf else None
    if depth > 0:
        for feat, key, masks in candidate_splits(df):
            if any(not m.any() for m in masks):
                continue
            kids = [fit_policy_tree(df[m], scores, logged, depth - 1) for m in masks]
            if all(kids) and (best is None or sum(k[1] for k in kids) > best[1]):
                best = ((feat, key, [k[0] for k in kids]), sum(k[1] for k in kids))
    return best


def apply_policy_tree(test: pd.DataFrame, node) -> np.ndarray:
    family, x = test["family"].to_numpy(), test["state"].to_numpy(float)
    actions = np.zeros(len(test), dtype=int)

    def walk(node, sel: np.ndarray) -> None:
        if node[0] == "leaf":
            actions[sel] = node[1]
        elif node[0] == "state":
            m = x[sel] < node[1]
            walk(node[2][0], sel[m])
            walk(node[2][1], sel[~m])
        else:
            for f, kid in zip(node[1], node[2]):
                walk(kid, sel[family[sel] == f])

    walk(node, np.arange(len(test)))
    return actions


# ---- Doubly robust evaluation on the held-out patient -----------------------------------

def fold_base(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Held-out rows with the logged arm's propensity and, per arm, ridge predictions of the
    reward and the endpoint on state and family, all fit on the training patients."""
    base = test.assign(e_logged=e_logged(test, p_high(train, test)))
    for a, arm in ((0, "low"), (1, "high")):
        sub = train.loc[train["action_logged"] == a]
        pred = np.asarray(ridge(["state"], ["family"]).fit(sub[["state", "family"]], sub[["reward_observed", "endpoint"]])
                          .predict(base[["state", "family"]]))
        base[f"q_reward_{arm}"], base[f"q_endpoint_{arm}"] = pred[:, 0], pred[:, 1]
    return base


def evaluate_policy(base: pd.DataFrame, name: str, actions: np.ndarray) -> pd.DataFrame:
    """Per-row doubly robust values of the reward (value_dr) and the endpoint (endpoint_dr)."""
    logged = base["action_logged"].to_numpy(int)
    w = (actions == logged) / base["e_logged"].to_numpy(float)
    out = base.assign(policy=name, action_policy=actions, policy_high=actions.astype(float),
                      policy_cost=np.where(actions == 1, base["cost_high"], base["cost_low"]))
    for col, y, q in (("value_dr", "reward_observed", "q_reward"), ("endpoint_dr", "endpoint", "q_endpoint")):
        q_policy = np.where(actions == 1, base[f"{q}_high"], base[f"{q}_low"])
        q_logged = np.where(logged == 1, base[f"{q}_high"], base[f"{q}_low"])
        out[col] = q_policy + w * (base[y] - q_logged)
    return out


FIXED = {"always_low": lambda train, test: np.zeros(len(test), dtype=int),
         BASELINE: lambda train, test: np.ones(len(test), dtype=int)}


def evaluate_over_folds(df: pd.DataFrame, builders: dict) -> pd.DataFrame:
    """Row-level evaluation of each policy over every fold, stacked; builders map a policy name
    to fn(train, test) -> actions, fit per fold."""
    parts = []
    for _, train, test in iter_lopo(df):
        base = fold_base(train, test)
        parts += [evaluate_policy(base, name, build(train, test)) for name, build in builders.items()]
    return pd.concat(parts, ignore_index=True)


def candidates_over_folds(df: pd.DataFrame, lambda0: float, *, audit: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every rule on the training patients of each fold and evaluate it on the held-out
    patient. audit=False fits only the prespecified classes and the baseline."""
    df = df.assign(reward_observed=df["endpoint"] - float(lambda0) * df["cost"])
    rows, thresholds = [], []
    for held, train, test in iter_lopo(df):
        e_train = e_logged(train, p_high(train, train))
        rule, table = fit_threshold(train, e_train)
        thresholds.append(table.assign(fold_patient=held))
        actions = {"always_low": np.zeros(len(test), dtype=int), "threshold": apply_threshold(test, rule),
                   "one_step_cql": cql_actions(train, test)}
        if audit:
            scores, logged = dr_action_scores(train), train["action_logged"].to_numpy(int)
            trees = {d: fit_policy_tree(train, scores, logged, d) for d in (1, 2)}
            actions |= {
                "family_only": apply_threshold(test, fit_threshold(train, e_train, state=False)[0]),
                "cql_no_state": cql_actions(train, test, state=False),
                **{name: cql_actions(train, test, eta=eta) for name, eta in CQL_AUDIT_ETAS.items()},
                "t_learner_rf": t_learner_actions(train, test),
                **{f"policy_tree_depth{d}": apply_policy_tree(test, trees[d][0]) if trees[d]
                   else np.ones(len(test), dtype=int) for d in (1, 2)},
            }
        actions[BASELINE] = np.ones(len(test), dtype=int)
        base = fold_base(train, test)
        rows += [evaluate_policy(base, name, a) for name, a in actions.items()]
    return pd.concat(rows, ignore_index=True), pd.concat(thresholds, ignore_index=True)


def summarize(rowwise: pd.DataFrame) -> pd.DataFrame:
    """Patient-weighted policy values: means over each patient's held-out rows, then over patients."""
    per_patient = rowwise.groupby(["policy", "patient"]).agg(
        value_dr=("value_dr", "mean"), endpoint=("endpoint_dr", "mean"), cost=("policy_cost", "mean"),
        high_fraction=("policy_high", "mean"))
    out = per_patient.groupby("policy").mean().add_suffix("_patient")
    return out.loc[[p for p in POLICIES if p in out.index]].reset_index()


def retain(summary: pd.DataFrame) -> str:
    """Stage 2: the prespecified class with the highest patient-weighted value of the reward."""
    return str(summary.set_index("policy").loc[list(CANDIDATES), "value_dr_patient"].idxmax())


def reference_lambda0(df: pd.DataFrame) -> float:
    """Empirical break-even rate |dV_Y| / |dV_C| between full de-escalation and the baseline,
    from patient-weighted DR values over the folds. It uses only the two fixed arms, whose
    endpoint values do not depend on lambda."""
    s = summarize(evaluate_over_folds(df.assign(reward_observed=df["endpoint"]), FIXED)).set_index("policy")
    d = s.loc["always_low"] - s.loc[BASELINE]
    return float(abs(d["endpoint_patient"]) / abs(d["cost_patient"]))


def lambda0_dropone(df: pd.DataFrame, lambda0: float) -> pd.Series:
    """lambda0 recomputed without each patient, as a ratio to the full-data value."""
    return pd.Series({p: reference_lambda0(df[df["patient"] != p].reset_index(drop=True)) / lambda0
                      for p in sorted(df["patient"].unique())})


# ---- Stage 3: certification bounds ------------------------------------------------------

def split_by_cluster(cluster_ids: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(cluster_ids, kind="stable")
    sorted_ids = np.asarray(cluster_ids)[order]
    return [np.asarray(g, dtype=int) for g in np.split(order, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1)]


def wide(rowwise: pd.DataFrame, col: str) -> pd.DataFrame:
    return rowwise.pivot(index=["episode_uid", "patient"], columns="policy", values=col)


def paired_contrast(rowwise: pd.DataFrame, policy: str, ref: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Finite per-row contrasts policy - ref and the patient of each row."""
    w = wide(rowwise, col)
    delta = (w[policy] - w[ref]).to_numpy(float)
    keep = np.isfinite(delta)
    return delta[keep], w.index.get_level_values("patient").to_numpy()[keep]


def cluster_bootstrap(groups: list[list[np.ndarray]], n_boot: int, rng) -> np.ndarray:
    """Paired two-stage cluster bootstrap of patient-weighted means: patients with replacement,
    then wells within each sampled patient. groups[p] holds one array per variable over
    patient p's wells, so every variable is resampled on the same wells. Returns an
    (n_boot, n_variables) array."""
    boot = np.empty((int(n_boot), len(groups[0])), dtype=float)
    for b in range(int(n_boot)):
        patient_means = []
        for sid in rng.integers(0, len(groups), size=len(groups)):
            g = groups[int(sid)]
            idx = rng.integers(0, len(g[0]), size=len(g[0]))
            patient_means.append([float(np.mean(v[idx])) for v in g])
        boot[b] = np.mean(patient_means, axis=0)
    return boot


def lcb_paired(delta, pid, n_boot, rng, q):
    groups = [[delta[i]] for i in split_by_cluster(pid)]
    boot = cluster_bootstrap(groups, n_boot, rng)[:, 0]
    return float(np.mean([g[0].mean() for g in groups])), float(np.quantile(boot, q))


def lcb_wild(delta, pid, n_boot, rng, q):
    """Wild-cluster screen: Rademacher signs on the centered patient-level contrasts, all 2^G
    sign patterns when G <= 16, so attainable levels are multiples of 2^-G."""
    means = np.array([delta[i].mean() for i in split_by_cluster(pid)])
    point, G = float(means.mean()), len(means)
    centered = means - point
    if G <= 16:
        signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * G))).T.reshape(-1, G)
        return point, float(np.quantile(point + signs @ centered / G, q, method="inverted_cdf"))
    boot = [point + np.mean(rng.choice([-1.0, 1.0], size=G) * centered) for _ in range(int(n_boot))]
    return point, float(np.quantile(boot, q))


BOUNDS = {"paired": lcb_paired, "wild": lcb_wild}


def lcb(rowwise: pd.DataFrame, policy: str, ref: str, col: str, n_boot: int, seed: int, bound: str,
        q: float) -> tuple[float, float]:
    """Point contrast policy - ref and its one-sided lower confidence bound at level q."""
    delta, pid = paired_contrast(rowwise, policy, ref, col)
    return BOUNDS[bound](delta, pid, n_boot, np.random.default_rng(seed), q)


def certify(rowwise: pd.DataFrame, summary: pd.DataFrame, lambda0: float, n_boot: int, seed: int, *,
            readings=READINGS, policies=POLICIES) -> pd.DataFrame:
    """One row per lower-burden policy and reading: burden saving B, endpoint contrast against
    the baseline, its lower bound, the margin LCB + lambda0 B, and the critical rate
    lambda* = -LCB / B, also as a ratio to lambda0."""
    cost = summary.set_index("policy")["cost_patient"]
    rows = []
    for policy in (p for p in policies if p != BASELINE):
        saving = cost[BASELINE] - float(cost[policy])
        for reading, bound, q in readings:
            delta, low = lcb(rowwise, policy, BASELINE, "endpoint_dr", n_boot, seed + BOOT_OFFSET[policy], bound, q)
            crit = max(0.0, -low) / saving if saving > 0 else float("nan")
            rows.append({"policy": policy, "reading": reading, "burden_saving": saving, "endpoint_delta": delta,
                         "lcb": low, "margin": low + lambda0 * saving, "critical_lambda": crit,
                         "critical_lambda_ratio": crit / lambda0})
    return pd.DataFrame(rows)


def joint_intervals(rowwise: pd.DataFrame, lambda0: float, n_boot: int, seed: int) -> pd.DataFrame:
    """Endpoint contrast and burden saving resampled together on each policy's certificate
    stream (same seed, so delta_lo90 equals the certificate's bound). Reports 90% intervals for
    both and the critical rate with the saving resampled as well: the lowest lambda at which
    the ALPHA quantile of delta* + lambda B* reaches zero. Each resample's margin rises with
    lambda because B* >= 0, so the quantile does too and bisection applies."""
    endpoint, cost = wide(rowwise, "endpoint_dr"), wide(rowwise, "policy_cost")
    patients = endpoint.index.get_level_values("patient").to_numpy()
    rows = []
    for policy in (p for p in POLICIES if p != BASELINE):
        delta = (endpoint[policy] - endpoint[BASELINE]).to_numpy(float)
        saving = (cost[BASELINE] - cost[policy]).to_numpy(float)
        groups = [[delta[i], saving[i]] for i in split_by_cluster(patients)]
        boot = cluster_bootstrap(groups, n_boot, np.random.default_rng(seed + BOOT_OFFSET[policy]))
        q = lambda lam: float(np.quantile(boot[:, 0] + lam * boot[:, 1], ALPHA))
        lo, hi = 0.0, lambda0
        while q(hi) < 0 and hi < 1e3 * lambda0:
            hi *= 2
        if q(0.0) >= 0:
            crit = 0.0
        elif q(hi) < 0:
            crit = float("inf")
        else:
            for _ in range(60):
                lo, hi = (lo, (lo + hi) / 2) if q((lo + hi) / 2) >= 0 else ((lo + hi) / 2, hi)
            crit = hi
        rows.append({
            "policy": policy, "endpoint_delta": float(np.mean([g[0].mean() for g in groups])),
            "delta_lo90": float(np.quantile(boot[:, 0], ALPHA)), "delta_hi90": float(np.quantile(boot[:, 0], 1 - ALPHA)),
            "burden_saving": float(np.mean([g[1].mean() for g in groups])),
            "saving_lo90": float(np.quantile(boot[:, 1], ALPHA)), "saving_hi90": float(np.quantile(boot[:, 1], 1 - ALPHA)),
            "critical_lambda_joint": crit, "critical_lambda_ratio_joint": crit / lambda0,
        })
    return pd.DataFrame(rows)


def knob_sensitivity(df: pd.DataFrame, lambda0: float, n_boot: int, seed: int) -> pd.DataFrame:
    """Paired certificate of the knob-dependent classes across threshold grids and minimum
    tree leaf sizes (KNOB_SWEEP), with the class Stage 2 retains at each setting. Bootstrap
    streams match the main run, so differences come from the learned rules alone."""
    global THRESHOLD_GRID, TREE_MIN_LEAF
    rows, default = [], (THRESHOLD_GRID, TREE_MIN_LEAF)
    try:
        for grid in KNOB_SWEEP["threshold_grid"]:
            for leaf in KNOB_SWEEP["tree_min_leaf"]:
                THRESHOLD_GRID, TREE_MIN_LEAF = grid, leaf
                rowwise, _ = candidates_over_folds(df, lambda0)
                summary = summarize(rowwise)
                cert = certify(rowwise, summary, lambda0, n_boot, seed, readings=READINGS[:1],
                               policies=("threshold", "policy_tree_depth1", "policy_tree_depth2"))
                rows.append(cert[["policy", "burden_saving", "lcb", "critical_lambda_ratio"]].assign(
                    threshold_grid=grid, tree_min_leaf=leaf, retained=retain(summary)))
    finally:
        THRESHOLD_GRID, TREE_MIN_LEAF = default
    return pd.concat(rows, ignore_index=True)


def retention_by_lambda(df: pd.DataFrame, lambda0: float) -> pd.DataFrame:
    """Stage 2 rerun at multiples of lambda0, without bootstrap: the class retained at each rate."""
    parts = []
    for ratio in LAMBDA_RATIOS:
        summary = summarize(candidates_over_folds(df, lambda0 * ratio)[0])
        parts.append(summary.assign(lambda_ratio=ratio, retained=retain(summary)))
    return pd.concat(parts, ignore_index=True)


# ---- Tables and Fig 4 -------------------------------------------------------------------

def per_patient(rowwise: pd.DataFrame) -> pd.DataFrame:
    """Patient-level view of every policy against the baseline: the endpoint contrast the
    clustered bounds resample, the burden saving, the share kept on the high arm, and the
    mean state."""
    endpoint, cost, high = (wide(rowwise, c) for c in ("endpoint_dr", "policy_cost", "policy_high"))
    state = rowwise.drop_duplicates("episode_uid").set_index(["episode_uid", "patient"])["state"]
    parts = []
    for p in (p for p in POLICIES if p != BASELINE):
        d = pd.DataFrame({"endpoint_delta": endpoint[p] - endpoint[BASELINE], "burden_saving": cost[BASELINE] - cost[p],
                          "high_fraction": high[p], "state_mean": state}).groupby(level="patient")
        parts.append(d.mean().assign(n_wells=d.size(), policy=p).reset_index())
    return pd.concat(parts, ignore_index=True)


def composition(df: pd.DataFrame) -> pd.DataFrame:
    """Wells (or organoids) per patient, family, and arm."""
    return df.groupby(["patient", "family", "arm"]).size().rename("wells").reset_index()


def stress_inputs(rowwise: pd.DataFrame, summary: pd.DataFrame, retained: str, lambda0: float) -> dict[str, float]:
    """Calibration targets for stress_test.py from the retained rule's per-row endpoint
    contrast: patient count, mean rows per patient, between-patient SD of patient means,
    pooled within-patient SD, and the point margin-adjusted effect at lambda0."""
    delta, pid = paired_contrast(rowwise, retained, BASELINE, "endpoint_dr")
    g = pd.Series(delta).groupby(pid)
    cost = summary.set_index("policy")["cost_patient"]
    return {"G": int(g.ngroups), "n_per": float(g.size().mean()),
            "sigma_b": float(g.mean().std(ddof=1)), "sigma_w": float(np.sqrt(g.var(ddof=1).mean())),
            "effect": float(g.mean().mean() + lambda0 * (cost[BASELINE] - cost[retained]))}


def audit_table(cert: pd.DataFrame, summary: pd.DataFrame, rowwise: pd.DataFrame, retained: str) -> pd.DataFrame:
    """One row per lower-burden policy: role, burden saving, endpoint contrast, and per reading
    the lower bound, the margin at lambda0, and lambda*/lambda0; plus the patient-weighted share
    kept on the high arm and the action agreement with the eta=1 CQL rule."""
    by_reading = cert.pivot(index="policy", columns="reading", values=["lcb", "margin", "critical_lambda_ratio"])
    by_reading.columns = [f"{v}_{r}" for v, r in by_reading.columns]
    out = cert.drop_duplicates("policy").set_index("policy")[["burden_saving", "endpoint_delta"]].join(by_reading)
    actions = rowwise.pivot(index="episode_uid", columns="policy", values="action_policy")
    out["high_fraction"] = summary.set_index("policy")["high_fraction_patient"]
    out["agreement_with_cql_eta1"] = actions.eq(actions["one_step_cql"], axis=0).mean()
    out.insert(0, "role", ["retained" if p == retained else "prespecified" if p in CANDIDATES else "audit"
                           for p in out.index])
    return out.loc[[p for p in POLICIES if p in out.index]].reset_index()


def certification_figure(results: dict[str, tuple[pd.DataFrame, str, float]]) -> None:
    """Fig 4: certification margin M(lambda) = LCB + lambda B on the lambda/lambda0 axis for each
    retained rule. The two critical ratios split the axis into bands where neither reading
    certifies, where only the wild-cluster screen does, and where the paired certificate holds
    as well; read against the band Lambda."""
    lo, hi = REFERENCE_REGION
    names = {"always_low": "fixed low-burden arm", "threshold": "early-death rule", "one_step_cql": "conservative Q-learning"}
    fig, axes = plt.subplots(len(results), 1, figsize=(3.5, 3.9))
    for a, (task, (cert, retained, lam)) in zip(np.atleast_1d(axes), results.items()):
        rob = cert.set_index(["policy", "reading"]).loc[retained]
        B = float(rob.loc["certificate", "burden_saving"])
        wild_r, paired_r = (float(rob.loc[k, "critical_lambda_ratio"]) for k in ("screen", "certificate"))
        xmax = max(hi * 1.15, paired_r * 1.35)
        grid = np.linspace(0, xmax, 100)
        a.axvspan(0, wild_r, color="#eeeeee", alpha=0.7, label="below both")
        a.axvspan(wild_r, paired_r, color="#fff3cd", alpha=0.8, label="wild only")
        a.axvspan(paired_r, xmax, color="#e8f5e9", alpha=0.8, label="wild + paired")
        a.plot(grid, float(rob.loc["screen", "lcb"]) + grid * lam * B, "-", color="C0", lw=2)
        a.plot(grid, float(rob.loc["certificate", "lcb"]) + grid * lam * B, "--", color="C1", lw=2)
        a.axhline(0, color="k", lw=0.8)
        for x in (lo, hi):
            a.axvline(x, color="0.4", ls=(0, (5, 4)), lw=1.0)
        a.axvline(1.0, color="0.5", ls=":", lw=1.1)
        trans = a.get_xaxis_transform()
        a.text((lo + hi) / 2, 0.94, "$\\Lambda$", transform=trans, ha="center", va="top", fontsize=8, color="0.3")
        a.text(1.0, 0.05, "$\\lambda_0$", transform=trans, ha="center", va="bottom", fontsize=6, color="0.45")
        a.annotate(f"{wild_r:.2f}", (wild_r, 0), textcoords="offset points", xytext=(4, 7), color="C0", fontsize=7)
        a.annotate(f"{paired_r:.2f}", (paired_r, 0), textcoords="offset points", xytext=(4, -12), color="C1", fontsize=7)
        a.set_title(f"{task}: {names[retained]}", fontsize=8)
        a.set_xlim(0, xmax)
        a.set_xlabel("$\\lambda/\\lambda_0$")
        a.set_ylabel("Certification margin")
        a.tick_params(labelsize=7)
        a.legend(fontsize=5, loc="lower right", labelspacing=0.3, handlelength=1.2, handletextpad=0.4, borderpad=0.3)
    fig.tight_layout()
    save(fig, "fig4_certification_curve", bbox_inches="tight")


def run_slice(label: str, df: pd.DataFrame, n_boot: int, seed: int) -> tuple[dict, dict[str, pd.DataFrame], pd.DataFrame]:
    lambda0 = reference_lambda0(df)
    rowwise, thresholds = candidates_over_folds(df, lambda0)
    summary = summarize(rowwise)
    retained = retain(summary)
    cert = certify(rowwise, summary, lambda0, n_boot, seed)
    dropone = lambda0_dropone(df, lambda0)
    tables = {"stage1": stage1(df), "certification": cert, "per_patient": per_patient(rowwise),
              "audit": audit_table(cert, summary, rowwise, retained), "composition": composition(df),
              "thresholds": thresholds, "retention_by_lambda": retention_by_lambda(df, lambda0),
              "joint_intervals": joint_intervals(rowwise, lambda0, n_boot, seed),
              "knob_sensitivity": knob_sensitivity(df, lambda0, n_boot, seed)}
    margins = cert.set_index(["policy", "reading"]).loc[retained, "margin"]
    row = {"dataset": label, "lambda0": lambda0, "lambda0_dropone_min": dropone.min(),
           "lambda0_dropone_max": dropone.max(), "lambda0_dropone_argmin": dropone.idxmin(),
           **within_cell_order(df), "retained_policy": retained,
           **{f"retained_margin_{k}": float(margins[k]) for k, _, _ in READINGS},
           **{f"stress_{k}": v for k, v in stress_inputs(rowwise, summary, retained, lambda0).items()}}
    return row, tables, cert


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=OUT / "processed")
    ap.add_argument("--outdir", type=Path, default=OUT / "results")
    # 50,000 resamples keep the Monte Carlo spread of the paired bound near 0.0003; at 1,200
    # it was 0.002, enough to flip a certificate whose margin sits near zero.
    ap.add_argument("--n_boot", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows, tables, figure = [], {}, {}
    for label in ("Schuster", "Spiller", "Wells"):
        row, slice_tables, cert = run_slice(label, load_checkpoint(args.data / f"{label.lower()}.csv"),
                                            args.n_boot, args.seed)
        rows.append(row)
        for name, t in slice_tables.items():
            t.insert(0, "dataset", label)
            tables.setdefault(name, []).append(t)
        figure[label] = (cert, row["retained_policy"], row["lambda0"])
    pd.DataFrame(rows).to_csv(args.outdir / "summary.csv", index=False)
    for name, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(args.outdir / f"{name}.csv", index=False)
    # Fig 4 draws the two chemotherapy rules; the radiation certificate is reported in the text.
    certification_figure({k: figure[k] for k in ("Schuster", "Spiller")})


if __name__ == "__main__":
    main()
