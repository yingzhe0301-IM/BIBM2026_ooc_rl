"""Data-calibrated simulations of the certificate's operating characteristics.

Study A simulates the margin-adjusted contrast directly and applies the wild screen and
the paired certificate: calibration against the true margin, size under rising
between-patient heterogeneity, and power by patient count. Study B generates wells and
reruns the doubly robust estimator with both bounds. The selection study reruns Stage 2
retention and certification on null data. Calibration targets come from
out/results/summary.csv, so protocol.py runs first. Writes the stress_*.csv tables and Fig 5.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import protocol as P
from prep import OUT, save

ALPHA = P.ALPHA
DELTAS = [round(-0.06 + 0.01 * i, 2) for i in range(19)]  # -0.06 to 0.12 in steps of 0.01
ICCS = [None, 0.05, 0.15, 0.30]
POWER_GS = list(range(3, 21))
# Well-level studies use the observed replicates per patient x family x arm cell: wells for
# the chemotherapy slices (Schuster 2-4, Spiller 3) and tracked organoids for Wells (33-111,
# mean 71).
FAMILIES = {"Schuster": ["F1", "F2", "F3", "F4"], "Spiller": ["IRI", "OXA"], "Wells": ["IR"]}
N_REP = {"Schuster": 3, "Spiller": 3, "Wells": 71}
CALIB: dict[str, dict] = {}  # filled in main() from summary.csv


def seed_of(*parts) -> int:
    """Deterministic, process-independent seed (Python's hash is salted per process)."""
    s = 2166136261
    for p in parts:
        for ch in str(p):
            s = ((s ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return s % 100000


def load_calib(results: Path) -> dict[str, dict]:
    """Per slice: patient count, rows per patient (truncated mean), between- and within-patient
    SD of the retained rule's contrast, and the point margin-adjusted effect at lambda0, the
    last three rounded to three decimals."""
    s = pd.read_csv(results / "summary.csv").set_index("dataset")
    return {name: {"G": int(r["stress_G"]), "n_per": int(r["stress_n_per"]),
                   **{k: round(float(r[f"stress_{k}"]), 3) for k in ("sigma_b", "sigma_w", "effect")}}
            for name, r in s.iterrows()}


def _map(pool, fn, tasks):
    return map(fn, tasks) if pool is None else pool.map(fn, tasks)


def add_mc_se(df: pd.DataFrame, cols: list[str], reps: int) -> None:
    """Binomial Monte Carlo standard errors of the reported rates."""
    for col in cols:
        df[f"{col}_mc_se"] = np.sqrt(df[col] * (1.0 - df[col]) / reps)


# ---- Study A: the bounds on simulated contrasts -----------------------------------------

def synth_rowwise(rng: np.random.Generator, G: int, n_per: int, delta: float, sigma_b: float,
                  sigma_w: float) -> pd.DataFrame:
    """Two-policy frame whose per-row contrast has true mean delta, between-patient SD
    sigma_b, and within-patient SD sigma_w."""
    rows = []
    for g in range(G):
        u = rng.normal(0.0, sigma_b)
        z = delta + u + rng.normal(0.0, sigma_w, size=n_per)
        for i in range(n_per):
            rows += [(f"p{g}_w{i}", str(g), "cand", float(z[i])), (f"p{g}_w{i}", str(g), "baseline", 0.0)]
    return pd.DataFrame(rows, columns=["episode_uid", "patient", "policy", "value"])


def _cert_chunk(task):
    """Worker: wild and paired certification counts over one chunk of seeds."""
    i, c, n_boot, seeds = task
    hits = np.zeros(2)
    for s in seeds:
        rw = synth_rowwise(np.random.default_rng(s), c["G"], c["n_per"], c["delta"], c["sigma_b"], c["sigma_w"])
        hits += [P.lcb(rw, "cand", "baseline", "value", n_boot, s + 7, b, ALPHA)[1] >= -1e-12 for b in ("wild", "paired")]
    return i, hits


def cert_spec(row: dict, c: dict, delta: float, *seed_parts, G=None, sigma_b=None, sigma_w=None) -> tuple[dict, dict]:
    """One known-truth cell: its table row and its simulation parameters."""
    return row, {"G": c["G"] if G is None else G, "n_per": c["n_per"], "delta": delta,
                 "sigma_b": c["sigma_b"] if sigma_b is None else sigma_b,
                 "sigma_w": c["sigma_w"] if sigma_w is None else sigma_w, "base_seed": seed_of(*seed_parts)}


def cert_table(specs: list[tuple[dict, dict]], reps: int, n_boot: int, pool, chunk: int = 50) -> pd.DataFrame:
    """Certification rates of the wild screen and the paired certificate per cell. Every
    replicate has its own seed, so splitting cells into chunks across processes leaves the
    results unchanged."""
    tasks = []
    for i, (_, c) in enumerate(specs):
        seeds = [c["base_seed"] + 1009 * r for r in range(reps)]
        tasks += [(i, c, n_boot, seeds[k:k + chunk]) for k in range(0, reps, chunk)]
    hits = np.zeros((len(specs), 2))
    for i, h in _map(pool, _cert_chunk, tasks):
        hits[i] += h
    out = pd.DataFrame([{**row, "cert_wild": w / reps, "cert_paired": p / reps} for (row, _), (w, p) in zip(specs, hits)])
    add_mc_se(out, ["cert_wild", "cert_paired"], reps)
    return out


def calibration_table(reps: int, n_boot: int, pool) -> pd.DataFrame:
    """Certification rate against the true margin Delta: Delta < 0 is a truly inferior rule
    (certifying it is a false certification), Delta = 0 the non-inferiority boundary, and
    Delta > 0 a non-inferior rule (certifying it is power)."""
    regime = lambda d: "inferior" if d < -1e-9 else ("boundary" if abs(d) < 1e-9 else "non-inferior")
    return cert_table([cert_spec({"dataset": name, "G": c["G"], "delta": d, "regime": regime(d)}, c, d, name, d)
                       for name, c in CALIB.items() for d in DELTAS], reps, n_boot, pool)


def icc_robustness(reps: int, n_boot: int, pool) -> pd.DataFrame:
    """Boundary size (Delta = 0) as the between-patient share of variance rises."""
    specs = []
    for name, c in CALIB.items():
        total = c["sigma_b"] ** 2 + c["sigma_w"] ** 2
        for icc in ICCS:
            tag = "data" if icc is None else f"{icc:.2f}"
            sb = c["sigma_b"] if icc is None else float(np.sqrt(icc * total))
            sw = c["sigma_w"] if icc is None else float(np.sqrt((1 - icc) * total))
            specs.append(cert_spec({"dataset": name, "G": c["G"], "icc": tag, "sigma_b": sb, "sigma_w": sw},
                                   c, 0.0, name, "icc", tag, sigma_b=sb, sigma_w=sw))
    return cert_table(specs, reps, n_boot, pool)


def power_curve(reps: int, n_boot: int, pool) -> pd.DataFrame:
    """Power to certify at each slice's observed margin-adjusted effect against patient count."""
    return cert_table([cert_spec({"dataset": name, "G": G, "delta": c["effect"]}, c, c["effect"], name, "pow", G, G=G)
                       for name, c in CALIB.items() for G in POWER_GS], reps, n_boot, pool)


# ---- Study B and the selection study: the full estimator on generated wells -------------

def synth_wells(rng: np.random.Generator, G: int, families: list[str], n_rep: int, delta_true: float,
                sigma_b: float, sigma_w: float) -> pd.DataFrame:
    """Generated wells in the load_checkpoint schema with a fixed low-against-high contrast of
    known patient-weighted size delta_true = E[Y(low)] - E[Y(high)]; both arms are observed in
    every patient and family."""
    rows = []
    for g in range(G):
        u = rng.normal(0.0, sigma_b)  # patient endpoint shift, shared across arms
        for k, fam in enumerate(families):
            fam_base = 0.35 + 0.05 * (k % 5)
            for arm in ("low", "high"):
                eff = delta_true if arm == "low" else 0.0
                for r in range(n_rep):
                    state = np.clip(0.15 + 0.4 * (fam_base - 0.35) + rng.normal(0, 0.08), 0, 1)
                    endpoint = float(np.clip(fam_base + u + eff + rng.normal(0, sigma_w), 0, 1))
                    rows.append({"patient": str(g), "family": fam, "arm": arm, "state": state, "endpoint": endpoint,
                                 "action_logged": int(arm == "high"), "cost": 72.0 if arm == "high" else 4.0,
                                 "episode_uid": f"{g}|{fam}|{arm}|{r}"})
    return pd.DataFrame(rows).assign(cost_low=4.0, cost_high=72.0)


def _cov_chunk(task):
    """Worker: rerun the DR estimator and both bounds on one chunk of replicates. Tallies the
    point estimates, coverage of the true contrast (wild, paired), false certifications
    (wild, paired), and replicates."""
    name, G, families, n_rep, delta_true, sb, sw, n_boot, reps = task
    tally = np.zeros(6)
    for r in reps:
        df = synth_wells(np.random.default_rng(seed_of(name, "cov", r)), G, families, n_rep, delta_true, sb, sw)
        rw = P.evaluate_over_folds(df.assign(reward_observed=df["endpoint"]), P.FIXED)
        (pt, wild), (_, paired) = (P.lcb(rw, "always_low", P.BASELINE, "endpoint_dr", n_boot, r + 11, b, ALPHA)
                                   for b in ("wild", "paired"))
        tally += [pt, wild <= delta_true + 1e-12, paired <= delta_true + 1e-12, wild >= -1e-12, paired >= -1e-12, 1]
    return name, tally


def coverage_study(reps: int, n_boot: int, pool, chunk: int = 40) -> pd.DataFrame:
    """For the fixed low arm with a known, slightly inferior contrast, the DR point bias and the
    one-sided coverage P(LCB <= truth) of both bounds at each slice's geometry."""
    delta_true = -0.02
    tasks = [(name, c["G"], FAMILIES[name], N_REP[name], delta_true, c["sigma_b"], c["sigma_w"], n_boot,
              range(k, min(k + chunk, reps))) for name, c in CALIB.items() for k in range(0, reps, chunk)]
    totals = {name: np.zeros(6) for name in CALIB}
    for name, tally in _map(pool, _cov_chunk, tasks):
        totals[name] += tally
    out = pd.DataFrame([{"dataset": name, "G": CALIB[name]["G"], "delta_true": delta_true, "dr_point_mean": t[0] / t[5],
                         "dr_point_bias": t[0] / t[5] - delta_true, "lcb_coverage_wild": t[1] / t[5],
                         "lcb_coverage_paired": t[2] / t[5], "false_cert_wild": t[3] / t[5],
                         "false_cert_paired": t[4] / t[5]} for name, t in totals.items()])
    add_mc_se(out, ["lcb_coverage_wild", "lcb_coverage_paired", "false_cert_wild", "false_cert_paired"], reps)
    return out


# Under a uniform-effect null with an uninformative state, no state-dependent rule beats the
# fixed arms, and at the data-driven break-even lambda0 every lower-burden rule sits at its
# own non-inferiority boundary. Rerunning discovery, retention, and certification measures
# how much choosing among the prespecified classes inflates false certification.

def _selection_trial(task):
    name, G, families, n_rep, delta_arm, sb, sw, n_boot, seed = task
    df = synth_wells(np.random.default_rng(seed), G, families, n_rep, delta_arm, sb, sw)
    lam0 = P.reference_lambda0(df)
    rowwise, _ = P.candidates_over_folds(df, lam0, audit=False)
    cert = P.certify(rowwise, P.summarize(rowwise), lam0, n_boot, seed + 7, policies=P.CANDIDATES)
    flags = (cert.set_index(["reading", "policy"])["margin"] >= -1e-12).unstack()[list(P.CANDIDATES)]
    flags.insert(0, "any", flags.any(axis=1))
    return name, flags.astype(int)


def selection_study(reps: int, n_boot: int, pool, delta_arm: float = -0.04) -> pd.DataFrame:
    """How often the full pipeline certifies some prespecified rule when none is non-inferior."""
    tasks = [(name, c["G"], FAMILIES[name], N_REP[name], delta_arm, c["sigma_b"], c["sigma_w"], n_boot,
              seed_of(name, "sel", r)) for name, c in CALIB.items() for r in range(reps)]
    totals = {}
    for name, flags in _map(pool, _selection_trial, tasks):
        totals[name] = totals[name] + flags if name in totals else flags
    parts = []
    for name, t in totals.items():
        rates = (t / reps).add_prefix("cert_").reset_index()
        rates.insert(0, "dataset", name)
        parts.append(rates)
    out = pd.concat(parts, ignore_index=True)
    add_mc_se(out, [c for c in out.columns if c.startswith("cert_")], reps)
    return out


# ---- Fig 5 ------------------------------------------------------------------------------

def stress_figure(results: Path) -> None:
    """Calibration and power small multiples: wild screen blue with circles, paired
    certificate orange with diamonds. Calibration panels split at the boundary into frosted
    inferior and non-inferior sides; power panels carry a left-to-right credibility gradient
    and the 0.8 reference line."""
    INF, NONINF = "#ecd2d6", "#d0dbe8"
    cred_cmap = LinearSegmentedColormap.from_list("credibility", ["#eef0f3", "#6b7d92"])
    calib = pd.read_csv(results / "stress_calibration.csv")
    power = pd.read_csv(results / "stress_power.csv")
    WILD = {"col": "cert_wild", "se": "cert_wild_mc_se", "color": "#1f77b4", "ls": "-", "marker": "o"}
    PAIR = {"col": "cert_paired", "se": "cert_paired_mc_se", "color": "#ff7f0e", "ls": (0, (5, 2)), "marker": "D"}
    style = {
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.4,
        "axes.linewidth": 0.8, "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    }
    rows = [("Schuster", r"Schuster · $G{=}3$"), ("Spiller", r"Spiller · $G{=}4$")]
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.95), sharex="col", sharey=True, layout="constrained")
        for i, (name, tag) in enumerate(rows):
            c = calib.loc[calib["dataset"] == name].sort_values("delta")
            p = power.loc[power["dataset"] == name].sort_values("G")
            top, bot = i == 0, i == len(rows) - 1

            ax = axes[i, 0]  # certification rate against the true margin Delta
            xlo, xhi = float(c["delta"].min()) - 0.008, float(c["delta"].max()) + 0.008
            ax.axvspan(xlo, 0.0, color=INF, alpha=0.7, lw=0, zorder=0)
            ax.axvspan(0.0, xhi, color=NONINF, alpha=0.7, lw=0, zorder=0)
            ax.axvline(0.0, color="0.45", lw=0.8, zorder=1)
            for s in (WILD, PAIR):
                se = c[s["se"]]
                ax.fill_between(c["delta"], c[s["col"]] - 1.96 * se, c[s["col"]] + 1.96 * se,
                                color=s["color"], alpha=0.20, lw=0, zorder=2)
                ax.plot(c["delta"], c[s["col"]], ls=s["ls"], color=s["color"], lw=1.8,
                        marker=s["marker"], ms=2.6, mec="white", mew=0.4, zorder=3)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(xlo, xhi)
            ax.set_xticks([-0.05, 0.0, 0.05, 0.10])
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_ylabel("Certification rate")
            ax.annotate(tag, xy=(0.045, 0.95), xycoords="axes fraction", fontsize=7.4,
                        fontweight="bold", va="top", ha="left")
            if top:
                ax.set_title("Calibration")
            if bot:
                ax.set_xlabel(r"$\Delta$")

            ax = axes[i, 1]  # power at the observed effect against patient count G
            px0, px1 = 2.4, 20.6
            ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), extent=[px0, px1, -0.02, 1.02],
                      aspect="auto", origin="lower", cmap=cred_cmap, alpha=0.6,
                      interpolation="bilinear", zorder=0)
            ax.axhline(0.8, color="0.5", lw=1.1, ls=":", zorder=1)
            for s in (WILD, PAIR):
                se = p[s["se"]]
                ax.fill_between(p["G"], p[s["col"]] - 1.96 * se, p[s["col"]] + 1.96 * se,
                                color=s["color"], alpha=0.20, lw=0, zorder=2)
                ax.plot(p["G"], p[s["col"]], ls=s["ls"], color=s["color"], lw=1.8,
                        marker=s["marker"], ms=2.6, mec="white", mew=0.4, zorder=3)
            ax.set_xlim(px0, px1)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks([4, 8, 12, 16, 20])
            if top:
                ax.set_title("Power")
            if bot:
                ax.set_xlabel(r"$G$")

        handles = [
            Line2D([], [], color=WILD["color"], marker="o", ls="-", lw=1.6, ms=4,
                   mec="white", mew=0.4, label="wild-cluster screen"),
            Line2D([], [], color=PAIR["color"], marker="D", ls=(0, (5, 2)), lw=1.6, ms=4,
                   mec="white", mew=0.4, label="paired two-stage"),
            Patch(facecolor=INF, alpha=0.7, label="inferior"),
            Patch(facecolor=NONINF, alpha=0.7, label="non-inferior"),
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False,
                   fontsize=6.8, handlelength=1.8, columnspacing=1.8, handletextpad=0.6)
        save(fig, "fig5_stress", bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=OUT / "results")
    ap.add_argument("--reps", type=int, default=3000, help="trials per Study A cell")
    ap.add_argument("--cov_reps", type=int, default=3000, help="trials per slice in Study B and the selection study")
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--figure", action="store_true", help="only redraw Fig 5 from the saved tables")
    args = ap.parse_args()
    if not args.figure:
        CALIB.update(load_calib(args.results))
        pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
        studies = {"calibration": lambda: calibration_table(args.reps, args.n_boot, pool),
                   "icc": lambda: icc_robustness(args.reps, args.n_boot, pool),
                   "power": lambda: power_curve(args.reps, args.n_boot, pool),
                   "coverage": lambda: coverage_study(args.cov_reps, args.n_boot, pool),
                   "selection": lambda: selection_study(args.cov_reps, args.n_boot, pool)}
        try:
            for name, run in studies.items():
                run().to_csv(args.results / f"stress_{name}.csv", index=False)
                print(f"wrote stress_{name}.csv")
        finally:
            if pool is not None:
                pool.shutdown()
    stress_figure(args.results)


if __name__ == "__main__":
    main()
