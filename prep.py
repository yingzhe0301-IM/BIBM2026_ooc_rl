"""Download the three public cohorts, build one table per benchmark slice, and draw Figs 2-3.

Each slice becomes one row per well, or per tracked organoid for Wells, with the logged
arm (low or high burden), the burden of that arm, the state read before the arms
diverge, a second state channel where one exists, and the final endpoint:
  Schuster  propidium-iodide death at 4 h (state), Caspase 3/7 at 4 h (second channel),
            death at 72 h (endpoint); burden is the exposure in hours, 4 or 72
  Spiller   dead-organoid fraction at day 0 (state) and day 7 (endpoint); burden is the
            concentration times 168 h
  Wells     caspase reporter before the first fraction (state), organoid area in the same
            image (second channel), reporter at day 14 (endpoint); burden is the total Gy
Schuster and Wells readings are log-ratios to the same patient's control at the same time.
"""
import argparse
import re
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd
from matplotlib.lines import Line2D
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
FIGURES = ROOT / "paper" / "overleaf_ready" / "figures"
DOCS = ROOT / "docs"  # PNG copies of the figures for the README
RAW = {
    "schuster": ("https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-19058-4/MediaObjects/"
                 "41467_2020_19058_MOESM13_ESM.xlsx", "41467_2020_19058_MOESM13_ESM.xlsx"),
    "spiller": ("https://raw.githubusercontent.com/eitm-org/organoid_drug_response/master/data/consolidated.csv",
                "spiller_consolidated.csv"),
    "wells": ("https://raw.githubusercontent.com/JanesLab/WellsCJ_OrganoSeg2/main/Figure%206/excelFilesZip.zip",
              "wells2026_excelFiles.zip"),
}

# Schuster 2020 source data, sheet "Figure 5a-b": one block per patient starting at these
# rows, with each marker's time series in the given column range.
SCHUSTER_BLOCKS = [2, 57, 108]
SCHUSTER_MARKERS = {"apoptosis_caspase": (4, 22), "death_PI": (23, 43)}
SCHUSTER_FAMILIES = ("FOLFIRINOX", "FOLFIRI", "FOLFOX", "Gemcitabine + 5FU")
SCHUSTER_ARMS = {f"{f} {s}": (f, arm) for f in SCHUSTER_FAMILIES for s, arm in (("4 hour", "low"), ("72 hour", "high"))}
# Spiller 2021: the dose pair of each drug and the lines screened at both doses.
SPILLER_DOSES = {"IRI": (20.0, 40.0), "OXA": (2.0, 20.0)}
SPILLER_LINES = {"IRI": ["12415", "12527", "12620", "12911"], "OXA": ["12415", "12527", "12620"]}
# Wells 2026: one NV488 caspase-reporter well per case and dose. The column codes map to 0,
# 1, and 6 Gy per day over five days (0, 5, and 30 Gy in total), as in the authors'
# figure6_fluorescenceAnalysisResutls.m. Images are P1 (and P2 for two cases) before the
# reporter, R1 after loading the reporter and before the first fraction, R2-R5 during
# radiation, and A1-A2 after; the authors drop the last two columns, and so do we.
WELLS_ROW = {"149": "C", "155": "D", "157": "B", "160": "D", "161": "B"}
WELLS_COLUMNS = {"149": ("02", "03", "04"), "155": ("02", "03", "04"), "157": ("04", "05", "06"),
                 "160": ("02", "03", "04"), "161": ("08", "09", "10")}
WELLS_GY = (0.0, 5.0, 30.0)

# Figs 2-3: legible fonts at IEEE column width, applied through rc_context.
FIG_STYLE = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
}
ARM_COLORS = {"low": "#4C72B0", "high": "#DD8452"}
PATIENT_COLORS = {"1": "#30638E", "2": "#D1495B", "3": "#66A182"}  # Schuster patients
FROSTED = {"frameon": True, "fancybox": True, "facecolor": "#fbfaf7", "edgecolor": "#d4d2cc", "framealpha": 0.6}


def save(fig: plt.Figure, name: str, **kw) -> None:
    """Write a figure to the paper as PDF and to docs/ as PNG for the README."""
    fig.savefig(FIGURES / f"{name}.pdf", **kw)
    DOCS.mkdir(exist_ok=True)
    fig.savefig(DOCS / f"{name}.png", dpi=300, **kw)
    plt.close(fig)


def download(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for url, name in RAW.values():
        path = raw_dir / name
        if not path.exists():
            with urlopen(Request(url, headers={"User-Agent": "ooc-reproducibility"})) as r, \
                    open(path.with_suffix(".part"), "wb") as f:
                shutil.copyfileobj(r, f)
            path.with_suffix(".part").replace(path)
    return {k: raw_dir / name for k, (_, name) in RAW.items()}


def read_schuster(workbook: Path) -> pd.DataFrame:
    """Per-well marker series as log-ratios to the patient's negative-control mean."""
    wb = load_workbook(workbook, data_only=True)
    ws, records = wb["Figure 5a-b"], []
    for patient, start in enumerate(SCHUSTER_BLOCKS, start=1):
        end = (SCHUSTER_BLOCKS + [ws.max_row + 1])[patient] - 1
        regimen, replicate = None, 0
        for row in range(start + 2, end + 1):
            cell = ws.cell(row, 3).value
            if cell is not None:
                regimen, replicate = str(cell).strip(), 1
            elif regimen is not None:
                replicate += 1
            else:
                continue
            for marker, (c0, c1) in SCHUSTER_MARKERS.items():
                for col in range(c0, c1 + 1):
                    time_h, value = ws.cell(start + 1, col).value, ws.cell(row, col).value
                    if time_h is not None and value is not None:
                        records.append({"patient": str(patient), "regimen": regimen, "replicate": replicate,
                                        "marker": marker, "time_h": int(time_h), "value": float(value)})
    wb.close()
    tidy = pd.DataFrame(records).join(
        pd.DataFrame.from_dict(SCHUSTER_ARMS, orient="index", columns=["family", "arm"]), on="regimen")
    controls = (tidy.loc[tidy["regimen"] == "Negative Control"]
                .groupby(["patient", "marker", "time_h"], as_index=False)["value"].mean()
                .rename(columns={"value": "control_mean"}))
    tidy = tidy.merge(controls, on=["patient", "marker", "time_h"], how="left")
    tidy["score"] = np.log(tidy["value"] / tidy["control_mean"])
    return tidy


def build_schuster(workbook: Path) -> pd.DataFrame:
    tidy = read_schuster(workbook)
    wide = tidy.loc[tidy["arm"].notna() & tidy["time_h"].isin([4, 72])].pivot_table(
        index=["patient", "regimen", "replicate", "family", "arm"], columns=["marker", "time_h"],
        values="score", aggfunc="first")
    wide.columns = [f"{m}_{t}h" for m, t in wide.columns]
    wide = wide.reset_index()
    return wide.assign(cost=np.where(wide["arm"] == "high", 72.0, 4.0), state=wide["death_PI_4h"],
                       state2=wide["apoptosis_caspase_4h"], endpoint=wide["death_PI_72h"])


def read_spiller(csv_path: Path) -> pd.DataFrame:
    """Well-level dead fractions of the retained dose pairs, shared by the slice and Fig 3."""
    raw = pd.read_csv(csv_path, low_memory=False,
                      usecols=["Cell Type", "Compound", "Concentration", "Day", "LD", "ExpID", "Row", "Column"])
    raw = raw.rename(columns={"Cell Type": "patient", "Compound": "family", "Concentration": "dose", "Day": "day"})
    raw["patient"] = raw["patient"].astype(str).str.strip()
    raw["family"] = raw["family"].astype(str).str.strip()
    raw["well_uid"] = raw["ExpID"].astype(str) + "|R" + raw["Row"].astype(str) + "C" + raw["Column"].astype(str)
    raw["is_dead"] = (raw["LD"].astype(str) == "Dead").astype(float)
    wells = raw.groupby(["patient", "family", "dose", "day", "well_uid"], as_index=False).agg(dead_frac=("is_dead", "mean"))
    return pd.concat([wells.loc[(wells["family"] == fam) & wells["dose"].isin([lo, hi]) & wells["patient"].isin(SPILLER_LINES[fam])]
                      .assign(arm=lambda d, hi=hi: np.where(np.isclose(d["dose"], hi), "high", "low"))
                      for fam, (lo, hi) in SPILLER_DOSES.items()], ignore_index=True)


def build_spiller(csv_path: Path) -> pd.DataFrame:
    wells = read_spiller(csv_path)
    wide = wells.loc[wells["day"].isin([0, 7])].pivot_table(
        index=["patient", "family", "dose", "arm", "well_uid"], columns="day", values="dead_frac", aggfunc="first")
    wide = wide.dropna().reset_index()
    return wide.assign(
        regimen=wide["family"] + " " + wide["dose"].astype(str) + "uM",
        replicate=wide.groupby(["patient", "family", "dose"])["well_uid"].transform(lambda s: pd.factorize(s)[0] + 1),
        cost=wide["dose"] * 168.0, state=wide[0], state2=np.nan, endpoint=wide[7])


def read_wells(zip_path: Path) -> pd.DataFrame:
    """Per-organoid NV488 (95th-percentile pixel intensity) and area by image for the
    reporter well of every case and dose, read straight from the released archive."""
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            m = re.search(r"BCO(\d{3})_([A-D])(\d{2})_", name)
            if not m or m.group(2) != WELLS_ROW[m.group(1)]:
                continue
            patient, col = m.group(1), m.group(3)
            gy = WELLS_GY[WELLS_COLUMNS[patient].index(col)]
            wb = xlrd.open_workbook(file_contents=z.read(name))
            for sheet, var in (("Percentile Pixel Intensity2", "nv488"), ("Area2", "area")):
                sh = wb.sheet_by_name(sheet)
                images = ["P1"] + (["P2"] if sh.ncols == 12 else []) + ["R1", "R2", "R3", "R4", "R5", "A1", "A2"]
                for r in range(3, sh.nrows):
                    vals = sh.row_values(r)
                    rows += [{"patient": patient, "gy": gy, "organoid": int(vals[0]), "var": var, "image": img,
                              "value": float(v)} for img, v in zip(images, vals[1:]) if v != ""]
    return pd.DataFrame(rows)


def build_wells(zip_path: Path) -> pd.DataFrame:
    """5 Gy (low) against 30 Gy (high) for organoids read before the first fraction (R1)
    and at day 14 (A2), as log-ratios to the patient's 0 Gy mean in the same image."""
    long = read_wells(zip_path)
    controls = long.loc[long["gy"] == 0].groupby(["patient", "var", "image"], as_index=False)["value"].mean()
    long = long.merge(controls.rename(columns={"value": "control_mean"}), on=["patient", "var", "image"])
    long["score"] = np.log(long["value"] / long["control_mean"])
    wide = long.loc[long["gy"] > 0].pivot_table(index=["patient", "gy", "organoid"], columns=["var", "image"],
                                                values="score", aggfunc="first")
    wide = wide.loc[wide[[("nv488", "R1"), ("nv488", "A2"), ("area", "R1")]].notna().all(axis=1)]
    wide = pd.DataFrame({"state": wide[("nv488", "R1")], "state2": wide[("area", "R1")],
                         "endpoint": wide[("nv488", "A2")]}).reset_index()
    return wide.assign(regimen="IR " + wide["gy"].map("{:g} Gy".format), replicate=wide["organoid"], family="IR",
                       arm=np.where(wide["gy"] == WELLS_GY[-1], "high", "low"), cost=wide["gy"])


def load_checkpoint(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient": str})
    cost = df.pivot_table(index="family", columns="arm", values="cost", aggfunc="mean")
    return df.assign(
        action_logged=(df["arm"] == "high").astype(int),
        episode_uid=df["patient"] + "|" + df["regimen"] + "|" + df["family"] + "|" + df["arm"] + "|" + df["replicate"].astype(str),
        cost_low=df["family"].map(cost["low"]).astype(float), cost_high=df["family"].map(cost["high"]).astype(float))


def figure_checkpoint(workbook: Path) -> None:
    """Fig 2: family-wise checkpoint geometry in Schuster; color is patient and marker is
    arm, so the within-patient ordering by the checkpoint is visible."""
    pi = read_schuster(workbook).query("marker == 'death_PI'")
    arm_marker = {"low": "o", "high": "^"}
    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.7), sharex=True, sharey=True, layout="constrained")
        for ax, family in zip(axes.ravel(), SCHUSTER_FAMILIES):
            for regimen, arm in ((f"{family} 4 hour", "low"), (f"{family} 72 hour", "high")):
                sub = pi[pi["regimen"] == regimen]
                merged = sub[sub["time_h"] == 4].merge(sub[sub["time_h"] == 72], on=["patient", "regimen", "replicate"],
                                                       suffixes=("_4", "_72"))
                ax.scatter(merged["score_4"], merged["score_72"], color=merged["patient"].map(PATIENT_COLORS),
                           marker=arm_marker[arm], s=18, edgecolors="white", linewidths=0.35, alpha=0.9)
            ax.axhline(0, color="0.6", linestyle=":", linewidth=0.8, zorder=0)
            ax.axvline(0, color="0.6", linestyle=":", linewidth=0.8, zorder=0)
            ax.set_title(family, fontsize=7)
            ax.tick_params(labelsize=6.5)
        fig.supxlabel("Checkpoint death", fontsize=8)
        fig.supylabel("Final death", fontsize=8)
        pats = [Line2D([], [], ls="", marker="o", ms=4.5, color=c, label=f"Patient {p}") for p, c in PATIENT_COLORS.items()]
        arms = [Line2D([], [], ls="", marker=arm_marker[a], ms=4.5, color="0.45", label=lab)
                for a, lab in (("low", "Early stop"), ("high", "Continuous"))]
        # Legends fill column by column: patients on the first row, arms on the second.
        fig.legend(handles=[pats[0], arms[0], pats[1], arms[1], pats[2]], loc="outside upper center", ncol=3,
                   fontsize=6.5, handletextpad=0.3, columnspacing=1.0, **FROSTED)
        save(fig, "fig2_checkpoint_scatter", bbox_inches="tight")


def figure_spiller(csv_path: Path) -> None:
    """Fig 3: Spiller dose-arm trajectories, per line (thin) and averaged (thick)."""
    wells = read_spiller(csv_path)
    labels = {(fam, arm): f"{name} ({SPILLER_DOSES[fam][j]:g}\u2009µM)"
              for fam in SPILLER_DOSES for j, (arm, name) in enumerate((("low", "Low"), ("high", "High")))}
    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.7), sharex=True, sharey=True, layout="constrained")
        for ax, family in zip(axes, SPILLER_DOSES):
            sub = wells[wells["family"] == family]
            ax.set_title(f"{family} · {sub['patient'].nunique()} lines")
            for arm in ("low", "high"):
                by_line = sub.loc[sub["arm"] == arm].groupby(["patient", "day"], as_index=False)["dead_frac"].mean()
                for _, p in by_line.groupby("patient"):
                    p = p.sort_values("day")
                    ax.plot(p["day"], p["dead_frac"], color=ARM_COLORS[arm], alpha=0.35, linewidth=1.0)
                stats = by_line.groupby("day")["dead_frac"].agg(["mean", "sem"])
                ax.plot(stats.index, stats["mean"], color=ARM_COLORS[arm], linewidth=2.2, marker="o",
                        label=labels[(family, arm)])
                ax.fill_between(stats.index, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"],
                                color=ARM_COLORS[arm], alpha=0.18)
            ax.set_xticks([0, 1, 3, 7])
            ax.legend(loc="upper left", fontsize=6, **FROSTED)
        axes[-1].set_xlabel("Day")
        fig.supylabel("Dead-organoid fraction", fontsize=8)
        save(fig, "fig3_spiller_trajectories", bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=OUT / "raw", help="download cache (not committed)")
    ap.add_argument("--out", type=Path, default=OUT / "processed")
    args = ap.parse_args()
    raw = download(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, build in (("schuster", build_schuster), ("spiller", build_spiller), ("wells", build_wells)):
        cols = ["patient", "regimen", "replicate", "family", "arm", "cost", "state", "state2", "endpoint"]
        build(raw[name])[cols].to_csv(args.out / f"{name}.csv", index=False)
    figure_checkpoint(raw["schuster"])
    figure_spiller(raw["spiller"])


if __name__ == "__main__":
    main()
