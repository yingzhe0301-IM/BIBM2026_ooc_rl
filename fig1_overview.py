"""Fig 1: the decision structure the benchmark shares and the three stages of the protocol.

Schematic, not data-driven. The top band follows one organoid well: a shared exposure, the
state x read at the checkpoint, and the higher- and lower-burden arms read at the same
endpoint Y. The cards below show the three stages: the within-patient order of Y by x,
the ranking of the prespecified lower-burden rules by the value of Y - lambda0 C, and the
certification margin, which crosses zero at the critical rate lambda*. Arm colors match
Figs 3-5; the font is the house Seravek where it is installed. All lengths are in points.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from prep import ARM_COLORS, PATIENT_COLORS, save

INK, MUTE, RULE = "#3a3f45", "#7a8088", "#c3c8ce"
LOW, HIGH = ARM_COLORS["low"], ARM_COLORS["high"]
TITLE, MARK, LABEL, NOTE = 7.6, 6.7, 6.4, 6.0  # font sizes
W, H = 252.0, 134.0               # one IEEE column wide
M, GAP = 6.0, 4.0                 # outer margin and gap between cards
CARD_W, CARD_H = (W - 2 * M - 2 * GAP) / 3, 77.0
CARD_X = [M + i * (CARD_W + GAP) for i in range(3)]
YC = H - 28.0                     # centerline of the top band


def text(ax, x, y, s, size=LABEL, color=INK, **kw):
    ax.text(x, y, s, fontsize=size, color=color, **{"ha": "left", "va": "center", "zorder": 6, **kw})


def rounded(ax, x0, y0, w, h, face, edge, r, lw=0.6, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=face, edgecolor=edge, lw=lw, zorder=z))


def top_band(ax):
    """One organoid well over time: shared exposure, state x at the checkpoint, two arms, endpoint Y."""
    for i, wx in enumerate(M + 3.5 + 9.0 * np.arange(3)):
        for wy in (YC + 4.8, YC - 4.8):
            ax.add_patch(Circle((wx, wy), 3.5, facecolor="white", edgecolor="#a3a9b0", lw=0.6, zorder=2))
            ax.add_patch(Circle((wx + 0.55 * (i - 1), wy + 0.45), 1.4, facecolor="#cfd5dc", lw=0, zorder=2))
    text(ax, M, YC - 14.0, "organoid wells", NOTE, MUTE)
    x0, xc, x1 = 42.0, 94.0, CARD_X[2]  # the endpoint sits over the left edge of card 3
    xf = xc + 0.42 * (x1 - xc)
    ax.plot([x0, xc], [YC, YC], color="#b4bac1", lw=2.2, solid_capstyle="round", zorder=2)
    text(ax, (x0 + xc) / 2, YC - 7.5, "shared exposure", NOTE, MUTE, ha="center")
    for y, color, lw in ((YC + 9.5, HIGH, 3.4), (YC - 9.5, LOW, 1.7)):
        mx = (xc + xf) / 2  # S-curve with horizontal tangents, then straight to the endpoint
        path = MplPath([(xc, YC), (mx, YC), (mx, y), (xf, y)], [MplPath.MOVETO] + [MplPath.CURVE4] * 3)
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=lw, capstyle="round", zorder=3))
        ax.plot([xf, x1], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=3)
        ax.add_patch(Circle((x1, y), 1.8, facecolor=color, lw=0, zorder=4))
    ax.add_patch(Circle((xc, YC), 2.7, facecolor=INK, lw=0, zorder=5))
    text(ax, xc, YC + 8.0, "state x", MARK, ha="center")
    ax.plot([x1, x1], [YC - 14.0, YC + 14.0], color=RULE, lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    text(ax, x1, YC + 18.5, "endpoint Y", MARK, ha="center")
    text(ax, (xf + x1) / 2, YC, "burden C", NOTE, MUTE, ha="center")
    lx0, lx1 = x1 + 10.0, W - M  # legend card, right edge on the card grid
    rounded(ax, lx0, YC - 13.0, lx1 - lx0, 26.0, "#fbfaf7", "#d4d2cc", r=3.0, z=2)
    for y, color, lw, label in ((YC + 5.4, HIGH, 3.4, "higher burden"), (YC - 5.4, LOW, 1.7, "lower burden")):
        ax.plot([lx0 + 6.0, lx0 + 17.0], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=3)
        text(ax, lx0 + 21.0, y, label)


def stage_state(ax, x0, y0, x1, y1):
    """Each patient's wells rise with x around the patient's own level."""
    ax.plot([x0 + 3, x0 + 3, x1], [y1, y0 + 3, y0 + 3], color=RULE, lw=0.7, zorder=2)
    text(ax, x1, y0 - 0.5, "x", NOTE, MUTE, ha="right")
    text(ax, x0 - 1.5, y1 - 1.0, "Y", NOTE, MUTE, ha="center")
    rng = np.random.default_rng(3)
    for k, (color, level) in enumerate(zip(PATIENT_COLORS.values(), (0.0, 9.5, 19.0))):
        xs = x0 + 9 + 9 * k + np.linspace(0, 24, 5)
        ys = y0 + 9 + level + 0.5 * (xs - xs[0]) + rng.normal(0, 1.1, 5)
        ax.plot(xs[[0, -1]], np.polyval(np.polyfit(xs, ys, 1), xs[[0, -1]]), color=color, lw=1.0, alpha=0.8, zorder=3)
        ax.scatter(xs, ys, s=7.5, color=color, edgecolors="white", linewidths=0.3, zorder=4)


def stage_rank(ax, x0, y0, x1, y1):
    """The three prespecified lower-burden classes by value; the leader is retained."""
    rows = (("fixed arm", 0.52, False), ("threshold", 0.86, True), ("one-step CQL", 0.70, False))
    track, step = x1 - x0 - 6.0, (y1 - y0) / 3
    for i, (name, v, win) in enumerate(rows):
        y = y1 - step * (i + 0.5)
        ax.add_patch(Rectangle((x0, y - 4.4), track, 8.8, facecolor="#eef0f3", lw=0, zorder=2))
        ax.add_patch(Rectangle((x0, y - 4.4), track * v, 8.8, facecolor=LOW if win else "#c8d3e4", lw=0, zorder=3))
        text(ax, x0 + 2.5, y, name, NOTE, "white" if win else INK)
        if win:  # check mark after the retained rule
            cx = x0 + track * v + 4.4
            ax.plot([cx - 2.0, cx - 0.6, cx + 2.0], [y, y - 1.5, y + 1.6], color=LOW, lw=1.1,
                    solid_capstyle="round", solid_joinstyle="round", zorder=6)


def stage_certify(ax, x0, y0, x1, y1):
    """The margin M(lambda) = LCB + lambda B rises with lambda and certifies from lambda*."""
    ya = y0 + 7.0  # the lambda axis labels sit below the plot
    x_star, x_ref, y_zero = x0 + 0.42 * (x1 - x0), x0 + 0.68 * (x1 - x0), (ya + y1) / 2
    ax.add_patch(Rectangle((x0, ya), x_star - x0, y1 - ya, facecolor="#eceef0", lw=0, zorder=1))
    ax.add_patch(Rectangle((x_star, ya), x1 - x_star, y1 - ya, facecolor="#e3f1e5", lw=0, zorder=1))
    ax.plot([x0, x1], [y_zero, y_zero], color=INK, lw=0.6, zorder=2)
    slope = 0.75 * (y_zero - ya) / (x_star - x0)  # the margin line crosses zero at x_star
    ax.plot([x0, x1], [y_zero - slope * (x_star - x0), y_zero + slope * (x1 - x_star)], color=INK, lw=1.4,
            solid_capstyle="round", zorder=3)
    ax.plot([x_ref, x_ref], [ya, y1], color=MUTE, lw=0.7, ls=(0, (1.2, 1.4)), zorder=2)
    ax.add_patch(Circle((x_star, y_zero), 1.7, facecolor=INK, lw=0, zorder=4))
    text(ax, x_star, y0 + 1.8, "λ*", MARK, ha="center")
    text(ax, x_ref, y0 + 1.8, "λ₀", MARK, MUTE, ha="center")
    text(ax, x1, y0 + 1.8, "λ", NOTE, MUTE, ha="right")
    text(ax, x1 - 1.5, ya + 5.0, "certified", NOTE, "#3d7a4a", ha="right")


def main() -> None:
    try:
        font_manager.fontManager.addfont("/System/Library/Fonts/Supplemental/Seravek.ttc")
        family = ["Seravek", "DejaVu Sans"]
    except (FileNotFoundError, RuntimeError):
        family = ["DejaVu Sans"]
    with plt.rc_context({"font.family": family, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=(W / 72, H / 72))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        rounded(ax, 1.0, 1.0, W - 2.0, H - 2.0, "#f5f7fa", "#d9dee6", r=6.0, lw=0.8, z=0)
        top_band(ax)
        stages = (("State diagnostic", "within-patient order", stage_state),
                  ("Rule ranking", "value of Y − λ₀C", stage_rank),
                  ("Certification", "LCB + λB ≥ 0", stage_certify))
        for n, (x0, (title, sub, draw)) in enumerate(zip(CARD_X, stages), start=1):
            top = M + CARD_H
            rounded(ax, x0, M, CARD_W, CARD_H, "white", "#dfe3e8", r=4.0)
            ax.add_patch(Circle((x0 + 8.5, top - 9.0), 4.1, facecolor=INK, lw=0, zorder=4))
            text(ax, x0 + 8.5, top - 9.1, str(n), NOTE, "white", ha="center")
            text(ax, x0 + 15.5, top - 9.0, title, TITLE)
            text(ax, x0 + 15.5, top - 18.5, sub, NOTE, MUTE)
            draw(ax, x0 + 6.0, M + 5.0, x0 + CARD_W - 6.0, top - 27.0)
        save(fig, "fig1_overview")


if __name__ == "__main__":
    main()
