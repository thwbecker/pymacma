"""
pymacma.paper_figures
=====================

Recreates the model figures of Grigne & Combes (2020, G-cubed,
doi:10.1029/2020GC009123) with pymacma:

  fig01c  subcontinental shear force vs continent age (Fig. 1c)
  fig02   present-day plate speeds vs age at the trench (Fig. 2)
  fig03   fixed plate boundaries: configurations, T(t), Nu-Ra (Fig. 3)
  fig04   plate reorganization over ~40 Myr (Fig. 4)
  fig05   T, heat flow, radiogenic heating, ridges over 3 Gyr (Fig. 5)
  fig06   ensemble statistics over 3 Gyr (Fig. 6)
  fig07   effect of activation energy and viscosity (Fig. 7)
  fig08   effect of continental strength and critical age (Fig. 8)
  figA1   convective forces, velocity and viscosity vs T (Fig. A1)

Usage (from the package directory):

  python make_paper_figures.py --mode quick              # all figures, ~1 h on 1 core
  python make_paper_figures.py --mode full --procs 32    # paper ensemble sizes
  python make_paper_figures.py --figs fig02 fig03 --outdir figs
  python make_paper_figures.py --set E=300e3 --fig-set n_ens=50
  python make_paper_figures.py --config configs/reference_2020.toml

Model results are cached in <outdir>/cache (keyed by all settings), so
figures can be restyled and regenerated without rerunning; use --refresh to
recompute.

Observational data (Herzberg et al. 2010 temperatures, plate
reconstructions, collision counts) are not distributed with this package.
If CSV files are placed in --data-dir they are overlaid; see README.md for
the expected file names and columns.

Because evolving-boundary runs are chaotic, individual runs (Figs. 4-5)
cannot reproduce the paper's particular events; the ensemble figures are
the meaningful comparison.

Written by Claude (Anthropic), October 2026, for Thorsten Becker.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, fields, replace, asdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import Params, Model, Plate, Ocean, Cont, fixed_earth, earth_plates_test, _coerce
from . import plots, experiments as ex


# ---------------------------------------------------------------------------
# settings for the figure computations (distinct from model parameters)
# ---------------------------------------------------------------------------
@dataclass
class FigSettings:
    """sizes and numerics of the figure computations.
    Presets: 'quick' (default values below) and 'full' (paper sizes)."""
    n_ens: int = 16                  # runs in Fig. 6 (paper: 200)
    n_sweep: int = 4                 # runs per parameter value, Figs. 7-8 (paper: 50)
    n_cal: int = 2                   # runs used to calibrate T0 per value
    n_coll: int = 16                 # runs summed for collision counts (paper: 16)
    ens_dx: float = 1.0              # numerics of evolving ensembles (deg)
    ens_dt_max: float = 2e5          # (yr); quick values bias T(0) by a few K
    fixed_dx: float = 1.0            # numerics of fixed-boundary runs
    fixed_dt_max: float = 2e5
    fixed_tol: float = 20.0          # tolerance of fixed start ages (Myr)
    ref_seed: int = 1                # run shown in Figs. 4-5
    ref_window: tuple = (1000.0, 2000.0)   # ages (Ma) searched for the Fig. 4 event
    out_every: float = 2.5           # output interval of ensemble runs (Myr)
    E_values: tuple = (250e3, 300e3, 350e3, 400e3, 450e3)       # Fig. 7a-c (J/mol)
    eta_factors: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)             # Fig. 7d-f
    Flim_values: tuple = (1e12, 1.5e12, 2e12, 4e12, float("inf"))  # Fig. 8a-c (N/m)
    tau_values: tuple = (140.0, 160.0, 180.0, 200.0, 220.0)     # Fig. 8d-f (Myr)
    procs: int = 0                   # processes (0: all cores)

    @classmethod
    def preset(cls, mode):
        if mode == "full":
            return cls(n_ens=200, n_sweep=50, n_cal=8, ens_dx=0.5, ens_dt_max=5e4,
                       fixed_dx=0.5, fixed_dt_max=5e4, fixed_tol=5.0, out_every=2.5)
        if mode == "quick":
            return cls()
        raise ValueError("mode must be 'quick' or 'full'")

    def update(self, d):
        kw = {}
        for f in fields(self):
            if f.name in d:
                v = d[f.name]
                cur = getattr(self, f.name)
                if isinstance(cur, tuple) and isinstance(v, str):
                    v = tuple(float(x) for x in v.split(","))
                elif isinstance(cur, tuple):
                    v = tuple(v)
                elif isinstance(cur, int) and not isinstance(cur, bool):
                    v = int(float(v))
                else:
                    v = _coerce(cur, v)
                kw[f.name] = v
        bad = set(d) - {f.name for f in fields(self)}
        if bad:
            raise KeyError("unknown figure setting(s): %s" % ", ".join(sorted(bad)))
        return replace(self, **kw)


# Fig. 2 plate data as printed in the table of 2020 Fig. 2 (values from
# Argus et al., 2011): length (km), observed speed and 95% limit (deg/Myr)
FIG2_PLATES = {
    "JF": (570, 0.951, 0.256), "CO": (1930, 1.198, 0.045),
    "NZ": (4550, 0.696, 0.029), "PA": (11970, 0.651, 0.011),
    "NA": (7300, 0.209, 0.013), "SA": (8560, 0.109, 0.011),
}
# for the overriding plates NA and SA, the paper uses the trench age and
# slab depth of the plates they override (PA and NZ) and their own maximum
# oceanic age for ridge push: 180 Ma (NA) and 110 Ma (SA)
FIG2_RP_AGE = {"NA": 180.0, "SA": 110.0}
# approximate present-day ranges of seafloor age at the trench (Myr), used to
# highlight the relevant part of each curve. These are estimates made for
# this script (in the spirit of the paper, which used the age grid of
# Muller et al., 2008), not values taken from the paper; edit as needed.
# NA and SA use the ranges of the plates they override (PA and NZ).
FIG2_AGE_RANGES = {"JF": (5, 12), "CO": (10, 25), "NZ": (25, 50), "PA": (70, 180),
                   "NA": (70, 180), "SA": (25, 50)}
FIG2_COLORS = {"JF": "tab:red", "CO": "tab:red", "NZ": "tab:orange",
               "PA": "tab:orange", "NA": "teal", "SA": "tab:blue"}

PAPER_FIXED_AGES = {1: 2550.0, 2: 1150.0, 4: 660.0, 8: 400.0}   # 2020 Fig. 3 labels


class Ctx:
    """shared state of one figure-making session"""

    def __init__(self, base, S, outdir, data_dir, refresh):
        self.base, self.S = base, S
        self.outdir = outdir
        self.cache_dir = os.path.join(outdir, "cache")
        self.data_dir = data_dir
        self.refresh = refresh
        os.makedirs(self.cache_dir, exist_ok=True)

    def cache(self, name, fn, *key):
        path = os.path.join(self.cache_dir, "%s_%s.pkl" % (name, ex.params_key(*key)))
        return ex.cached(path, fn, self.refresh)

    @property
    def procs(self):
        return self.S.procs or None

    def ens_params(self, **kw):
        return self.base.copy(dx=self.S.ens_dx, dt_max=self.S.ens_dt_max, **kw)

    def fixed_params(self, **kw):
        return self.base.copy(dx=self.S.fixed_dx, dt_max=self.S.fixed_dt_max, **kw)

    def obs(self, name):
        """optional observational data: <data_dir>/<name>.csv with a header"""
        if not self.data_dir:
            return None
        path = os.path.join(self.data_dir, name + ".csv")
        if not os.path.exists(path):
            return None
        return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)

    def save(self, fig, name):
        path = os.path.join(self.outdir, name + ".png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print("  wrote", path, flush=True)


def _herzberg(ax, ctx):
    d = ctx.obs("herzberg2010")
    if d is not None:
        ax.plot(d["age_Ga"], d["T_K"], "o", mfc="gold", mec="k", ms=5,
                label="Herzberg et al. (2010)", zorder=5)


def _overlay(ax, ctx, name, ycol):
    d = ctx.obs(name)
    if d is None:
        return
    labels = d["label"] if "label" in d.dtype.names else np.array(["data"] * len(d))
    for lab in np.unique(labels):
        m = labels == lab
        ax.plot(d["age_Ga"][m], d[ycol][m], "-", lw=1.5, label=str(lab))
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Fig. 1c
# ---------------------------------------------------------------------------
def fig01c(ctx):
    """subcontinental shear force F(t) for continents of 2000-16000 km,
    present day (T = 1625 K, eta_um = 1e21 Pa s, H = 13 TW) and late
    Archean (T = 1850 K, H = 25.7 TW, E = 300 kJ/mol), as in the caption of
    2020 Fig. 1c"""
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    t = np.linspace(0, 1000e6, 400)
    for ax, (T, H, lab) in zip(axs, [(1625.0, 13.0, "present: T = 1625 K, H = 13 TW"),
                                     (1850.0, 25.7, "past: T = 1850 K, H = 25.7 TW")]):
        p = ctx.base.copy(eta_um_p=1e21, E=300e3, T_p=1625.0, H_constant=H)
        m = Model(p, 0, [Plate([Ocean([0, 360], [0, 0])])], [], T, 0)
        for a_km, col in [(2000, "tab:red"), (4000, "tab:red"), (8000, "tab:orange"),
                          (16000, "orange")]:
            c = Cont(a_km * 1e3 / m.deg2m)
            F = []
            for tt in t:
                c.life = tt
                F.append(m.cont_force(c) / 1e12)
            ax.plot(t / 1e6, F, color=col, lw=1 + a_km / 8000)
            ax.text(t[-1] / 1e6, F[-1], " %d km" % a_km, fontsize=7, va="center")
        ax.axhline(ctx.base.F_lim / 1e12, color="k", ls=":", lw=0.8)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("age of continent (Myr)")
        ax.set_xlim(0, 1150)
    axs[0].set_ylabel("subcontinental shear force F (TN/m)")
    axs[0].set_ylim(0, 9)
    fig.suptitle("Fig. 1c: dotted line is F_lim = %.1f TN/m" % (ctx.base.F_lim / 1e12))
    fig.tight_layout()
    ctx.save(fig, "fig01c")


# ---------------------------------------------------------------------------
# Fig. 2
# ---------------------------------------------------------------------------
def fig02(ctx):
    """speeds from the force balance (Eq. 8) for plates of the lengths of
    JF, CO, NZ, PA, NA and SA, versus seafloor age at the trench, for slab
    depths 500-2800 km; observed speeds (Argus et al., 2011) as bands.
    Thick parts of the curves: approximate present-day ranges of trench ages
    (FIG2_AGE_RANGES, estimates); faint: full range."""
    sp = earth_plates_test(ctx.base)
    ages = np.linspace(1, 200, 120)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {500: ":", 1000: "-.", 2000: "--", 2800: "-"}
    for name, (L, obs, err) in FIG2_PLATES.items():
        col = FIG2_COLORS[name]
        sub = name not in FIG2_RP_AGE
        a0, a1 = FIG2_AGE_RANGES[name]
        for Z, ls in styles.items():
            v = np.array([sp(L, a, Z, sub, FIG2_RP_AGE.get(name)) for a in ages])
            ax.plot(ages, v, ls, color=col, lw=0.5, alpha=0.35)
            m = (ages >= a0) & (ages <= a1)
            ax.plot(ages[m], v[m], ls, color=col, lw=1.8)
        ax.axhspan(obs - err, obs + err, color=col, alpha=0.15, lw=0)
        ax.axhline(obs, color=col, lw=1.5)
        ax.text(201, obs, name, color=col, fontsize=8, va="center")
    for Z, ls in styles.items():
        ax.plot([], [], ls, color="k", label="slab depth %d km" % Z)
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 1.8)
    ax.set_xlabel("age at the trench (Ma)")
    ax.set_ylabel("speed (deg/Myr)")
    sec = ax.secondary_yaxis("right", functions=(lambda d: d * 11.12, lambda c: c / 11.12))
    sec.set_ylabel("speed (cm/yr)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Fig. 2: predicted plate speeds (lines; thick: approximate present trench "
                 "ages) and observed (bands)",
                 fontsize=9)
    fig.tight_layout()
    ctx.save(fig, "fig02")


# ---------------------------------------------------------------------------
# fixed boundaries (Fig. 3 and the thin lines of Figs. 6-8)
# ---------------------------------------------------------------------------
def fixed_case(ctx, n_cont, params=None, end_Ma=-250.0):
    """fixed-boundary run starting at 2000 K whose present temperature is
    1625 K; returns (start age, history with a snapshot at the start)"""
    p = params or ctx.fixed_params()

    def go():
        age0 = ex.fixed_start_age(n_cont, p, guess=PAPER_FIXED_AGES.get(n_cont, 1150.0),
                                  tol=ctx.S.fixed_tol)
        m = fixed_earth(n_cont, params=p, T0=2000.0, age_Ma=age0)
        h = m.run(end_Ma, out_every_Myr=5.0, snap_ages=(age0,))
        return age0, h
    return ctx.cache("fixed%d" % n_cont, go, p, n_cont, end_Ma, ctx.S.fixed_tol)


def nu_ra(h, p):
    """Nusselt and Rayleigh numbers of a run (Eqs. 13-14). The heat flux q
    is the total heat flow divided by Earth's surface."""
    d = p.d
    eta = p.eta_lm_p * np.exp(p.E / 8.314 * (1 / h["T"] - 1 / p.T_p))
    dT = h["T"] - p.T_surf
    Ra = p.alpha_um * p.rho_um_p * p.g * dT * d ** 3 / (p.kappa * eta)
    q = h["Q"] * 1e12 / (4 * np.pi * p.R_E ** 2)
    Nu = q * d / (p.k_ocean * dT)
    return Nu, Ra


def fig03(ctx):
    """fixed boundaries: initial configurations (a-d), temperature (e), and
    Nu versus Ra (f) for 1, 2, 4 and 8 continents (3-24 plates)"""
    fig = plt.figure(figsize=(12, 9))
    cases = (1, 2, 4, 8)
    styles = {1: ("royalblue", "-."), 2: ("k", "-"), 4: ("forestgreen", "--"),
              8: ("magenta", "-.")}
    res = {}
    for k, n in enumerate(cases):
        age0, h = fixed_case(ctx, n)
        res[n] = (age0, h)
        ax = fig.add_axes([0.02 + 0.245 * k, 0.53, 0.23, 0.42])
        snap = h["snaps"][age0]
        plots.draw_annulus(ax, snap, title="(%s) %d continent(s), %d plates\nstart %.0f Ma "
                           "(paper %.0f Ma), T = 2000 K"
                           % ("abcd"[k], n, 3 * n, age0, PAPER_FIXED_AGES[n]), fontsize=8)
    axe = fig.add_axes([0.08, 0.07, 0.4, 0.45])
    axf = fig.add_axes([0.57, 0.07, 0.4, 0.45])
    p = ctx.fixed_params()
    for n in cases:
        age0, h = res[n]
        col, ls = styles[n]
        axe.plot(h["age_Ma"] / 1e3, h["T"], ls, color=col, label="N$_{cont}$ = %d" % n)
        Nu, Ra = nu_ra(h, p)
        keep = h["age_Ma"] < age0 - 100.0        # skip the initial transient
        Nu, Ra = Nu[keep], Ra[keep]
        sl = np.polyfit(np.log(Ra), np.log(Nu), 1)[0]
        axf.loglog(Ra, Nu, ls, color=col, label="N$_{cont}$ = %d (slope %.2f)" % (n, sl))
    _herzberg(axe, ctx)
    axe.axhline(1625, color="0.5", lw=0.6)
    axe.axvline(0, color="0.5", lw=0.6)
    axe.set_xlim(-0.25, 3)
    axe.set_ylim(1550, 1980)
    axe.set_xlabel("age (Ga)")
    axe.set_ylabel("temperature (K)")
    axe.set_title("(e) temperature evolution", fontsize=9)
    axe.legend(fontsize=8)
    # reference slope 1/3
    Ra0 = np.logspace(5.5, 8, 10)
    axf.fill_between(Ra0, 0.6 * 20 * (Ra0 / 1e6) ** (1 / 3), 1.6 * 20 * (Ra0 / 1e6) ** (1 / 3),
                     color="0.85", lw=0, label="Nu ~ Ra$^{1/3}$")
    axf.set_xlabel("Rayleigh number")
    axf.set_ylabel("Nusselt number")
    axf.set_title("(f) Rayleigh-Nusselt relation (first 100 Myr of each run omitted)",
                  fontsize=9)
    axf.legend(fontsize=8)
    ctx.save(fig, "fig03")


# ---------------------------------------------------------------------------
# Figs. 4 and 5 (one reference run)
# ---------------------------------------------------------------------------
REORG_OFFSETS = {"A": 0.0, "B": 21.0, "C": 41.0, "D": 75.0, "E": 90.0, "F": 105.0}
# offsets (Myr) after time A of the snapshots in 2020 Figs. 4 and 5
# (A 1415, B 1394, C 1374, D 1340, E 1325, F 1310 Ma)


def reference_run(ctx):
    """run ref_seed at the default numerics, locate a strong heat-flow
    increase (a reorganization) in ref_window, and rerun the same
    (deterministic) case with snapshots at the times A-F"""
    p = ctx.base

    def go():
        h = ex.run_one(ctx.S.ref_seed, p, out_every=1.0)
        a, Q = h["age_Ma"], h["Q"]
        lo, hi = ctx.S.ref_window
        best, tA = -np.inf, None
        for i in range(len(a)):
            if not (lo <= a[i] <= hi):
                continue
            j = np.searchsorted(-a, -(a[i] - 20.0))   # 20 Myr later
            if j < len(a) and h["moving"][i]:
                inc = Q[j] - Q[i]
                if inc > best:
                    best, tA = inc, a[i]
        if tA is None:                         # no moving phase in the window
            tA = float(a[np.argmax(np.diff(Q, prepend=Q[0]))])
        snaps = [tA - off for off in REORG_OFFSETS.values()]
        h2 = ex.run_one(ctx.S.ref_seed, p, out_every=0.5, snap_ages=snaps, keep_snaps=True)
        return tA, h2
    return ctx.cache("reference", go, p, ctx.S.ref_seed, ctx.S.ref_window)


def fig04(ctx):
    """plate organization, seafloor ages and heat flux at three times over
    ~40 Myr around a reorganization of the reference run"""
    tA, h = reference_run(ctx)
    fig = plt.figure(figsize=(13, 13))
    for r, lab in enumerate("ABC"):
        age = tA - REORG_OFFSETS[lab]
        snap = h["snaps"][age]
        y0 = 0.655 - 0.305 * r
        ax0 = fig.add_axes([0.01, y0, 0.3, 0.29])
        ax2 = fig.add_axes([0.38, y0 + 0.16, 0.59, 0.12])
        ax1 = fig.add_axes([0.38, y0 + 0.02, 0.59, 0.13], sharex=ax2)
        nplates = len(snap["plates"])
        plots.draw_annulus(ax0, snap, title="(%s) %d plates" % (lab, nplates), fontsize=10)
        plots.draw_age_heat(ax1, ax2, snap)
        ax2.set_title("Age = %.0f Ma   T = %.1f K   Q = %.1f TW"
                      % (snap["age_Ma"], snap["T"], snap["Q"]), fontsize=10)
        plt.setp(ax2.get_xticklabels(), visible=False)
        if r == 2:
            ax1.set_xlabel("position (deg, counterclockwise)")
    fig.suptitle("Fig. 4: seed %d. Plate colours: red/green moving toward larger/smaller "
                 "angles, blue immobile" % ctx.S.ref_seed, fontsize=10, y=0.985)
    ctx.save(fig, "fig04")


def fig05(ctx):
    """temperature (a), heat flow, radiogenic heating and number of ridges
    (b, with a zoom on the reorganization A-F) and seafloor ages at D, E, F
    (c) for the reference run"""
    tA, h = reference_run(ctx)
    fig = plt.figure(figsize=(14, 8))
    axa = fig.add_axes([0.07, 0.62, 0.5, 0.33])
    axb = fig.add_axes([0.07, 0.08, 0.5, 0.5], sharex=axa)
    age = h["age_Ma"] / 1e3
    axa.plot(age, h["T"], "k-")
    axa.axhspan(1600, 1650, color="tab:blue", alpha=0.2, lw=0)
    axa.text(0.3, 1630, "T = 1625 ± 25 K", fontsize=8)
    axa.axvline(tA / 1e3, color="b", lw=0.8)
    axa.set_ylabel("temperature (K)")
    axa.set_title("(a)", loc="left", fontsize=9)
    plt.setp(axa.get_xticklabels(), visible=False)
    axb.plot(age, h["Q"], "k-", lw=0.6, label="heat flow Q")
    axb.plot(age, h["H"], "r-", lw=1.5, label="radioactive heating")
    axb.set_ylabel("power (TW)")
    axb.set_xlabel("age (Ga)")
    axb.axvline(tA / 1e3, color="b", lw=0.8)
    tw = axb.twinx()
    tw.step(age, h["n_ridges"], color="orange", lw=0.8, where="post")
    tw.set_ylabel("number of ridges", color="orange")
    tw.set_ylim(-0.5, max(8, h["n_ridges"].max() + 1))
    axb.legend(fontsize=8, loc="upper left")
    axb.set_xlim(3.0, -0.02)
    axb.set_title("(b)", loc="left", fontsize=9)
    # inset: zoom on the reorganization
    ins = axb.inset_axes([0.42, 0.55, 0.3, 0.4])
    sel = (h["age_Ma"] <= tA + 15) & (h["age_Ma"] >= tA - 120)
    ins.plot(h["age_Ma"][sel] / 1e3, h["Q"][sel], "k-", lw=1)
    for lab, off in REORG_OFFSETS.items():
        ins.axvline((tA - off) / 1e3, color="g", lw=0.6)
        ins.text((tA - off) / 1e3, ins.get_ylim()[1], lab, fontsize=7, ha="center",
                 va="bottom", color="g")
    ins.invert_xaxis()
    ins.tick_params(labelsize=7)
    ins.set_ylabel("Q (TW)", fontsize=7)
    # (c) ages at D, E, F
    for r, lab in enumerate("DEF"):
        ax = fig.add_axes([0.65, 0.69 - 0.305 * r, 0.33, 0.22])
        snap = h["snaps"][tA - REORG_OFFSETS[lab]]
        plots.draw_age_heat(ax, None, snap)
        ax.set_title("(%s) Age = %.0f Ma   Q = %.1f TW" % (lab, snap["age_Ma"], snap["Q"]),
                     fontsize=9)
        if r == 2:
            ax.set_xlabel("position (deg)")
    fig.suptitle("Fig. 5: reference run, seed %d (T0 = %.1f K at %.0f Ma)"
                 % (h["seed"], h["T0"], h["start_Ma"]), fontsize=10)
    ctx.save(fig, "fig05")


# ---------------------------------------------------------------------------
# Fig. 6
# ---------------------------------------------------------------------------
def ensemble(ctx, params, n, T0=None, name="ens"):
    return ctx.cache(name, lambda: ex.run_ensemble(n, params, processes=ctx.procs,
                                                   T0=T0, out_every=ctx.S.out_every),
                     params, n, T0, ctx.S.out_every)


def fig06(ctx):
    """long-term evolution over 3 Gyr: median and percentiles of n_ens runs
    from 1875 +/- 5 K at 3000 +/- 50 Ma; thin line: fixed boundaries with
    six plates"""
    hists = ensemble(ctx, ctx.ens_params(), ctx.S.n_ens)
    age0, hf = fixed_case(ctx, 2)
    ages = np.arange(0.0, 3005.0, 5.0)
    x = ages / 1e3
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    axs = axs.ravel()
    keep = hf["age_Ma"] < age0 - 20.0            # omit the initial transient
    hf = {k: v[keep] for k, v in hf.items() if isinstance(v, np.ndarray)}
    fa, fT = hf["age_Ma"] / 1e3, hf["T"]
    # (a) temperature
    plots.band(axs[0], x, plots.stack(hists, "T", ages))
    axs[0].plot(fa, fT, "k-", lw=0.6)
    axs[0].text(fa[len(fa) // 3], fT[len(fa) // 3], " fixed", fontsize=7)
    axs[0].axhline(1625, color="0.5", lw=0.6)
    _herzberg(axs[0], ctx)
    axs[0].set_ylim(1580, 1960)
    axs[0].set_ylabel("temperature (K)")
    # (b) plate speed and (c) continent speed, immobile phases filtered
    plots.band(axs[1], x, plots.stack(hists, "urms", ages, True), log=True)
    axs[1].plot(fa, hf["urms"], "k-", lw=0.6)
    _overlay(axs[1], ctx, "plate_speed", "speed_cm_yr")
    axs[1].set_ylabel("plate speed (cm/yr), RMS")
    plots.band(axs[2], x, plots.stack(hists, "cont_rms", ages, True), log=True)
    _overlay(axs[2], ctx, "continent_speed", "speed_cm_yr")
    axs[2].set_ylabel("continent speed (cm/yr), RMS")
    for ax in axs[1:3]:
        ax.set_ylim(0.5, 200)
    # (d) subduction area flux
    plots.band(axs[3], x, plots.stack(hists, "flux", ages, True))
    axs[3].plot(fa, hf["flux"], "k-", lw=0.6)
    _overlay(axs[3], ctx, "subduction_flux", "flux_km2_yr")
    axs[3].set_ylabel("subduction area flux (km$^2$/yr)")
    # (e) number of plates converted to a sphere (Appendix D)
    plots.band(axs[4], x, plots.n_plates_sphere(plots.stack(hists, "n_plates", ages)))
    axs[4].axhline(plots.n_plates_sphere([6])[0], color="k", lw=0.6)
    _overlay(axs[4], ctx, "n_plates", "n_plates")
    axs[4].set_ylabel("number of plates (modified to sphere)")
    # (f) continental collisions per 100 Myr, summed over n_coll runs
    nc = min(ctx.S.n_coll, len(hists))
    c, cnt = plots.collisions_per_window(hists[:nc])
    axs[5].plot(c / 1e3, cnt, "kD-", ms=3, label="this model (%d runs)" % nc)
    m = c <= 2500
    if m.sum() > 2:
        a, b = np.polyfit(c[m] / 1e3, cnt[m], 1)
        axs[5].plot(c[m] / 1e3, a * c[m] / 1e3 + b, "k--", lw=0.8,
                    label="regression < 2.5 Ga: %.1f per Ga" % (-a))
    _overlay(axs[5], ctx, "collisions", "count")
    axs[5].legend(fontsize=7)
    axs[5].set_ylabel("number of continental collisions")
    for k, ax in enumerate(axs):
        ax.set_xlim(3, -0.02)
        ax.set_xlabel("age (Ga)")
        ax.set_title("(%s)" % "abcdef"[k], loc="left", fontsize=9)
    T_end = np.array([h["T"][-1] for h in hists])
    fig.suptitle("Fig. 6: %d runs, median (line), 25-75%% and 5-95%% (grey); present "
                 "T = %.1f ± %.1f K; numerics dx = %g°, dt_max = %g kyr"
                 % (len(hists), T_end.mean(), T_end.std(), ctx.S.ens_dx,
                    ctx.S.ens_dt_max / 1e3), fontsize=10)
    fig.tight_layout()
    ctx.save(fig, "fig06")


# ---------------------------------------------------------------------------
# Figs. 7 and 8: parameter sweeps
# ---------------------------------------------------------------------------
def sweep_value(ctx, params):
    """calibrate T0 for one parameter set, then run the ensemble"""
    def go():
        T0, tried = ex.calibrate_T0(params, n_seeds=ctx.S.n_cal, processes=ctx.procs)
        hs = ex.run_ensemble(ctx.S.n_sweep, params, processes=ctx.procs, T0=T0,
                             out_every=ctx.S.out_every)
        return T0, tried, hs
    return ctx.cache("sweep", go, params, ctx.S.n_cal, ctx.S.n_sweep, ctx.S.out_every)


def _sweep_row(ctx, axs, items, title, fixed_fn):
    """one row of Fig. 7/8: items = [(label, params)]"""
    Tbins = np.arange(1600, 1910, 10.0)
    ages = np.arange(0, 3005, 10.0)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(items)))
    for (lab, p), col in zip(items, cols):
        T0, tried, hs = sweep_value(ctx, p)
        Tm = plots._nanpct(plots.stack(hs, "T", ages), 50)
        axs[0].plot(ages / 1e3, Tm, "-", color=col, lw=2,
                    label="%s (T0 %.0f K)" % (lab, T0))
        tc, mU = plots.binned_by_T(hs, "urms", Tbins, True)
        axs[1].semilogy(tc, mU, "o-", color=col, ms=3, lw=1.5)
        tc, mN = plots.binned_by_T(hs, "n_plates", Tbins)
        axs[2].plot(tc, mN, "o-", color=col, ms=3, lw=1.5)
        fx = fixed_fn(p)
        if fx is not None:
            age0, hf = fx
            axs[0].plot(hf["age_Ma"] / 1e3, hf["T"], "-", color=col, lw=0.7)
            axs[1].semilogy(hf["T"], hf["urms"], "-", color=col, lw=0.7)
    _herzberg(axs[0], ctx)
    axs[0].set_xlim(3, 0)
    axs[0].set_ylim(1580, 1960)
    axs[0].set_xlabel("age (Ga)")
    axs[0].set_ylabel("temperature (K)")
    axs[0].legend(fontsize=7, title=title, title_fontsize=8)
    axs[1].set_xlabel("temperature (K)")
    axs[1].set_ylabel("plate speed (cm/yr), RMS")
    axs[2].set_xlabel("temperature (K)")
    axs[2].set_ylabel("number of plates (circle)")
    for ax in axs[1:]:
        ax.axvline(1625, color="0.6", lw=0.6)
        ax.set_xlim(1590, 1880)


def fig07(ctx):
    """effect of the activation energy E (a-c) and of the reference
    viscosity eta_lm0 (d-f; eta_um0 and eta_pl0 are scaled by the same
    factor, keeping the paper's ratios eta_lm0/eta_um0 = 100 and
    eta_pl0/eta_um0 = 250). Thick lines: evolving boundaries (median T,
    means in 10 K bins); thin lines: fixed boundaries, six plates"""
    S, b = ctx.S, ctx.ens_params()
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))

    def fixed_for(p):
        pf = ctx.fixed_params(E=p.E, eta_lm_p=p.eta_lm_p, eta_um_p=p.eta_um_p,
                              eta_pl_p=p.eta_pl_p)
        return fixed_case(ctx, 2, pf)
    itemsE = [("E = %.0f kJ/mol" % (E / 1e3), b.copy(E=E)) for E in S.E_values]
    _sweep_row(ctx, axs[0], itemsE, "activation energy", fixed_for)
    itemsV = [("eta_lm0 = %.3g Pa s" % (b.eta_lm_p * f),
               b.copy(eta_lm_p=b.eta_lm_p * f, eta_um_p=b.eta_um_p * f,
                      eta_pl_p=b.eta_pl_p * f)) for f in S.eta_factors]
    _sweep_row(ctx, axs[1], itemsV, "reference viscosity", fixed_for)
    for k, ax in enumerate(axs.ravel()):
        ax.set_title("(%s)" % "abcdef"[k], loc="left", fontsize=9)
    fig.suptitle("Fig. 7: %d runs per value, T0 calibrated on %d runs; thin lines: fixed "
                 "boundaries (six plates)" % (S.n_sweep, S.n_cal), fontsize=10)
    fig.tight_layout()
    ctx.save(fig, "fig07")


def fig08(ctx):
    """effect of the continental strength F_lim (a-c) and of the present-day
    critical age for subduction tau_subd0 (d-f). The fixed-boundary case
    does not depend on these, so one thin line is shown."""
    S, b = ctx.S, ctx.ens_params()
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fixed_ref = fixed_case(ctx, 2)
    shown = {"done": False}

    def fixed_once(p):
        if shown["done"]:
            return None
        shown["done"] = True
        return fixed_ref
    itemsF = [("F_lim = %s TN/m" % ("inf" if not np.isfinite(F) else "%g" % (F / 1e12)),
               b.copy(F_lim=F)) for F in S.Flim_values]
    _sweep_row(ctx, axs[0], itemsF, "continental strength", fixed_once)
    shown["done"] = False
    itemsT = [("tau_subd0 = %g Myr" % t, b.copy(tau_sub_p=t)) for t in S.tau_values]
    _sweep_row(ctx, axs[1], itemsT, "critical age", fixed_once)
    for k, ax in enumerate(axs.ravel()):
        ax.set_title("(%s)" % "abcdef"[k], loc="left", fontsize=9)
    fig.suptitle("Fig. 8: %d runs per value, T0 calibrated on %d runs; thin line: fixed "
                 "boundaries (six plates)" % (S.n_sweep, S.n_cal), fontsize=10)
    fig.tight_layout()
    ctx.save(fig, "fig08")


# ---------------------------------------------------------------------------
# Fig. A1
# ---------------------------------------------------------------------------
def figA1(ctx):
    """relative convective force (Eq. A7) and relative velocity and
    viscosity (Eq. A6) versus mantle temperature, for activation energies
    0-450 kJ/mol; cell length fixed, Delta T taken as T - T_s"""
    p = ctx.base
    T = np.linspace(1580, 1920, 200)
    T0 = p.T_p
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    Es = [0, 20, 35, 50, 150, 250, 350, 450]
    cols = plt.cm.plasma(np.linspace(0, 0.9, len(Es)))
    for E, col in zip(Es, cols):
        eta_r = np.exp(E * 1e3 / 8.314 * (1 / T - 1 / T0))
        dT_r = (T - p.T_surf) / (T0 - p.T_surf)
        F = eta_r ** (1 / 3) * dT_r ** (2 / 3)          # Eq. A7
        U = eta_r ** (-2 / 3) * dT_r ** (2 / 3)         # Eq. A6
        axs[0].plot(T, F, color=col, label="%d" % E)
        axs[1].semilogy(T, eta_r, "-", color=col)
        axs[1].semilogy(T, U, "--", color=col)
    axs[0].set_ylabel("relative convective forces F/F$_0$")
    axs[1].set_ylabel("relative viscosity (solid) and velocity (dashed)")
    axs[0].legend(title="E (kJ/mol)", fontsize=7)
    for ax in axs:
        ax.axvline(T0, color="0.5", ls="--", lw=0.7)
        ax.set_xlabel("temperature (K)")
    axs[0].set_title("(a)", loc="left")
    axs[1].set_title("(b)", loc="left")
    fig.tight_layout()
    ctx.save(fig, "figA1")


FIGURES = {"fig01c": fig01c, "fig02": fig02, "fig03": fig03, "fig04": fig04,
           "fig05": fig05, "fig06": fig06, "fig07": fig07, "fig08": fig08,
           "figA1": figA1}


def main(argv=None):
    ap = argparse.ArgumentParser(description="recreate the figures of Grigne & Combes "
                                 "(2020) with pymacma")
    ap.add_argument("--mode", default="quick", choices=["quick", "full"],
                    help="quick: small ensembles and coarse numerics; full: paper sizes")
    ap.add_argument("--figs", nargs="*", default=list(FIGURES),
                    help="subset of: " + " ".join(FIGURES))
    ap.add_argument("--outdir", default="paper_figures")
    ap.add_argument("--data-dir", default=None, help="folder with optional CSV data")
    ap.add_argument("--config", default=None,
                    help="TOML/JSON file with [params] and/or [figures] tables")
    ap.add_argument("--set", nargs="*", default=[], help="model parameters name=value")
    ap.add_argument("--fig-set", nargs="*", default=[],
                    help="figure settings name=value (e.g. n_ens=50 procs=8)")
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--refresh", action="store_true", help="ignore cached results")
    a = ap.parse_args(argv)

    base = Params()
    S = FigSettings.preset(a.mode)
    if a.config:
        base = Params.from_file(a.config)
        if a.config.endswith(".toml"):
            import tomllib
            with open(a.config, "rb") as f:
                cfg = tomllib.load(f)
            S = S.update(cfg.get("figures", {}))
    base = Params.from_dict(dict(s.split("=", 1) for s in a.set), base)
    S = S.update(dict(s.split("=", 1) for s in a.fig_set))
    if a.procs is not None:
        S = S.update({"procs": a.procs})
    unknown = [f for f in a.figs if f not in FIGURES]
    if unknown:
        sys.exit("unknown figure(s): %s" % unknown)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "settings.txt"), "w") as f:
        f.write("mode: %s\n\n[figure settings]\n" % a.mode)
        for k, v in asdict(S).items():
            f.write("%s = %s\n" % (k, v))
        f.write("\n[model parameters]\n" + base.describe() + "\n")
    ctx = Ctx(base, S, a.outdir, a.data_dir, a.refresh)
    print("pymacma paper figures, mode %s, output in %s" % (a.mode, a.outdir))
    if base.describe(only_changed=True):
        print("changed parameters:\n" + base.describe(only_changed=True))
    for name in a.figs:
        t = time.time()
        print("%s ..." % name, flush=True)
        FIGURES[name](ctx)
        print("  %.0f s" % (time.time() - t), flush=True)


if __name__ == "__main__":
    main()
