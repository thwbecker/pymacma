"""
pymacma.plots
=============

Plotting helpers in the style of the figures of Grigne & Combes (2020).
Building blocks (draw into given axes):

  draw_annulus(ax, snap)            ring view of plates, continents, slabs
  draw_age_heat(ax_age, ax_q, snap) seafloor age and heat flux vs position

Composite figures:

  plot_snapshot(snap)        ring + age + heat flux        (cf. Fig. 4)
  plot_history(h)            T, Q, H, plates, speeds       (cf. Fig. 5)
  plot_ensemble(hists)       percentile bands vs age       (cf. Fig. 6)
  plot_vs_temperature(...)   sweeps binned by temperature  (cf. Figs. 7-8)

The paper-figure script (pymacma.paper_figures) uses these.

Written by Claude (Anthropic), October 2026, for Thorsten Becker.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from .model import SEC_YR

# colours as in Fig. 4 of the paper: moving toward larger angles, smaller
# angles, or immobile
COL_PLUS, COL_MINUS, COL_ZERO = "tab:red", "tab:green", "tab:blue"


def _nanpct(Y, q, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(Y, q, axis=axis)


def n_plates_sphere(nc):
    """equivalent number of plates on a sphere for nc equal plates on a great
    circle, using equal spherical equilateral triangles (Appendix D)"""
    nc = np.asarray(nc, dtype=float)
    out = nc.copy()
    m = nc > 3                      # Eq. D2 valid for nc > 3; else N_S = N_C
    th = 2 * np.pi / nc[m]                                    # Eq. D1
    ca = (np.cos(th) - np.cos(th) ** 2) / np.sin(th) ** 2     # Eq. D2
    al = np.arccos(ca)
    out[m] = 4 * np.pi / (3 * al - np.pi)                     # Eq. D3
    return out


def _vel_color(U, tol=1e-3):
    if U > tol:
        return COL_PLUS
    if U < -tol:
        return COL_MINUS
    return COL_ZERO


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def draw_annulus(ax, snap, title=None, fontsize=9):
    """ring view: plates (coloured by direction of motion), continents
    (grey arcs outside), ridges (triangles) and slabs (blue lines whose
    length is proportional to slab depth, tilted toward the overriding
    plate). Angles increase counterclockwise."""
    p = snap["p"]
    r_out, r_pl, r_in = 1.0, 0.93, 0.55          # surface, plate base, CMB (schematic)
    ax.add_patch(Wedge((0, 0), r_pl, 0, 360, width=r_pl - r_in, fc="#f6c56b",
                       ec="none", alpha=0.5))
    ax.add_patch(Wedge((0, 0), r_in, 0, 360, fc="#f08a4b", ec="none", alpha=0.5))
    for x, L, U in snap["plates"]:
        if L <= 0:
            continue
        ax.add_patch(Wedge((0, 0), r_out, x, x + L, width=r_out - r_pl,
                           fc=_vel_color(U), ec="k", lw=0.5, alpha=0.6))
    for c0, w in snap["conts"]:
        ax.add_patch(Wedge((0, 0), r_out + 0.06, c0, c0 + w, width=0.05,
                           fc="0.35", ec="none"))
    for pos, kind, sub, Z, active in snap["bounds"]:
        th = math.radians(pos)
        c, s = math.cos(th), math.sin(th)
        if kind == "R":
            ax.plot(c * r_out, s * r_out, "^", ms=6, color="k" if active else "0.6",
                    mec="k")
        else:
            zfrac = min(Z / p.d, 1.0)
            r_end = r_out - zfrac * (r_out - r_in) * 0.999
            tilt = -sub * math.radians(4.0)
            ax.plot([c * r_out, math.cos(th + tilt) * r_end],
                    [s * r_out, math.sin(th + tilt) * r_end], "-", color="navy", lw=2.2)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is None:
        title = "%.1f Ma   T = %.1f K   Q = %.1f TW" % (snap["age_Ma"], snap["T"], snap["Q"])
    if title:
        ax.set_title(title, fontsize=fontsize)


def draw_age_heat(ax_age, ax_q, snap, qmax=600.0):
    """seafloor age (and heat flux, if ax_q is not None) versus position;
    continents shaded grey, ridges dotted, trenches solid"""
    p = snap["p"]
    coef = p.k_ocean * (snap["T"] - p.T_surf) / math.sqrt(math.pi * p.kappa)
    for x, a in snap["oceans"]:
        xx = np.mod(x, 360.0)
        cut = np.where(np.diff(xx) < -1)[0]           # split pieces crossing 360
        for sg in np.split(np.arange(len(xx)), cut + 1):
            if len(sg) < 2:
                continue
            ax_age.fill_between(xx[sg], a[sg] / 1e6, color="tab:cyan", alpha=0.6, lw=0)
            ax_age.plot(xx[sg], a[sg] / 1e6, "k-", lw=0.5)
            if ax_q is not None:
                xf = np.linspace(xx[sg][0], xx[sg][-1], max(20, 8 * len(sg)))
                af = np.interp(xf, xx[sg], a[sg])
                q = coef / np.sqrt(np.maximum(af, 1e3) * SEC_YR) * 1e3   # mW/m^2
                ax_q.plot(xf, np.minimum(q, qmax), "r-", lw=0.8)
    for c0, w in snap["conts"]:
        for k in (0, -1):
            a0 = c0 % 360 + 360 * k
            ax_age.axvspan(a0, a0 + w, color="0.6", alpha=0.6, lw=0)
    for pos, kind, sub, Z, active in snap["bounds"]:
        ax_age.axvline(pos, color="k" if kind == "R" else "navy",
                       ls=":" if kind == "R" else "-", lw=0.8)
    ax_age.set_xlim(0, 360)
    ax_age.set_ylim(bottom=0)
    ax_age.set_ylabel("seafloor age (Myr)")
    if ax_q is not None:
        ax_q.set_xlim(0, 360)
        ax_q.set_ylim(0, qmax * 1.03)
        ax_q.set_ylabel("q (mW/m$^2$)")


# ---------------------------------------------------------------------------
# composite figures
# ---------------------------------------------------------------------------
def plot_snapshot(snap, fig=None, title=None):
    """ring view (left), heat flux and seafloor age versus position (right)"""
    if fig is None:
        fig = plt.figure(figsize=(12, 5))
    ax0 = fig.add_axes([0.02, 0.08, 0.38, 0.84])
    ax1 = fig.add_axes([0.47, 0.12, 0.5, 0.38])
    ax2 = fig.add_axes([0.47, 0.56, 0.5, 0.34], sharex=ax1)
    draw_annulus(ax0, snap, title, fontsize=10)
    draw_age_heat(ax1, ax2, snap)
    ax1.set_xlabel("position (deg, counterclockwise)")
    plt.setp(ax2.get_xticklabels(), visible=False)
    return fig


def plot_history(h, fig=None, T_present=(1625.0, 25.0)):
    """temperature, heat flow and radiogenic heating, number of plates and
    ridges, RMS speeds versus age for one run"""
    if fig is None:
        fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    else:
        axs = fig.axes
    age = h["age_Ma"] / 1e3
    axs[0].plot(age, h["T"], "k-")
    if T_present is not None:
        axs[0].axhspan(T_present[0] - T_present[1], T_present[0] + T_present[1],
                       color="tab:blue", alpha=0.2, lw=0)
    axs[0].set_ylabel("T (K)")
    axs[1].plot(age, h["Q"], "k-", lw=0.7, label="oceanic heat flow Q")
    axs[1].plot(age, h["H"], "r-", label="radiogenic H")
    axs[1].set_ylabel("power (TW)")
    axs[1].legend(fontsize=8)
    axs[2].step(age, h["n_plates"], "k-", where="post", label="plates")
    axs[2].step(age, h["n_ridges"], color="tab:orange", where="post", label="ridges")
    axs[2].set_ylabel("number")
    axs[2].legend(fontsize=8)
    mv = h["moving"].astype(bool)
    axs[3].semilogy(age[mv], h["urms"][mv], "k.", ms=2, label="plates (RMS)")
    axs[3].semilogy(age[mv], np.maximum(h["cont_rms"][mv], 1e-2), ".",
                    color="tab:green", ms=2, label="continents (RMS)")
    axs[3].set_ylabel("speed (cm/yr)")
    axs[3].set_xlabel("age (Ga)")
    axs[3].legend(fontsize=8)
    axs[0].invert_xaxis()
    fig.tight_layout()
    return fig


def band(ax, x, Y, color="k", label=None, log=False):
    """median line with 25-75% (dark) and 5-95% (light) bands of rows of Y"""
    q = _nanpct(Y, [5, 25, 50, 75, 95])
    ax.fill_between(x, q[0], q[4], color="0.85", lw=0)
    ax.fill_between(x, q[1], q[3], color="0.6", lw=0)
    ax.plot(x, q[2], color=color, lw=1.2, label=label)
    if log:
        ax.set_yscale("log")


def stack(hists, key, ages, filter_immobile=False):
    """put one diagnostic of each run on a common age axis (Ma); each value
    is the last output at or before the given age. With filter_immobile,
    times with a single immobile plate are set to NaN (as in Fig. 6)."""
    out = np.full((len(hists), len(ages)), np.nan)
    for i, h in enumerate(hists):
        a = h["age_Ma"][::-1]
        y = h[key][::-1].astype(float)
        if filter_immobile:
            y = np.where(h["moving"][::-1], y, np.nan)
        idx = np.clip(np.searchsorted(a, ages), 0, len(a) - 1)
        out[i] = y[idx]
        out[i, (ages < a[0]) | (ages > a[-1])] = np.nan
    return out


def collisions_per_window(hists, window=100.0, t_max=3000.0, kind="collision_cc"):
    """number of events of a kind per age window, summed over runs"""
    edges = np.arange(0, t_max + window, window)
    counts = np.zeros(len(edges) - 1)
    for h in hists:
        ev = np.array([e[0] for e in h["events"] if e[1] == kind])
        if len(ev):
            counts += np.histogram(ev, edges)[0]
    return 0.5 * (edges[1:] + edges[:-1]), counts


def plot_ensemble(hists, fig=None, t_max=3000.0, dt=5.0):
    """quick-look ensemble figure (paper_figures.fig06 has the paper layout)"""
    ages = np.arange(0.0, t_max + dt, dt)
    if fig is None:
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = np.ravel(fig.axes)
    x = ages / 1e3
    band(axs[0], x, stack(hists, "T", ages))
    axs[0].axhspan(1600, 1650, color="tab:blue", alpha=0.15, lw=0)
    axs[0].set_ylabel("T (K)")
    band(axs[1], x, stack(hists, "urms", ages, True), log=True)
    axs[1].set_ylabel("plate speed, RMS (cm/yr)")
    band(axs[2], x, stack(hists, "cont_rms", ages, True), log=True)
    axs[2].set_ylabel("continent speed, RMS (cm/yr)")
    band(axs[3], x, stack(hists, "flux", ages))
    axs[3].set_ylabel("subduction area flux (km$^2$/yr)")
    band(axs[4], x, n_plates_sphere(stack(hists, "n_plates", ages)))
    axs[4].set_ylabel("number of plates (sphere equivalent)")
    c, n = collisions_per_window(hists, t_max=t_max)
    axs[5].plot(c / 1e3, n / max(len(hists), 1), "ko-", ms=3)
    axs[5].set_ylabel("cont. collisions per 100 Myr per run")
    for ax in axs:
        ax.set_xlim(t_max / 1e3, 0)
        ax.set_xlabel("age (Ga)")
    fig.suptitle("%d runs: median, 25-75%% (dark), 5-95%% (light)" % len(hists))
    fig.tight_layout()
    return fig


def binned_by_T(hists, key, Tbins, filter_immobile=False):
    """mean of a diagnostic in temperature bins, pooled over runs"""
    Ts = np.concatenate([h["T"] for h in hists])
    if filter_immobile:
        Y = np.concatenate([np.where(h["moving"], h[key], np.nan) for h in hists])
    else:
        Y = np.concatenate([h[key] for h in hists]).astype(float)
    k = np.digitize(Ts, Tbins) - 1
    out = np.full(len(Tbins) - 1, np.nan)
    for j in range(len(out)):
        sel = (k == j) & np.isfinite(Y)
        if sel.any():
            out[j] = Y[sel].mean()
    return 0.5 * (Tbins[1:] + Tbins[:-1]), out


def plot_vs_temperature(groups, fig=None, Tbins=np.arange(1600, 1910, 10)):
    """for each group (label -> list of runs): median T vs age, RMS plate
    speed and mean number of plates in 10 K temperature bins"""
    if fig is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs = fig.axes
    ages = np.arange(0, 3001, 10.0)
    for lab, hists in groups.items():
        Tm = _nanpct(stack(hists, "T", ages), 50)
        l, = axs[0].plot(ages / 1e3, Tm, label=str(lab))
        tc, mU = binned_by_T(hists, "urms", Tbins, True)
        tc, mN = binned_by_T(hists, "n_plates", Tbins)
        axs[1].semilogy(tc, mU, "o-", ms=3, color=l.get_color())
        axs[2].plot(tc, mN, "o-", ms=3, color=l.get_color())
    axs[0].set_xlim(3, 0)
    axs[0].set_xlabel("age (Ga)")
    axs[0].set_ylabel("median T (K)")
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel("T (K)")
    axs[1].set_ylabel("plate RMS speed (cm/yr)")
    axs[2].set_xlabel("T (K)")
    axs[2].set_ylabel("mean number of plates (circle)")
    fig.tight_layout()
    return fig
