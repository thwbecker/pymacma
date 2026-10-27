"""
pymacma command-line interface
==============================

  python -m pymacma run      [--seed S] [--T0 K] [--start Ma] [--end Ma] [--snaps 3000,1500,0]
  python -m pymacma fixed    [--ncont N] [--start Ma] [--T0 K]
  python -m pymacma ensemble [--n N] [--procs P]
  python -m pymacma sweep    --param F_lim --values 1e12,2e12,4e12 [--n N] [--calibrate]
  python -m pymacma params   # print all parameters and their values

Model parameters can be changed with a file and/or individual overrides
(later ones win):

  --config my_params.toml          (TOML with a [params] table, or JSON)
  --set E=300e3 F_lim=inf dx=1

Outputs are PNG figures and a pickle of results, named with --out.
The figures of the 2020 paper are made with make_paper_figures.py.

Written by Claude (Anthropic), October 2026, for Thorsten Becker.
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import Params, random_earth, fixed_earth
from . import plots, experiments


def build_params(config=None, sets=()):
    """Params from defaults, then a config file, then name=value overrides"""
    p = Params.from_file(config) if config else Params()
    return Params.from_dict(dict(s.split("=", 1) for s in (sets or [])), p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pymacma", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "fixed", "ensemble", "sweep", "params"])
    ap.add_argument("--seed", type=int, default=0, help="random seed (run)")
    ap.add_argument("--T0", type=float, default=None, help="initial temperature (K)")
    ap.add_argument("--start", type=float, default=None, help="start age (Ma)")
    ap.add_argument("--end", type=float, default=0.0, help="end age (Ma)")
    ap.add_argument("--ncont", type=int, default=None, help="number of continents")
    ap.add_argument("--n", type=int, default=8, help="runs (ensemble, sweep)")
    ap.add_argument("--procs", type=int, default=None, help="processes (default: all)")
    ap.add_argument("--param", default=None, help="parameter to sweep")
    ap.add_argument("--values", default=None, help="comma-separated sweep values")
    ap.add_argument("--calibrate", action="store_true",
                    help="sweep: adjust T0 per value so that T(0) ~ 1625 K")
    ap.add_argument("--snaps", default="", help="comma-separated snapshot ages (Ma)")
    ap.add_argument("--config", default=None, help="parameter file (.toml or .json)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides name=value")
    ap.add_argument("--out", default="pymacma", help="output file prefix")
    a = ap.parse_args(argv)
    p = build_params(a.config, a.set)
    t0 = time.time()

    if a.cmd == "params":
        print(p.describe())
        return

    if a.cmd == "run":
        T0 = 1875.0 if a.T0 is None else a.T0
        start = 3000.0 if a.start is None else a.start
        snaps = [float(s) for s in a.snaps.split(",") if s] or [start, a.end]
        m = random_earth(p, T0=T0, age_Ma=start, n_cont=a.ncont or 4, seed=a.seed)
        h = m.run(a.end, out_every_Myr=1.0, snap_ages=snaps)
        print("T(end) = %.1f K, Q = %.1f TW, %d plates, %d steps, %.0f s"
              % (h["T"][-1], h["Q"][-1], h["n_plates"][-1], h["steps"], time.time() - t0))
        plots.plot_history(h).savefig(a.out + "_history.png", dpi=120)
        for age, s in sorted(h["snaps"].items(), reverse=True):
            f = plots.plot_snapshot(s)
            f.savefig("%s_snap_%07.1fMa.png" % (a.out, age), dpi=110)
            plt.close(f)
        h.pop("snaps")
        with open(a.out + "_run.pkl", "wb") as f:
            pickle.dump(h, f)

    elif a.cmd == "fixed":
        n = a.ncont or 2
        start = 1181.0 if a.start is None else a.start
        T0 = 2000.0 if a.T0 is None else a.T0
        m = fixed_earth(n, params=p, T0=T0, age_Ma=start)
        h = m.run(a.end, out_every_Myr=5.0, snap_ages=(start,))
        print("fixed, %d continents, %d plates: T(end) = %.1f K, %.0f s"
              % (n, len(m.plates), h["T"][-1], time.time() - t0))
        plots.plot_history(h).savefig(a.out + "_fixed_history.png", dpi=120)
        plots.plot_snapshot(h["snaps"][start]).savefig(a.out + "_fixed_snap.png", dpi=110)

    elif a.cmd == "ensemble":
        hists = experiments.run_ensemble(a.n, params=p, processes=a.procs, T0=a.T0,
                                         age_Ma=a.start)
        Tend = np.array([h["T"][-1] for h in hists])
        print("%d runs: present T = %.1f +/- %.1f K (min %.1f, max %.1f), %.0f s"
              % (len(hists), Tend.mean(), Tend.std(), Tend.min(), Tend.max(),
                 time.time() - t0))
        with open(a.out + "_ensemble.pkl", "wb") as f:
            pickle.dump(hists, f)
        plots.plot_ensemble(hists).savefig(a.out + "_ensemble.png", dpi=120)

    elif a.cmd == "sweep":
        if not a.param or not a.values:
            ap.error("sweep needs --param and --values")
        groups, res = {}, {}
        for v in a.values.split(","):
            pv = Params.from_dict({a.param: v}, p)
            T0 = None
            if a.calibrate:
                T0, _ = experiments.calibrate_T0(pv, processes=a.procs)
            hs = experiments.run_ensemble(a.n, pv, processes=a.procs, T0=T0)
            res[v] = (T0, hs)
            groups["%s=%s" % (a.param, v) + ("" if T0 is None else " (T0=%.0f K)" % T0)] = hs
            print("%s = %s: present T = %.1f K" % (a.param, v,
                                                   np.mean([h["T"][-1] for h in hs])))
        with open(a.out + "_sweep.pkl", "wb") as f:
            pickle.dump(res, f)
        plots.plot_vs_temperature(groups).savefig(a.out + "_sweep.png", dpi=120)
        print("sweep done, %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
