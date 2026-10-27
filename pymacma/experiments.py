"""
pymacma.experiments
===================

Ensembles, calibration and caching used by the command-line interface and
the paper-figure script.

  run_one(seed, params, ...)          one evolving-boundary run
  run_ensemble(n, params, ...)        many seeds, optionally in parallel
  calibrate_T0(params, ...)           initial temperature giving ~1625 K today
  fixed_start_age(n_cont, params,...) start age of a fixed-boundary run
  cached(path, fn)                    compute once, then reload from pickle

Written by Claude (Anthropic), October 2026, for Thorsten Becker.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .model import Params, random_earth, fixed_earth


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------
def params_key(*objs):
    """short hash of parameter objects / plain values, for cache file names"""
    def enc(o):
        if isinstance(o, Params):
            return o.to_dict()
        return o
    s = json.dumps([enc(o) for o in objs], sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def cached(path, fn, refresh=False):
    """return pickle at path if it exists (and not refresh), else fn() and save"""
    if path and os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    res = fn()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(res, f)
    return res


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
def run_one(seed, params=None, T0=None, age_Ma=None, end_Ma=0.0, out_every=2.5,
            n_cont=4, snap_ages=(), keep_snaps=False):
    """one evolving-boundary run with random initial geometry.
    T0 (K) and age_Ma default to the 2020 reference set-up, 1875 +/- 5 K at
    3000 +/- 50 Ma (drawn from the seed). A scalar T0 is used as the centre
    of a +/- 5 K interval, as in the paper."""
    rng = np.random.default_rng(10_000 + seed)
    T0 = rng.uniform(1870, 1880) if T0 is None else T0 + rng.uniform(-5, 5)
    age_Ma = rng.uniform(2950, 3050) if age_Ma is None else age_Ma
    m = random_earth(params or Params(), T0=T0, age_Ma=age_Ma, n_cont=n_cont, seed=seed)
    h = m.run(end_Ma, out_every_Myr=out_every, snap_ages=snap_ages)
    h["seed"], h["T0"], h["start_Ma"] = seed, T0, age_Ma
    h["anomalies"] = m.anomalies
    if not keep_snaps:
        h.pop("snaps", None)
    return h


def _job(args):
    seed, kw = args
    return run_one(seed, **kw)


def run_ensemble(n, params=None, seeds=None, processes=None, **kw):
    """run seeds 0..n-1 (or the given seeds), in parallel when processes > 1
    (default: all cores). Extra keywords go to run_one."""
    seeds = list(range(n)) if seeds is None else list(seeds)
    jobs = [(s, dict(params=params, **kw)) for s in seeds]
    processes = processes or os.cpu_count() or 1
    if processes > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(processes, len(jobs))) as ex:
            return list(ex.map(_job, jobs))
    return [_job(j) for j in jobs]


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def calibrate_T0(params=None, target=1625.0, n_seeds=4, T0_first=1875.0,
                 slope_guess=0.5, tol=3.0, max_evals=7, T0_range=(1650.0, 2250.0),
                 processes=None, seed0=1000, verbose=False):
    """initial temperature (centre of the +/- 5 K interval) for which the
    mean present-day temperature of n_seeds runs is close to `target`, as
    done for each parameter value in 2020 section 3.4.

    The present temperature usually increases with the initial one, but the
    response can be weak (efficient cooling regulates the temperature) and
    each evaluation carries a sampling error of roughly std/sqrt(n_seeds)
    because runs are chaotic. The search therefore first brackets the target
    (steps of increasing size from T0_first), then interpolates linearly
    inside the bracket; it stops when within tol K or after max_evals
    evaluations and returns the best value found. Calibration seeds
    (seed0, ...) differ from the ensemble seeds.
    Returns (T0, list of (T0_tried, mean_T_end))."""
    seeds = list(range(seed0, seed0 + n_seeds))
    lo_lim, hi_lim = T0_range

    def mean_end(T0):
        hs = run_ensemble(0, params, seeds=seeds, processes=processes, T0=T0,
                          out_every=50.0)
        return float(np.mean([h["T"][-1] for h in hs]))

    tried = []

    def ev(T0):
        T0 = float(np.clip(T0, lo_lim, hi_lim))
        E = mean_end(T0)
        tried.append((T0, E))
        return T0, E

    a, Ea = ev(T0_first)
    b, Eb = a, Ea
    step = float(np.clip((target - Ea) / slope_guess, -150, 150))
    # 1. bracket: walk in the direction that reduces the error
    while len(tried) < max_evals and abs(Eb - target) > tol:
        if (Ea - target) * (Eb - target) < 0:
            break
        nb = b + step
        if nb <= lo_lim or nb >= hi_lim:
            if b in (lo_lim, hi_lim):
                break
        a, Ea = b, Eb
        b, Eb = ev(nb)
        step *= 1.6
    # 2. interpolate inside the bracket (regula falsi)
    while (len(tried) < max_evals and (Ea - target) * (Eb - target) < 0
           and abs(Eb - target) > tol):
        c = a + (target - Ea) * (b - a) / (Eb - Ea)
        c, Ec = ev(c)
        if (Ec - target) * (Ea - target) < 0:
            b, Eb = c, Ec
        else:
            a, Ea = c, Ec
    best = min(tried, key=lambda t: abs(t[1] - target))
    if verbose:
        print("calibrate_T0:", ", ".join("%.1f->%.1f" % t for t in tried))
    return best[0], tried


def fixed_end_T(n_cont, start_Ma, params=None, T0=2000.0, end_Ma=0.0):
    m = fixed_earth(n_cont, params=params, T0=T0, age_Ma=start_Ma)
    return m.run(end_Ma, out_every_Myr=50.0)["T"][-1]


def fixed_start_age(n_cont, params=None, target=1625.0, T0=2000.0, guess=None,
                    tol=5.0):
    """start age (Ma) for which a fixed-boundary run starting at T0 reaches
    `target` K today (2020 section 3.1). Bisection; the bracket starts around
    `guess` (Ma) and is widened until it contains the solution."""
    g = guess or 1000.0
    lo, hi = 0.6 * g, 1.5 * g
    while fixed_end_T(n_cont, lo, params, T0) < target and lo > 20:
        lo *= 0.6                  # still too cold: start later
    while fixed_end_T(n_cont, hi, params, T0) > target and hi < 4400:
        hi *= 1.4                  # still too hot: start earlier
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if fixed_end_T(n_cont, mid, params, T0) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
