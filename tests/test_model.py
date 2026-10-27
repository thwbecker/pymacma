"""Tests of the model physics, behaviour laws and invariants.
Written by Claude (Anthropic), October 2026."""
import math

import numpy as np
import pytest

from pymacma.model import (Params, Model, Plate, Ocean, Cont, Boundary,
                           random_earth, fixed_earth, earth_plates_test, SEC_YR)
from pymacma.plots import n_plates_sphere

QUIET = dict(tau_sub_noise=0.0)


def ncont(m):
    return sum(isinstance(pc, Cont) for pl in m.plates for pc in pl.pieces)


def kinds(m):
    return "".join(b.kind for b in m.bounds)


# ------------------------------------------------------------ paper numbers
def test_radiogenic_heating_matches_paper():
    m = Model(Params(), 0, [Plate([Ocean([0, 360], [0, 0])])], [], 1625, 0)
    assert m.radiogenic_TW(0) == pytest.approx(13.0, abs=0.1)
    assert m.radiogenic_TW(3000) == pytest.approx(31.2, abs=0.1)


@pytest.mark.parametrize("L,ages,sub,rp,obs", [
    (4550, (30, 50), True, None, 0.696),     # Nazca
    (11970, (70, 180), True, None, 0.651),   # Pacific
    (7300, (70, 180), False, 180, 0.209),    # North America
    (8560, (30, 50), False, 110, 0.109),     # South America
])
def test_present_day_speeds_fig2(L, ages, sub, rp, obs):
    sp = earth_plates_test()
    v = [sp(L, a, Z, sub, rp) for a in ages for Z in (500, 1000, 2000, 2800)]
    assert min(v) / 1.25 <= obs <= max(v) * 1.25


def test_sphere_conversion_appendix_d():
    assert n_plates_sphere([3, 4, 6, 8]) == pytest.approx([3, 8, 22.8, 43.4], abs=0.15)


def test_tau_sub_brittle_scaling():
    p = Params(**QUIET)
    m = Model(p, 0, [Plate([Ocean([0, 360], [0, 0])])], [], 1825, 0)
    # 2020 paper: 140 and 220 Myr give 106 and 167 Myr at T0+200 K
    # (with their T_s = 273 K; the code uses 300 K, hence the tolerance)
    m.p = p.copy(tau_sub_p=140.0, T_surf=273.0)
    assert m.tau_sub() / 1e6 == pytest.approx(106, abs=1)
    m.p = p.copy(tau_sub_p=220.0, T_surf=273.0)
    assert m.tau_sub() / 1e6 == pytest.approx(167, abs=1)


# --------------------------------------------------------------- numerics
def test_heat_flow_linear_age_profile():
    p = Params()
    T = 1625.0
    tau = 100e6
    m = Model(p, 0, [Plate([Ocean(np.linspace(0, 360, 721), np.linspace(0, tau, 721))])],
              [], T, 0)
    coef = p.k_ocean * (T - p.T_surf) / math.sqrt(math.pi * p.kappa)
    exact = coef * m.surf * 2.0 / math.sqrt(tau * SEC_YR) / 1e12
    assert m.heat_flow_TW() == pytest.approx(exact, rel=1e-3)


def test_fixed_case_symmetric_and_resolution_independent():
    res = []
    for dx in (1.0, 0.5):
        m = fixed_earth(2, params=Params(dx=dx), T0=2000, age_Ma=1150)
        m.run(950, out_every_Myr=50, check=True)
        U = np.array([pl.U for pl in m.plates])
        assert abs(U[0]) < 1e-12 * abs(U[1]) and abs(U[3]) < 1e-12 * abs(U[1])  # continents
        assert U[1] == pytest.approx(-U[2]) and U[4] == pytest.approx(-U[5])
        res.append(m.T)
    assert abs(res[0] - res[1]) < 0.5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_runs_keep_invariants(seed):
    m = random_earth(Params(), seed=seed)
    h = m.run(2850, out_every_Myr=5, check=True)
    assert m.anomalies == 0
    assert np.all(np.isfinite(h["T"]))
    assert abs(sum(pl.L for pl in m.plates) - 360) < 1e-6


# ------------------------------------------------------- behaviour laws
def _age(spec):
    """piecewise linear age function from (x, age_Myr) knots"""
    xs, ys = zip(*spec)
    return lambda x: float(np.interp(x, xs, ys)) * 1e6


def test_subduction_initiation_at_passive_margin():
    p = Params(**QUIET, F_lim=np.inf)
    # continent [0,60] on plate P1, old ocean (200 Myr at the margin) to a
    # ridge at 180; P2 from that ridge subducts under P1 at 360
    bounds = [dict(pos=180, kind="R"), dict(pos=0, kind="T", sub=-1, Z=1000e3)]
    m = Model.from_layout(p, bounds, [(0, 60)],
                          _age([(60, 200), (180, 0), (360, 100)]), 1625, 100)
    assert kinds(m) == "RT"
    m.step()
    assert any(e[1] == "si" for e in m.events)
    pos = m.bound_pos()
    new = [k for k, b in enumerate(m.bounds) if b.kind == "T" and abs(pos[k] - 60) < 1]
    assert len(new) == 1
    assert m.bounds[new[0]].sub == 1          # ocean side (higher angle) subducts
    m.check_invariants()


def test_ridge_subduction_creates_staple():
    p = Params(**QUIET, F_lim=np.inf, si_at_staples=False, si_at_continents=False)
    bounds = [dict(pos=100, kind="T", sub=1, Z=2000e3), dict(pos=110, kind="R"),
              dict(pos=0, kind="T", sub=-1, Z=100e3)]
    m = Model.from_layout(p, bounds, [(0, 100)],
                          _age([(100, 5), (110, 0), (360, 60)]), 1625, 100)
    nb = len(m.bounds)
    h = m.run(50, out_every_Myr=1, check=True)
    assert any(e[1] == "vanish" for e in h["events"])
    assert len(m.bounds) <= nb - 2 + sum(e[1] == "backarc" for e in h["events"])


def test_continent_collision_merges_continents():
    p = Params(**QUIET, F_lim=np.inf, si_at_staples=False, si_at_continents=False)
    bounds = [dict(pos=100, kind="T", sub=1, Z=2000e3), dict(pos=200, kind="R"),
              dict(pos=0, kind="T", sub=-1, Z=2000e3)]
    m = Model.from_layout(p, bounds, [(40, 60), (150, 50)],
                          _age([(100, 80), (150, 80), (200, 0), (360, 100)]), 1625, 300)
    assert ncont(m) == 2
    h = m.run(0, out_every_Myr=1, check=True)
    ev = [e[1] for e in h["events"]]
    assert "collision_cc" in ev
    assert ncont(m) == 1


def test_backarc_ridge_when_upper_plate_pulled_away():
    p = Params(**QUIET, F_lim=np.inf, si_at_staples=False, si_at_continents=False)
    # Pa subducts under Pb at 180; Pb itself subducts under Pc at 270 with a
    # deep slab, so Pb is pulled away from the trench at 180
    bounds = [dict(pos=180, kind="T", sub=-1, Z=300e3),
              dict(pos=270, kind="T", sub=-1, Z=2500e3),
              dict(pos=0, kind="R")]
    m = Model.from_layout(p, bounds, [(290, 60)],
                          _age([(0, 0), (180, 60), (270, 60), (290, 60), (360, 0)]), 1625, 100)
    nr = kinds(m).count("R")
    m.step()
    assert any(e[1] == "backarc" for e in m.events)
    assert kinds(m).count("R") == nr + 1
    m.check_invariants()
    m.run(90, out_every_Myr=1, check=True)


def test_continental_breakup_between_two_trenches():
    p = Params(**QUIET, F_lim=1e11, si_at_staples=False, si_at_continents=False)
    bounds = [dict(pos=30, kind="T", sub=-1, Z=1000e3),
              dict(pos=150, kind="T", sub=1, Z=1000e3),
              dict(pos=250, kind="R")]
    m = Model.from_layout(p, bounds, [(30, 120)],
                          _age([(0, 70), (30, 100), (150, 100), (250, 0), (360, 50)]),
                          1625, 100, cont_life=500e6)
    m.step()
    assert any(e[1] == "breakup" for e in m.events)
    assert ncont(m) == 2
    m.check_invariants()


def test_breakup_rule_requires_both_trenches_by_default():
    base = dict(**QUIET, F_lim=1e11, si_at_staples=False, si_at_continents=False)
    bounds = [dict(pos=30, kind="T", sub=-1, Z=1000e3), dict(pos=200, kind="R")]
    lay = (bounds, [(30, 120)], _age([(0, 70), (30, 100), (150, 100), (200, 0), (360, 50)]))
    m = Model.from_layout(Params(**base), *lay, 1625, 100, cont_life=500e6)
    m.step()
    assert not any(e[1] == "breakup" for e in m.events)
    m = Model.from_layout(Params(**base, breakup_requires="one"), *lay, 1625, 100,
                          cont_life=500e6)
    m.step()
    assert any(e[1] == "breakup" for e in m.events)


def test_subcontinental_force_matches_fig1c():
    # Fig. 1c: present day, eta_um = 1e21, H = 13 TW: plateau roughly 5 TN/m
    # for a = 8000 km and 2.5 TN/m for a = 4000 km (values read by eye; MACMA
    # normalises heating by mantle mass, which gives ~15% lower values than
    # a per-volume estimate, hence the loose tolerance)
    p = Params(eta_um_p=1e21, H_constant=13.0)
    m = Model(p, 0, [Plate([Ocean([0, 360], [0, 0])])], [], p.T_p, 0)
    for a_km, F in ((8000, 5.2), (4000, 2.6)):
        c = Cont(a_km * 1e3 / m.deg2m, life=1e10)
        assert m.cont_force(c) / 1e12 == pytest.approx(F, rel=0.2)
