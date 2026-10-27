"""Tests of parameter handling, determinism, conservation, command-line
interface and figure script. Written by Claude (Anthropic), October 2026."""
import os
import math

import numpy as np
import pytest

from pymacma.model import (Params, Model, Plate, Ocean, Boundary, random_earth,
                           fixed_earth, SEC_YR)
from pymacma import plots, experiments
from pymacma.__main__ import main as cli
from pymacma import paper_figures as pf

HERE = os.path.dirname(__file__)
CONFIG = os.path.join(HERE, "..", "configs", "reference_2020.toml")


# --------------------------------------------------------------- parameters
def test_params_copy_rejects_unknown_names():
    with pytest.raises(KeyError):
        Params().copy(not_a_parameter=1.0)


def test_params_from_dict_coerces_strings():
    p = Params.from_dict({"F_lim": "inf", "fixed": "true", "si_mode": "constant",
                          "E": "3e5", "H_constant": "none"})
    assert math.isinf(p.F_lim) and p.fixed is True and p.si_mode == "constant"
    assert p.E == 3e5 and p.H_constant is None


def test_reference_config_file_matches_defaults():
    assert Params.from_file(CONFIG) == Params()


def test_config_file_override(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text("[params]\nE = 250e3\nbreakup_requires = \"one\"\n")
    p = Params.from_file(str(f))
    assert p.E == 250e3 and p.breakup_requires == "one" and p.F_lim == Params().F_lim


def test_viscosity_law_matches_fig1c_caption():
    # 2020 Fig. 1c caption: eta_um = 6.7e19 Pa s at 1850 K for E = 300 kJ/mol
    p = Params(eta_um_p=1e21, E=300e3)
    m = Model(p, 0, [Plate([Ocean([0, 360], [0, 0])])], [], 1850, 0)
    assert m.eta(p.eta_um_p) == pytest.approx(6.7e19, rel=0.02)
    assert m.eta(p.eta_um_p, T=p.T_p) == pytest.approx(p.eta_um_p)


# ------------------------------------------------------- model properties
def test_runs_are_deterministic_for_a_seed():
    h1 = random_earth(Params(dx=1.0), seed=7).run(2950, out_every_Myr=5)
    h2 = random_earth(Params(dx=1.0), seed=7).run(2950, out_every_Myr=5)
    assert np.array_equal(h1["T"], h2["T"]) and h1["events"] == h2["events"]


def test_energy_budget_of_one_step():
    m = random_earth(Params(), seed=2)
    T0 = m.T
    dt = m.step()
    p = m.p
    expected = (m.last_H - m.last_Q) * 1e12 * dt * SEC_YR / (p.M_earth * p.C_p)
    assert m.T - T0 == pytest.approx(expected, rel=1e-12)


def test_heat_flow_scales_with_temperature_contrast():
    p = Params()
    pl = [Plate([Ocean(np.linspace(0, 360, 11), np.linspace(1e6, 1e8, 11))])]
    q1 = Model(p, 0, pl, [], 1600.0, 0).heat_flow_TW()
    q2 = Model(p, 0, pl, [], 1900.0, 0).heat_flow_TW()
    assert q2 / q1 == pytest.approx((1900 - p.T_surf) / (1600 - p.T_surf))


def test_mean_slab_thickness_uniform_record():
    m = Model(Params(), 0, [Plate([Ocean([0, 360], [0, 0])])], [], 1625, 0)
    b = Boundary("T", sub=-1, Z=1500e3, H0=90e3)
    b.prof_z, b.prof_h = np.array([1500e3]), np.array([90e3])
    assert m._mean_slab_thickness(b, 90e3) == pytest.approx(90e3)
    m._update_slab_profile(b, 200e3)                 # adds entries at the top
    assert np.all(np.diff(b.prof_z) > 0)
    assert m._mean_slab_thickness(b, 90e3) == pytest.approx(90e3)


def test_fixed_case_start_ages_regression():
    # start ages found by bisection with this code (README); paper: 1150, 660 Ma
    for n, age in ((2, 1181.0), (4, 665.0)):
        T = experiments.fixed_end_T(n, age, Params(dx=1.0, dt_max=2e5))
        assert T == pytest.approx(1625.0, abs=3.0)
    # starting at the paper's ages gives a present temperature within 10 K
    T = experiments.fixed_end_T(2, 1150.0, Params(dx=1.0, dt_max=2e5))
    assert abs(T - 1625.0) < 10.0


@pytest.mark.slow
def test_reference_ensemble_present_temperature():
    hs = experiments.run_ensemble(4, Params(), processes=1, out_every=50.0)
    T = np.array([h["T"][-1] for h in hs])
    assert np.all(np.abs(T - 1625.0) < 25.0)        # paper: 1625 +/- 25 K
    assert all(h["anomalies"] == 0 for h in hs)


# ---------------------------------------------------------------- outputs
def test_plots_run_on_a_short_history():
    import matplotlib
    matplotlib.use("Agg")
    m = random_earth(Params(dx=1.0), seed=3)
    h = m.run(2980, out_every_Myr=1, snap_ages=(2990,))
    plots.plot_history(h)
    plots.plot_snapshot(h["snaps"][2990])
    plots.plot_ensemble([h, h], t_max=3000)


def test_cli_params_and_short_run(tmp_path, capsys):
    cli(["params", "--set", "E=300e3"])
    assert "E                  = 300000.0" in capsys.readouterr().out
    out = str(tmp_path / "r")
    cli(["run", "--seed", "1", "--end", "2990", "--set", "dx=1", "--out", out])
    assert os.path.exists(out + "_history.png") and os.path.exists(out + "_run.pkl")


def test_figure_settings_parsing():
    S = pf.FigSettings.preset("quick").update({"n_ens": "50", "E_values": "250e3,300e3",
                                                "Flim_values": "1e12,inf"})
    assert S.n_ens == 50 and S.E_values == (250e3, 300e3) and math.isinf(S.Flim_values[1])
    with pytest.raises(KeyError):
        S.update({"nonsense": 1})


def test_paper_figure_script_cheap_figures(tmp_path):
    pf.main(["--figs", "fig01c", "fig02", "figA1", "--outdir", str(tmp_path)])
    for name in ("fig01c", "fig02", "figA1"):
        assert os.path.exists(tmp_path / (name + ".png"))
    assert os.path.exists(tmp_path / "settings.txt")


@pytest.mark.parametrize("slope", [1.0, 0.5, 0.08])
def test_calibrate_T0_brackets_and_interpolates(monkeypatch, slope):
    # mock ensemble: present T linear in T0 with the given slope; slope 0.08
    # mimics the weak response of weak-continent runs
    def fake(n, params=None, seeds=None, processes=None, T0=None, **kw):
        return [{"T": np.array([T0, 1600.0 + slope * (T0 - 1875.0)])}]
    monkeypatch.setattr(experiments, "run_ensemble", fake)
    T0, tried = experiments.calibrate_T0(Params(), n_seeds=1, max_evals=10)
    expected = 1875.0 + 25.0 / slope
    if expected <= 2250.0:
        assert T0 == pytest.approx(expected, abs=3.0 / slope + 1e-6)
    else:
        assert T0 == 2250.0          # best value at the end of the allowed range
