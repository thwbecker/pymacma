# pymacma

A compact Python re-implementation of MACMA (MultiAgent Convective MAntle),
a model of plate tectonics and Earth's thermal history in which rigid plates
move along a great circle, plate velocities follow from a force balance, and
plate boundaries are created and destroyed by simple behaviour laws. The
package also contains a script that recreates the model figures of Grigné &
Combes (2020) and a test suite.

## Provenance and credit

**Original model and code.** MACMA was developed by M. Combes, C. Grigné and
co-workers and is described in:

- Combes, M., Grigné, C., Husson, L., Conrad, C. P., Le Yaouanq, S.,
  Parenthoën, M., Tisseau, C., & Tisseau, J. (2012). Multiagent simulation of
  evolutive plate tectonics applied to the thermal evolution of the Earth.
  *Geochemistry, Geophysics, Geosystems*, 13, Q05006.
  https://doi.org/10.1029/2011GC004014
- Grigné, C., & Combes, M. (2020). Thermal history of the Earth: On the
  importance of surface processes and the size of tectonic plates.
  *Geochemistry, Geophysics, Geosystems*, 21, e2020GC009123.
  https://doi.org/10.1029/2020GC009123

The original C++/Qt implementation by C. Grigné is available at
https://github.com/cecgri/MACMA (release v0.1-beta, archived as
https://doi.org/10.5281/zenodo.3946077), under the GNU General Public License
v3.0 or later.

**This implementation.** The Python code, tests, figure script and this
README were written by Claude (Anthropic, model Claude Opus 5.5) in October
2026, at the request and under the direction of Thorsten Becker (University
of Texas at Austin), who also reviewed intermediate results. The code was
written from the two papers and from reading the released C++ source; no C++
code was translated line by line, but the rules and formulas follow that
source closely where it differs from the papers (see below).

**Repository.** https://github.com/thwbecker/pymacma

**Status.** This is not an official release of MACMA and has not been
reviewed by the original authors. It reproduces several published results
(see "Checks against the paper"), but results that matter should be checked
against the original code.

**License.** Because this package follows the logic of the GPL-licensed
original closely, it may be prudent to treat it as a derivative work and
distribute it under GPL-3.0-or-later as well.

## Contents

```
pymacma/model.py          model: parameters, state, force balance, transport,
                          behaviour laws, initial conditions
pymacma/plots.py          plotting building blocks and quick-look figures
pymacma/experiments.py    ensembles, calibration of initial temperature, caching
pymacma/paper_figures.py  recreation of the figures of Grigné & Combes (2020)
pymacma/__main__.py       command-line interface (python -m pymacma ...)
make_paper_figures.py     wrapper for paper_figures
configs/reference_2020.toml   all parameters with units and comments
tests/                    test suite (pytest)
paper_figures_quick/      figures made with --mode quick (see below)
```

Requirements: Python 3.11 or later, numpy, matplotlib; pytest for the tests
(`pip install -r requirements.txt`).
No installation step is needed; run commands from this directory.

## Quick start

```
python -m pymacma run --seed 1 --snaps 3000,1500,0 --out ref   # one run, 3 Ga to present
python -m pymacma params                                       # list all parameters
python -m pymacma run --set E=300e3 F_lim=4e12 --out test      # change parameters
python -m pymacma run --config my_params.toml                  # parameters from a file
python -m pymacma fixed --ncont 2                              # fixed plate boundaries
python -m pymacma ensemble --n 16 --procs 8 --out ens          # many seeds
python -m pymacma sweep --param tau_sub_p --values 140,180,220 --n 8 --calibrate
python make_paper_figures.py --mode quick                      # all paper figures
python -m pytest -m "not slow"                                 # tests (under a minute)
```

From Python:

```python
from pymacma.model import Params, random_earth
from pymacma import plots

p = Params(F_lim=3e12, E=300e3)          # or Params.from_file("my.toml")
m = random_earth(p, T0=1875, age_Ma=3000, seed=4)
h = m.run(0.0, out_every_Myr=1.0, snap_ages=(1400,))
plots.plot_history(h)
plots.plot_snapshot(h["snaps"][1400])
```

`run` returns a dict of arrays (`age_Ma`, `T`, `Q`, `H`, `n_plates`,
`n_ridges`, `n_trenches`, `urms`, `cont_rms`, `flux`, `moving`, ...) plus
`events`, a list of `(age_Ma, kind)` with kinds `si` (subduction
initiation), `si_abort` (a new trench that diverged and was removed),
`backarc`, `breakup`, `vanish` (a plate disappeared, for example by ridge
subduction), `collision_cc` (two continents) and `collision_ct` (a continent
reaching a trench without continent on the other side).

## Changing parameters

All model inputs are fields of the `Params` dataclass in `pymacma/model.py`,
each with its unit and source in a comment. The file
`configs/reference_2020.toml` lists all of them with the same comments; copy
it, edit values, and pass it with `--config`. Individual values can be
overridden on the command line with `--set name=value` (applied after the
config file), or in Python with `Params(name=value)` or
`p.copy(name=value)`. Unknown names raise an error. `inf` is accepted (for
example `F_lim=inf` switches off continental breakup).

Settings of the figure script (ensemble sizes, sweep values, numerics,
number of processes) are fields of `FigSettings` in
`pymacma/paper_figures.py`. They can be changed with `--fig-set name=value`
or in a `[figures]` table of the config file, for example
`--fig-set n_ens=50 E_values=250e3,350e3,450e3 procs=16`.

## Recreating the figures of the paper

```
python make_paper_figures.py --mode quick                  # all figures
python make_paper_figures.py --mode full --procs 32        # paper ensemble sizes
python make_paper_figures.py --figs fig03 fig06 --outdir figs
```

| Output | Paper figure | Content |
|---|---|---|
| `fig01c` | 1c | subcontinental shear force versus continent age |
| `fig02` | 2 | present-day plate speeds from the force balance |
| `fig03` | 3 | fixed boundaries: configurations, temperature, Nu-Ra |
| `fig04` | 4 | plate reorganization over about 40 Myr |
| `fig05` | 5 | temperature, heat flow and ridges over 3 Gyr for one run |
| `fig06` | 6 | ensemble statistics over 3 Gyr |
| `fig07` | 7 | effect of activation energy and reference viscosity |
| `fig08` | 8 | effect of continental strength and critical age |
| `figA1` | A1 | convective forces, velocity and viscosity versus temperature |

Modes:

- `quick` (default): 16 runs for Fig. 6, 4 runs per parameter value for
  Figs. 7-8, initial temperatures calibrated on 2 runs, coarse numerics for
  ensembles (dx = 1°, dt ≤ 200 kyr). On one core this takes about an hour.
  The coarse numerics appear to shift present-day temperatures by several K
  (see below), so these figures are best used for exploration.
- `full`: the ensemble sizes of the paper (200 runs for Fig. 6, 50 per value
  for Figs. 7-8), calibration on 8 runs, default numerics. This is of the
  order of 100-200 CPU hours and is meant for a multi-core machine
  (`--procs`).

Model results are cached in `<outdir>/cache`, keyed by all parameters and
settings, so figures can be restyled and regenerated quickly; `--refresh`
forces recomputation. `<outdir>/settings.txt` records the settings used.

Notes on individual figures:

- Evolving-boundary runs are chaotic, so Figs. 4 and 5 show a reorganization
  chosen automatically in one run (`ref_seed`, `ref_window`) and cannot match
  the particular events shown in the paper. Figs. 6-8 are the meaningful
  comparison.
- Fig. 2: the paper restricts each curve to the observed range of seafloor
  ages at the trench. The ranges highlighted here (`FIG2_AGE_RANGES`) are
  approximate estimates made for this script, not values from the paper.
  The observed speeds are those printed in the paper's Fig. 2 (from Argus et
  al., 2011).
- Figs. 7-8: for each parameter value the initial temperature is adjusted so
  that the mean present-day temperature of a few calibration runs is close
  to 1625 K, as described in section 3.4 of the paper (the paper does not
  describe its procedure; here the target is first bracketed and then found
  by linear interpolation, see `experiments.calibrate_T0`). Because runs are
  chaotic, each calibration evaluation has a sampling error; with the quick
  settings (2 calibration runs, 4 runs per value) the ensemble means end
  within about 25 K of 1625 K. The thin lines are fixed-boundary runs with
  six plates, whose start age is found by bisection.

**Observational data.** The paper compares model results with petrological
temperatures (Herzberg et al., 2010) and with plate reconstructions. These
data are not included here. If CSV files with a header line are placed in a
folder given with `--data-dir`, they are overlaid:

| File | Columns | Used in |
|---|---|---|
| `herzberg2010.csv` | `age_Ga,T_K` | Figs. 3e, 6a, 7a,d, 8a,d |
| `plate_speed.csv` | `age_Ga,speed_cm_yr,label` | Fig. 6b |
| `continent_speed.csv` | `age_Ga,speed_cm_yr,label` | Fig. 6c |
| `subduction_flux.csv` | `age_Ga,flux_km2_yr,label` | Fig. 6d |
| `n_plates.csv` | `age_Ga,n_plates,label` | Fig. 6e |
| `collisions.csv` | `age_Ga,count,label` | Fig. 6f |

## Model summary

Plates are rigid and move along a circle of Earth's radius. Each plate
velocity follows from a balance of driving forces (ridge push, slab pull,
slab suction) and resisting forces (mantle drag, viscous shear along slabs,
bending at trenches), with Arrhenius viscosities that depend on a uniform
mantle temperature (Eq. 8 of the 2020 paper). Seafloor ages are carried on
Lagrangian nodes; new seafloor forms at ridges and is consumed at trenches.
Oceanic heat flow follows half-space cooling and is integrated exactly for
ages that vary linearly between nodes. Continents are perfectly insulating.
The mantle temperature evolves from M c_p dT/dt = H(t) − Q, with radiogenic
heating of a mantle depleted by a constant continental crust.

Behaviour laws: subduction starts at staples and passive margins when the
seafloor is older than a critical age; a ridge opens behind a trench when
the upper plate is pulled away (back-arc); a continent breaks up when a
subcontinental shear force exceeds a strength F_lim; and collisions remove
boundaries (ridge or trench subduction) or stop subduction (continent at a
trench, merging continents).

## Differences between this code, the original code and the 2020 paper

### Where the original code differs from the paper, the code is followed

The released code produced the published figures, and its example
configuration `doc/configs/config.macma` matches the 2020 reference case.
Its values are the defaults here.

| Item | 2020 paper | Original code, used here |
|---|---|---|
| Mantle drag | Eq. B5, lower-mantle thickness 2230 km | factor 2 (shear over half the layer) and lower-mantle thickness d/2 − D = 780 km; speeds about 2 times lower than with Eq. B5 as printed |
| Slab suction | prefactor 1/(4 d_lm) | prefactor 2/9 times Z_lm/d_lm |
| Ridge push | α = 2×10⁻⁵ K⁻¹, maximum age of the plate | α_pl = 3.5×10⁻⁵ K⁻¹, temperature-dependent plate density, age at the far end of the oceanic section next to the ridge (sections end at staples and continents) |
| Slab pull | thickness from the maximum age | mean thickness recorded along the slab in the upper mantle (minimum 10 km), temperature-dependent density contrast |
| Critical age | Eq. 11 | Eq. 11 plus uniform noise of ±20 Myr drawn at each check (`tau_sub_noise`) |
| New slabs | not specified | initial depth 0.1 times the plate thickness; initiation only if the driving force on the future subducting plate points toward the new trench |
| Breakup condition | at least one subduction zone bounds the plate | subduction zones at both ends of the plate (`breakup_requires = "both"`; `"one"` is available) |
| Subcontinental heating | per unit volume (Eq. C1) | per unit mass of mantle |
| Constants | T_s = 273 K, R = 6370 km | T_s = 300 K, R = 6371 km |
| Numerics | resolution 0.25°, dt ≤ 5 kyr | resolution 0.5°, dt ≤ 50 kyr |

Other rules taken from the code: a ridge whose plates converge becomes
inactive for one step (its two plates move together); a trench whose plates
diverge is replaced by a staple; when continents collide, their
subcontinental warming is averaged by width and converted to an equivalent
warming time; slabs lengthen at the convergence rate down to 2800 km; no
plate motion occurs when there is no subduction zone.

### Choices made here that are not in the papers or the code

- Representation: each plate is a list of oceanic and continental pieces
  rather than a set of agents. Positions, collisions and event times are
  computed directly from this representation.
- New back-arc ridges, breakup ridges and slivers start with zero width (the
  code uses 0.01°).
- The time step is shortened so that a plate vanishing or a continent
  reaching a trench happens exactly at the end of a step. The code also
  shortens steps when staples reach trenches; here staples are subducted
  within a step.
- The random initial conditions (`random_earth`) approximate the description
  in section 3.3 of the paper; the generator used there is not part of the
  released code. Four continents of random widths cover 40% of the surface;
  each ocean basin has a ridge at a random position and trenches at a random
  subset of continental edges; seafloor ages grow linearly from ridges at
  random spreading rates (2-8 cm/yr) and are capped at 0.95 times the
  critical age; initial slab depths are random between 200 and 1500 km.
- RMS speeds are reported both unweighted (`urms`, `cont_rms`) and weighted
  by length (`urmsL`, `cont_rmsL`).
- Not implemented: small-scale convection (off in the 2020 reference case),
  continental growth, core cooling, and subduction initiation within upper
  plates (options of the original code).

## Checks against the paper

| Quantity | This code | 2020 paper |
|---|---|---|
| Radiogenic heating today / at 3 Ga | 12.97 / 31.21 TW | 13.0 / 31.2 TW |
| Fixed boundaries, start age for 1625 K today (1 / 2 / 4 / 8 continents) | 2582 / 1181 / 665 / 396 Ma | 2550 / 1150 / 660 / 400 Ma |
| Fixed boundaries, slope of Nu versus Ra | 0.35-0.36 | close to 1/3 |
| Present-day temperature of runs from 1875 ± 5 K at 3 Ga | 1623.8 ± 2.5 K (8 runs) | all 200 runs within 1625 ± 25 K |
| Present-day number of plates | 3-9 in the 8 runs | quartiles 4, 6, 8 |

Plate speeds for the sizes and trench ages of the paper's Fig. 2 bracket the
observed values for the Nazca, Pacific, North American and South American
plates; Juan de Fuca comes out faster than in the paper's figure.

Results of `--mode quick` (figures in `paper_figures_quick/`; the model
settings are recorded in `paper_figures_quick/settings.txt`):

- Fig. 3: start ages 2579, 1183, 665 and 392 Ma (tolerance 20 Myr) and Nu-Ra
  slopes of 0.35.
- Fig. 6 (16 runs, coarse numerics): present-day temperature 1617.5 ± 7.7 K;
  median plate RMS speed decreasing from roughly 60-100 cm/yr at 3 Ga to
  about 8 cm/yr today; number of plates increasing toward the present; the
  number of continental collisions per 100 Myr increases with time with a
  regression slope of 1.6 per Ga over the last 2.5 Gyr (paper: 1.7 per Ga for
  its model and 1.8 per Ga for Condie et al., 2015).
- Fig. 7: plate speed versus temperature depends strongly on the activation
  energy and the viscosity, and stays close to the fixed-boundary speeds; a
  higher viscosity gives more plates; the thermal evolutions differ less.
- Fig. 8: plate speed versus temperature does not depend on F_lim or
  τ_subd0, while weaker continents or a younger critical age give more
  plates and require a hotter start (about 1950 K for F_lim = 1 TN/m and
  τ_subd0 = 140 Myr, about 1820-1850 K for unbreakable continents and
  τ_subd0 = 220 Myr), in line with Fig. 8 of the paper.

These are qualitative comparisons by eye; a quantitative comparison needs the
full mode and the paper's data.

Numerical checks: for fixed boundaries, the present-day temperature changes
by 0.2 K between dx = 1° and dx = 0.125°. For evolving boundaries, the same
8 seeds give 1623.8 ± 2.5 K with the default numerics and 1616.6 ± 6.5 K with
dx = 1° and dt ≤ 200 kyr, which suggests a bias of several K for the coarse
setting. Thirteen 3-Gyr runs covering the parameter ranges of Figs. 7-8 and
the alternative behaviour laws completed with all consistency checks
passing.

## Tests

```
python -m pytest -m "not slow"     # under a minute
python -m pytest                   # also a 4-run reference ensemble (~2 min)
```

`tests/test_model.py` checks physics against numbers in the paper (radiogenic
heating, Fig. 2 speeds, Appendix D conversion, critical ages, Fig. 1c
forces), exact heat-flow integration, symmetry and resolution independence of
fixed-boundary runs, consistency of the state during random runs, and each
behaviour law in a constructed configuration (subduction initiation at a
passive margin, ridge subduction, continental collision, back-arc opening,
continental breakup and its trench condition). `tests/test_interface.py`
checks parameter handling and config files, determinism, the energy budget
of a step, regression values of fixed-boundary start ages, plotting, the
command-line interface, the figure script, and the temperature calibration
(with a mocked ensemble, including a weak response).

## Performance

On one core: a 3-Gyr evolving run takes roughly 45-85 s with the default
numerics and about 13 s with dx = 1°, dt ≤ 200 kyr. Fixed-boundary runs
take seconds to a minute. Ensembles run in parallel over seeds.

## References

Model and code:

- Combes et al. (2012) and Grigné & Combes (2020), see above.
- Grigné, C. (2020). cecgri/MACMA: Pre-release (v0.1-beta) [software].
  Zenodo. https://doi.org/10.5281/zenodo.3946077

Sources of formulas and parameter values used in the code (as cited in the
papers):

- Argus, D. F., Gordon, R. G., & DeMets, C. (2011). Geologically current
  motion of 56 plates relative to the no-net-rotation reference frame.
  *Geochemistry, Geophysics, Geosystems*, 12, Q11001.
  https://doi.org/10.1029/2011GC003751
- Buffett, B. A. (2006). Plate force due to bending at subduction zones.
  *Journal of Geophysical Research*, 111, B09405.
  https://doi.org/10.1029/2006JB004295
- Conrad, C. P., & Lithgow-Bertelloni, C. (2002). How mantle slabs drive
  plate tectonics. *Science*, 298, 207-209.
- Herzberg, C., & Asimow, P. D. (2015). PRIMELT3 MEGA.XLSM software for
  primary magma calculation: Peridotite primary magma MgO contents from the
  liquidus to the solidus. *Geochemistry, Geophysics, Geosystems*, 16,
  563-578. https://doi.org/10.1002/2014GC005631
- Herzberg, C., Condie, K., & Korenaga, J. (2010). Thermal history of the
  Earth and its petrological expression. *Earth and Planetary Science
  Letters*, 292. https://doi.org/10.1016/j.epsl.2010.01.022
- McDonough, W. F., & Sun, S.-s. (1995). The composition of the Earth.
  *Chemical Geology*, 120, 223-253.
- Müller, R. D., Sdrolias, M., Gaina, C., & Roest, W. R. (2008). Age,
  spreading rates, and spreading asymmetry of the world's ocean crust.
  *Geochemistry, Geophysics, Geosystems*, 9(4).
  https://doi.org/10.1029/2007GC001743
- Parsons, B., & Richter, F. M. (1980). A relation between the driving
  force and geoid anomaly associated with mid-ocean ridges. *Earth and
  Planetary Science Letters*, 51, 445-450.
- Rudnick, R. L., & Gao, S. (2003). Composition of the continental crust.
  In R. L. Rudnick (Ed.), *Treatise on Geochemistry* (Vol. 3, pp. 1-64).
  Pergamon.
- Turcotte, D. L., & Schubert, G. (2002). *Geodynamics* (2nd ed.).
  Cambridge University Press.

Related later work that may be useful for comparisons:

- Grigné, C. (2025). Evolution of convective stresses in stagnant-lid
  planets. *Journal of Geophysical Research: Planets*.
  https://doi.org/10.1029/2024JE008817
- Merdith, A. S., et al. (2021). Extending full-plate tectonic models into
  deep time: Linking the Neoproterozoic and the Phanerozoic. *Earth-Science
  Reviews*, 214, 103477.
- Cao, X., et al. (2024). Earth's tectonic and plate boundary evolution over
  1.8 billion years. *Geoscience Frontiers*. Reports RMS plate speeds of
  4-7 cm/yr between 1.8 and 1.0 Ga, which can be compared with Fig. 6b.
