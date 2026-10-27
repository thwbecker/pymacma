"""
pymacma.model
=============

A compact re-implementation of the MACMA ("MultiAgent Convective MAntle")
model of plate tectonics and Earth's thermal history:

  Combes, M., Grigne, C., Husson, L., Conrad, C. P., Le Yaouanq, S.,
  Parenthoen, M., Tisseau, C., & Tisseau, J. (2012). Multiagent simulation of
  evolutive plate tectonics applied to the thermal evolution of the Earth.
  Geochem. Geophys. Geosyst., 13, Q05006. doi:10.1029/2011GC004014

  Grigne, C., & Combes, M. (2020). Thermal history of the Earth: On the
  importance of surface processes and the size of tectonic plates.
  Geochem. Geophys. Geosyst., 21, e2020GC009123. doi:10.1029/2020GC009123

  Original C++/Qt code: C. Grigne, https://github.com/cecgri/MACMA
  (v0.1-beta, doi:10.5281/zenodo.3946077, GPL-2.0-or-later)

Where the papers and the released C++ code differ, this module follows the
code (see README.md for the list). Equation numbers in comments refer to the
2020 paper unless marked "C12" (Combes et al., 2012) or "code" (the C++ file
in which the rule is implemented).

Python implementation: Claude (Anthropic), October 2026, prepared at the
request of Thorsten Becker.

Conventions
-----------
* The surface is a great circle; positions are angles in degrees, increasing
  counterclockwise ("+" direction). Velocities are stored in deg/yr, and a
  positive velocity moves material toward larger angles.
* Plates are stored in angular order. ``bounds[k]`` is the plate boundary
  between ``plates[k]`` (its lower-angle side) and ``plates[k+1]`` (its
  upper-angle side), indices modulo the number of plates. A state with no
  boundary at all is a single unbounded plate covering the circle.
* Each plate is a list of "pieces" that tile it from its low-angle end to its
  high-angle end:
    - ``Ocean``: Lagrangian nodes ``s`` (deg, relative to the piece start,
      ascending, s[0] = 0) with seafloor ages ``a`` (yr). Ages vary linearly
      between nodes.
    - ``Cont``: a continent of width ``w`` (deg) with a warming clock
      ``life`` (yr) used for continental breakup.
  Two adjacent Ocean pieces are separated by a "staple" (an age
  discontinuity). An Ocean/Cont contact inside a plate is a passive margin.
* Trench polarity ``sub``: -1 if the lower-angle plate subducts under the
  upper-angle plate, +1 if the upper-angle plate subducts.
* Units are SI inside force and heat computations; times are kept in years
  and ages in years (Ma/Myr only at the interface).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, replace, asdict

import numpy as np

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SEC_YR = 365.0 * 24.0 * 3600.0       # seconds per year (365-day year, as in MACMA)
LN2 = math.log(2.0)
THICK_COEFF = 2.0 * math.sqrt(math.log(3.0))  # H = c sqrt(kappa t), c ~ 2.1 (Eq. 3)
COEFF_T_PLATE = 0.486065             # mean plate temperature factor (code: earth.cpp)
R_GAS = 8.314                        # gas constant (J/mol/K)
EPS_W = 1e-9                         # width below which an ocean piece counts as empty (deg)


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
@dataclass
class Params:
    """All model inputs. Defaults reproduce the 2020 reference case as set in
    the MACMA repository file doc/configs/config.macma.

    Change parameters with ``Params(E=300e3)``, ``p.copy(F_lim=4e12)``,
    ``Params.from_file("my.toml")`` or, on the command line,
    ``--set E=300e3 F_lim=4e12`` / ``--config my.toml``.
    """
    # --- planet and thermal constants ------------------------------------
    R_E: float = 6371e3          # Earth radius (m)
    g: float = 10.0              # gravity (m/s^2)
    d: float = 2900e3            # mantle thickness (m)
    D: float = 670e3             # upper-mantle thickness (m)
    T_surf: float = 300.0        # surface temperature (K); paper Table 1: 273 K
    kappa: float = 8e-7          # thermal diffusivity (m^2/s)
    C_p: float = 1200.0          # heat capacity (J/kg/K)
    k_ocean: float = 3.0         # oceanic lithosphere conductivity (W/m/K)
    M_earth: float = 5.9736e24   # mass in the secular cooling term M C_p dT/dt (kg)
    M_mantle: float = 4.079e24   # mantle mass, for radiogenic heating (kg)
    cont_frac: float = 0.40      # continental fraction of the surface
    rho_cont: float = 2700.0     # continental crust density, for its mass (kg/m^3)
    cont_thick: float = 40e3     # continental crust thickness, for its mass (m)
    # --- rheology and densities (reference values at T_p) ----------------
    T_p: float = 1625.0          # present-day reference mantle temperature (K)
    eta_lm_p: float = 5e22       # lower-mantle viscosity at T_p (Pa s)
    eta_um_p: float = 5e20       # upper-mantle viscosity at T_p (Pa s)
    eta_pl_p: float = 1.25e23    # plate (bending) viscosity at T_p (Pa s)
    E: float = 350e3             # activation energy, all layers (J/mol), Eq. 2
    rho_um_p: float = 3300.0     # upper-mantle density at T_p (kg/m^3)
    drho_p: float = 65.0         # slab density excess at T_p (kg/m^3)
    alpha_um: float = 2e-5       # mantle thermal expansivity (1/K)
    alpha_pl: float = 3.5e-5     # plate expansivity, ridge push and densities (1/K)
    rho_dep_T: bool = True       # temperature-dependent densities (code default)
    R_min: float = 300e3         # minimum radius of curvature at trenches (m)
    V_sink_p: float = 1.0        # present slab sinking speed in the lower mantle (cm/yr)
    slab_max_depth: float = 2800e3   # slabs stop lengthening at this depth (m)
    # --- behaviour laws ----------------------------------------------------
    si_mode: str = "brittle"     # critical age law: 'brittle' (Eq. 11),
                                 # 'convective' (C12 Eq. 21) or 'constant'
    tau_sub_p: float = 180.0     # present-day critical age for subduction (Myr)
    tau_sub_noise: float = 20.0  # uniform noise on the critical age, redrawn at
                                 # every check (Myr); 0 switches it off
    si_at_staples: bool = True   # allow subduction initiation at staples
    si_at_continents: bool = True    # ... and at passive margins
    init_depth_factor: float = 0.1   # new slab depth = factor * plate thickness
    near_interface: float = 0.02     # no initiation closer than this to a boundary (deg)
    F_lim: float = 2e12          # continental strength (N/m); inf: no breakup
    h_warm: float = 350e3        # subcontinental warming layer thickness b (m)
    breakup_spread: float = 0.333    # rift position: middle +/- spread * width
    breakup_requires: str = "both"   # trenches needed on the continent's plate:
                                     # 'both' (code) or 'one' (2020 text)
    fixed: bool = False          # fixed plate boundaries (2020 section 3.1)
    # --- radiogenic heating (ppm) -------------------------------------------
    U_BSE: float = 0.020         # bulk silicate Earth (McDonough & Sun, 1995)
    Th_BSE: float = 0.079
    K_BSE: float = 240.0
    U_cont: float = 1.30         # continental crust (Rudnick & Gao, 2003)
    Th_cont: float = 5.60
    K_cont: float = 1.5e4
    H_constant: float | None = None  # if set, constant radiogenic power (TW)
    # --- numerics -----------------------------------------------------------
    dx: float = 0.5              # node spacing and CFL length (deg); paper: 0.25
    courant: float = 0.5         # Courant number
    dt_max: float = 5e4          # maximum time step (yr); paper: 5000
    dt_min: float = 1.0          # minimum time step (yr), except for events

    # ---- convenience --------------------------------------------------------
    def copy(self, **kw):
        """new Params with some fields replaced (unknown names raise)"""
        self._check_names(kw)
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def _check_names(cls, kw):
        names = {f.name for f in fields(cls)}
        bad = [k for k in kw if k not in names]
        if bad:
            raise KeyError("unknown parameter(s): %s" % ", ".join(bad))

    @classmethod
    def from_dict(cls, d, base=None):
        """build from a dict; strings 'inf'/'none' are accepted for floats"""
        base = base or cls()
        kw = {}
        for k, v in d.items():
            kw[k] = _coerce(getattr(base, k, None), v) if hasattr(base, k) else v
        return base.copy(**kw)

    @classmethod
    def from_file(cls, path, base=None):
        """read a .toml or .json file of parameter values"""
        if str(path).endswith(".toml"):
            import tomllib
            with open(path, "rb") as f:
                d = tomllib.load(f)
        else:
            with open(path) as f:
                d = json.load(f)
        d = d.get("params", d)          # allow a [params] table
        return cls.from_dict(d, base)

    def describe(self, only_changed=False):
        """text table of parameter values"""
        ref = Params()
        lines = []
        for f in fields(self):
            v = getattr(self, f.name)
            if only_changed and v == getattr(ref, f.name):
                continue
            lines.append("%-18s = %s" % (f.name, v))
        return "\n".join(lines)


def _coerce(current, value):
    """convert command-line/config strings to the type of a default value"""
    if not isinstance(value, str):
        return value
    v = value.strip()
    if v.lower() in ("none", "null"):
        return None
    if isinstance(current, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(current, str):
        return v
    return float(v)            # handles 'inf', '1e12', ...


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------
class Ocean:
    """oceanic lithosphere: node positions s (deg, s[0]=0) and ages a (yr)"""
    __slots__ = ("s", "a")

    def __init__(self, s, a):
        self.s = np.asarray(s, dtype=float)
        self.a = np.asarray(a, dtype=float)

    @property
    def w(self):
        """width (deg)"""
        return float(self.s[-1])


class Cont:
    """continent: width w (deg), warming clock life (yr), identifier cid"""
    __slots__ = ("w", "life", "cid")

    def __init__(self, w, life=0.0, cid=0):
        self.w = float(w)
        self.life = float(life)
        self.cid = cid


class Plate:
    """rigid plate: ordered pieces and velocity U (deg/yr)"""
    __slots__ = ("pieces", "U")

    def __init__(self, pieces, U=0.0):
        self.pieces = pieces
        self.U = U

    @property
    def L(self):
        """length (deg)"""
        return float(sum(p.w for p in self.pieces))


class Boundary:
    """plate boundary: kind 'R' (ridge) or 'T' (trench).

    Trench attributes: polarity ``sub`` (see module docstring), slab depth
    ``Z`` (m), a record of slab thickness versus depth (``prof_z``,
    ``prof_h``, used for the mean slab thickness in slab pull, as in the
    code), and the current plate thickness at the trench ``H_tr`` (m).
    Ridge attribute: ``active`` (False while the two plates converge; they
    then move as one unit for that step, as in the code).
    """
    __slots__ = ("kind", "sub", "Z", "prof_z", "prof_h", "active", "H_tr", "bid")

    def __init__(self, kind, sub=0, Z=0.0, H0=0.0, bid=0):
        self.kind = kind
        self.sub = sub
        self.Z = float(Z)
        self.active = True
        self.H_tr = H0
        self.prof_z = np.array([self.Z]) if kind == "T" else None
        self.prof_h = np.array([H0]) if kind == "T" else None
        self.bid = bid


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class Model:
    """state and time stepping.

    Usually created with :func:`random_earth`, :func:`fixed_earth` or
    :meth:`Model.from_layout`, then advanced with :meth:`run` or :meth:`step`.
    """

    def __init__(self, params: Params, x0, plates, bounds, T0, age_Ma, seed=0):
        self.p = params
        self.x0 = float(x0)          # position of the low-angle end of plates[0]
        self.plates = plates
        self.bounds = bounds
        self.T = float(T0)           # mantle potential temperature (K)
        self.age = float(age_Ma) * 1e6   # model time as age before present (yr)
        self.time = 0.0              # elapsed time (yr)
        self.rng = np.random.default_rng(seed)
        self.deg2m = math.pi / 180.0 * params.R_E
        self.surf = 4.0 * math.pi * params.R_E ** 2
        self.events = []             # list of (age_Ma, kind)
        self.anomalies = 0           # count of situations that should not occur
        self._next_id = 1000
        self._init_radiogenic()
        self.compute_velocities()

    # ======================================================================
    # construction
    # ======================================================================
    @classmethod
    def from_layout(cls, params, bounds, continents, age_fn, T0, age_Ma,
                    seed=0, cont_life=0.0):
        """build a state from a geometric description.

        bounds     : list of dicts {pos (deg), kind 'R'|'T', sub (trench
                     polarity), Z (initial slab depth, m)}, any order
        continents : list of (start_deg, width_deg); continents must not
                     straddle a boundary (they may end exactly at one)
        age_fn     : function x_deg -> seafloor age (yr) at global position x
        cont_life  : initial warming clock of continents (yr)
        """
        bl = sorted(bounds, key=lambda b: b["pos"] % 360.0)
        pos = [b["pos"] % 360.0 for b in bl]
        conts = [((c0 % 360.0), w) for c0, w in continents]
        dx = params.dx
        n = len(pos)
        plates, bobjs = [], []

        def cont_cover(a, b):
            """continent intervals inside [a, b) (unwrapped coordinates)"""
            out = []
            for c0, w in conts:
                for k in (-1, 0, 1):
                    cs = c0 + 360.0 * k
                    ce = cs + w
                    lo, hi = max(cs, a), min(ce, b)
                    if hi - lo > EPS_W:
                        out.append((lo, hi))
            return sorted(out)

        def ocean_piece(a, b):
            m = max(1, int(math.ceil((b - a) / dx)))
            xs = np.linspace(a, b, m + 1)
            ages = np.array([age_fn(x % 360.0) for x in xs])
            return Ocean(xs - a, ages)

        def make_pieces(a, b):
            pieces = []
            cur = a
            for lo, hi in cont_cover(a, b):
                if lo - cur > EPS_W:
                    pieces.append(ocean_piece(cur, lo))
                pieces.append(Cont(hi - lo, cont_life, cid=0))
                cur = hi
            if b - cur > EPS_W:
                pieces.append(ocean_piece(cur, b))
            if not pieces:
                pieces.append(Ocean([0.0, 0.0], [0.0, 0.0]))
            return pieces

        if n == 0:
            plates = [Plate(make_pieces(0.0, 360.0))]
            x0 = 0.0
        else:
            # plate k spans [pos[k], pos[k+1]]; its high-end boundary is bl[k+1]
            for k in range(n):
                a = pos[k]
                b = pos[(k + 1) % n] if k + 1 < n else pos[0] + 360.0
                if n == 1:
                    b = a + 360.0
                plates.append(Plate(make_pieces(a, b)))
            x0 = pos[0]
            for k in range(n):
                bd = bl[(k + 1) % n]
                bobjs.append(Boundary(bd["kind"], bd.get("sub", 0), Z=bd.get("Z", 0.0)))
        m = cls(params, x0, plates, bobjs, T0, age_Ma, seed=seed)
        for pl in m.plates:            # merge touching continents
            m._cleanup(pl)
        for pl in m.plates:
            for pc in pl.pieces:
                if isinstance(pc, Cont):
                    pc.cid = m._new_id()
        # slabs: minimum depth 0.1 H and a uniform thickness record (code: initSlabs)
        for k, b in enumerate(m.bounds):
            if b.kind == "T":
                H = m._H(m._trench_age(k))
                b.Z = max(b.Z, H * params.init_depth_factor)
                b.H_tr = H
                b.prof_z = np.array([b.Z])
                b.prof_h = np.array([H])
        m.compute_velocities()
        return m

    def _new_id(self):
        self._next_id += 1
        return self._next_id

    # ======================================================================
    # material properties
    # ======================================================================
    def eta(self, eta_p, T=None):
        """Arrhenius viscosity, Eq. 2"""
        T = self.T if T is None else T
        return eta_p * math.exp(self.p.E / R_GAS * (1.0 / T - 1.0 / self.p.T_p))

    def tau_sub(self, T=None):
        """critical seafloor age for subduction initiation (yr)"""
        p = self.p
        T = self.T if T is None else T
        if p.si_mode == "brittle":        # constant yield stress, Eq. 11
            t = p.tau_sub_p * ((p.T_p - p.T_surf) / (T - p.T_surf)) ** 2
        elif p.si_mode == "convective":   # critical Rayleigh number, C12 Eq. 21
            t = (p.tau_sub_p * ((p.T_p - p.T_surf) / (T - p.T_surf)) ** (2 / 3)
                 * math.exp(2 / 3 * p.E / R_GAS * (1 / T - 1 / p.T_p)))
        else:
            t = p.tau_sub_p
        return t * 1e6

    def _H(self, age_yr):
        """plate thickness for a seafloor age (m), Eq. 3"""
        return THICK_COEFF * math.sqrt(self.p.kappa * max(age_yr, 0.0) * SEC_YR)

    def _densities(self):
        """(plate density, upper-mantle density); T-dependent as in the code,
        with the plate at the mean temperature of a half-space plate"""
        p = self.p
        rho_pl_p = p.rho_um_p + p.drho_p
        if not p.rho_dep_T:
            return rho_pl_p, p.rho_um_p
        Tpl = p.T_surf + (self.T - p.T_surf) * COEFF_T_PLATE
        Tpl_p = p.T_surf + (p.T_p - p.T_surf) * COEFF_T_PLATE
        rho_pl = rho_pl_p * (1.0 - p.alpha_pl * (Tpl - Tpl_p))
        rho_um = p.rho_um_p * (1.0 - p.alpha_um * (self.T - p.T_p))
        return rho_pl, rho_um

    # ======================================================================
    # radiogenic heating (depleted mantle, constant continental mass)
    # ======================================================================
    def _init_radiogenic(self):
        """present-day depleted-mantle concentrations from BSE minus crust"""
        p = self.p
        m_cont = p.cont_frac * self.surf * p.rho_cont * p.cont_thick
        m_prim = p.M_mantle + m_cont
        iso = [  # heat (W/kg), half-life (Myr), BSE (ppm), crust (ppm)
            (5.69e-4, 704.0, p.U_BSE * 0.0072, p.U_cont * 0.0072),      # 235U
            (9.46e-5, 4470.0, p.U_BSE * 0.9927, p.U_cont * 0.9927),     # 238U
            (2.64e-5, 14.0e3, p.Th_BSE, p.Th_cont),                     # 232Th
            (2.92e-5, 1.25e3, p.K_BSE * 1.28e-4, p.K_cont * 1.28e-4),   # 40K
        ]
        self._iso = []
        for heat, thalf, c_bse, c_cont in iso:
            c_dep = (c_bse * m_prim - c_cont * m_cont) / p.M_mantle
            self._iso.append((heat, LN2 / thalf, c_dep))

    def radiogenic_TW(self, age_Ma=None):
        """mantle radiogenic power (TW) at a given age"""
        if self.p.H_constant is not None:
            return self.p.H_constant
        age_Ma = self.age / 1e6 if age_Ma is None else age_Ma
        s = sum(h * c * math.exp(age_Ma * lam) for h, lam, c in self._iso)
        return self.p.M_mantle * 1e-18 * s

    # ======================================================================
    # geometry helpers
    # ======================================================================
    def lo_b(self, i):
        """boundary at the low-angle end of plate i (None if unbounded)"""
        return self.bounds[(i - 1) % len(self.bounds)] if self.bounds else None

    def hi_b(self, i):
        """boundary at the high-angle end of plate i"""
        return self.bounds[i % len(self.bounds)] if self.bounds else None

    def plate_x(self):
        """low-angle end position of each plate (deg, not wrapped)"""
        xs, x = [], self.x0
        for pl in self.plates:
            xs.append(x)
            x += pl.L
        return xs

    def bound_pos(self):
        """position of each boundary (deg, in [0, 360))"""
        xs = self.plate_x()
        return [(xs[k] + self.plates[k].L) % 360.0 for k in range(len(self.bounds))]

    def _trench_age(self, k):
        """seafloor age (yr) of the subducting plate at trench k"""
        b = self.bounds[k]
        if b.sub == -1:
            pc = self.plates[k].pieces[-1]
            return pc.a[-1] if isinstance(pc, Ocean) else 0.0
        pc = self.plates[(k + 1) % len(self.plates)].pieces[0]
        return pc.a[0] if isinstance(pc, Ocean) else 0.0

    # ======================================================================
    # forces (all in N/m, positive toward larger angles)
    # ======================================================================
    def _trench_terms(self, b, H_tr):
        """slab pull SP, slab suction SS (driving, N/m) and the equivalent
        viscosities of vertical shear and bending (Pa s) for trench b"""
        p = self.p
        rho_pl, rho_um = self._densities()
        drho = (rho_pl - rho_um) if p.rho_dep_T else p.drho_p
        Zum = min(b.Z, p.D)
        # slab pull in the upper mantle only (Eq. 5), with the mean thickness
        # of the slab record and a 10 km minimum (code: subduction.cpp)
        Hm = self._mean_slab_thickness(b, H_tr)
        SP = drho * p.g * max(Hm, 10e3) * Zum
        # slab suction from the part in the lower mantle (Eq. 6; prefactor
        # 2/9 and division by the lower-mantle thickness as in the code)
        Zlm = max(b.Z - p.D, 0.0)
        SS = (2.0 / 9.0) * p.eta_lm_p * p.V_sink_p * 0.01 / SEC_YR * Zlm / (p.d - p.D)
        # vertical shear on the slab in the upper mantle (Eq. B2)
        eVS = 2.0 * self.eta(p.eta_um_p) * Zum / p.D
        # bending at the trench (Eq. B7, Buffett 2006)
        eB = (2.0 / 3.0) * self.eta(p.eta_pl_p) * (H_tr / p.R_min) ** 3
        return SP, SS, eVS, eB

    def _mean_slab_thickness(self, b, H_tr):
        """mean slab thickness over the upper-mantle part of the slab, from
        the record of thickness versus depth (trapezoidal rule)"""
        ztot = min(b.Z, self.p.D)
        if ztot <= 0.0:
            return H_tr
        z = np.concatenate(([0.0], b.prof_z))
        h = np.concatenate(([H_tr], b.prof_h))
        if ztot >= z[-1]:
            zz = np.append(z, ztot)
            hh = np.append(h, h[-1])
        else:
            k = np.searchsorted(z, ztot)
            zz = np.append(z[:k], ztot)
            hh = np.append(h[:k], np.interp(ztot, z, h))
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(trap(hh, zz) / ztot)

    def _rp(self, age_yr):
        """ridge push for a section whose far end has the given age (Eq. 4),
        with alpha_pl and the plate density as in the code"""
        p = self.p
        rho_pl, _ = self._densities()
        rho = rho_pl if p.rho_dep_T else p.rho_um_p + p.drho_p
        return (p.alpha_pl * p.g * p.kappa * rho * (self.T - p.T_surf)
                * max(age_yr, 0.0) * SEC_YR)

    def _eta_H(self, L_deg):
        """equivalent viscosity of mantle drag for a plate of length L
        (Eq. B5 as implemented in the code: factor 2 and lower-mantle
        thickness d/2 - D)"""
        p = self.p
        return 2.0 * L_deg * self.deg2m / (p.D / self.eta(p.eta_um_p)
                                           + (0.5 * p.d - p.D) / self.eta(p.eta_lm_p))

    def _end_terms(self, b, at_lo, pieces, tt):
        """contribution of boundary b at one end of a plate.

        at_lo  : True if b is at the low-angle end of the plate
        pieces : pieces of the plate (only the end pieces are used)
        tt     : trench terms keyed by id(b)
        returns (driving force N/m, resisting viscosity Pa s)
        """
        if b is None:
            return 0.0, 0.0
        sgn = 1.0 if at_lo else -1.0          # direction away from b
        if b.kind == "R":
            if not b.active:
                return 0.0, 0.0
            # ridge push uses the age at the far end of the oceanic section
            # touching the ridge (sections end at staples and continents)
            pc = pieces[0] if at_lo else pieces[-1]
            age = (pc.a[-1] if at_lo else pc.a[0]) if isinstance(pc, Ocean) else 0.0
            return sgn * self._rp(age), 0.0
        SP, SS, eVS, eB = tt[id(b)]
        subducting = (b.sub == 1) if at_lo else (b.sub == -1)
        F = -sgn * SS                         # suction acts on both plates
        if subducting:
            F -= sgn * SP                     # pull only on the subducting plate
        # shear and bending resist both the subducting and the overriding
        # plate (the trench moves with the overriding plate)
        return F, eVS + eB

    def _all_trench_terms(self):
        """trench terms for all trenches, keyed by id(boundary)"""
        tt = {}
        for k, b in enumerate(self.bounds):
            if b.kind == "T":
                b.H_tr = self._H(self._trench_age(k))
                tt[id(b)] = self._trench_terms(b, b.H_tr)
        return tt

    def plate_drive(self, i, tt=None):
        """net driving force on plate i alone (N/m)"""
        tt = self._all_trench_terms() if tt is None else tt
        pl = self.plates[i]
        F1, _ = self._end_terms(self.lo_b(i), True, pl.pieces, tt)
        F2, _ = self._end_terms(self.hi_b(i), False, pl.pieces, tt)
        return F1 + F2

    # ======================================================================
    # velocities
    # ======================================================================
    def compute_velocities(self):
        """plate velocities from the force balance, Eq. 8:
        U = (RP + SP + SS) / (eta_MD + eta_VS + eta_B).
        Plates joined by an inactive (converging) ridge move as one unit.
        Without any trench nothing moves (code: _noSubduction)."""
        n = len(self.bounds)
        for pl in self.plates:
            pl.U = 0.0
        if n == 0 or not any(b.kind == "T" for b in self.bounds):
            return
        tt = self._all_trench_terms()
        conv = SEC_YR / self.deg2m            # m/s -> deg/yr
        k0 = next(k for k, b in enumerate(self.bounds)
                  if not (b.kind == "R" and not b.active))
        i = (k0 + 1) % n                      # first plate after an active boundary
        done = 0
        while done < n:
            unit = [i]
            while True:
                b = self.bounds[unit[-1] % n]
                if b.kind == "R" and not b.active and len(unit) < n:
                    unit.append((unit[-1] + 1) % n)
                else:
                    break
            first, last = self.plates[unit[0]], self.plates[unit[-1]]
            F1, e1 = self._end_terms(self.lo_b(unit[0]), True, first.pieces, tt)
            F2, e2 = self._end_terms(self.hi_b(unit[-1]), False, last.pieces, tt)
            L = sum(self.plates[j].L for j in unit)
            eta = self._eta_H(L) + e1 + e2
            U = (F1 + F2) / eta * conv if eta > 0 else 0.0
            for j in unit:
                self.plates[j].U = U
            done += len(unit)
            i = (unit[-1] + 1) % n

    def bound_velocities(self):
        """boundary velocities (deg/yr): ridges at the mean of their two
        plates (symmetric spreading), trenches with the overriding plate"""
        n = len(self.bounds)
        V = []
        for k, b in enumerate(self.bounds):
            if self.p.fixed:
                V.append(0.0)
                continue
            Ul, Uu = self.plates[k].U, self.plates[(k + 1) % n].U
            if b.kind == "R":
                V.append(0.5 * (Ul + Uu) if b.active else Ul)
            else:
                V.append(Uu if b.sub == -1 else Ul)
        return V

    def convergence(self, k):
        """closing rate at boundary k (deg/yr), > 0 when plates converge"""
        n = len(self.plates)
        return self.plates[k].U - self.plates[(k + 1) % n].U

    # ======================================================================
    # heat flow
    # ======================================================================
    def heat_flow_TW(self):
        """oceanic heat flow (TW), Eq. 12 integrated exactly for ages varying
        linearly between nodes; continents are insulating. A node interval of
        angular width ds covers the lune area ds/360 * 4 pi R^2."""
        p = self.p
        coef = p.k_ocean * (self.T - p.T_surf) / math.sqrt(math.pi * p.kappa)
        area = self.surf / 360.0
        tot = 0.0
        for pl in self.plates:
            for pc in pl.pieces:
                if isinstance(pc, Ocean) and len(pc.s) > 1:
                    ds = np.diff(pc.s)
                    sa = np.sqrt(np.maximum(pc.a, 0.0) * SEC_YR)
                    den = sa[:-1] + sa[1:]
                    m = ds > 0
                    # mean of 1/sqrt(age) over a linear segment is
                    # 2/(sqrt(a1)+sqrt(a2)); floor ~1000 yr as in the code
                    den = np.maximum(den[m], 2.0 * math.sqrt(3e10))
                    tot += float(np.sum(2.0 * ds[m] / den))
        return coef * area * tot / 1e12

    # ======================================================================
    # time stepping
    # ======================================================================
    def step(self):
        """advance by one time step; returns dt (yr). Order as in the code
        (earth.cpp, _timeLoop): move material and boundaries with the current
        velocities, treat collisions, apply behaviour laws, recompute
        velocities, then update the mantle temperature."""
        p = self.p
        n = len(self.bounds)
        V = self.bound_velocities() if n else []
        # Courant condition on the fastest plate
        vmax = max([abs(pl.U) for pl in self.plates] + [1e-30])
        dt = min(p.dt_max, max(p.dt_min, p.courant * p.dx / vmax))
        # shorten the step so that topology changes happen exactly at its end:
        # a plate shrinking to zero length, or a continent reaching a
        # consuming plate end
        shrink = [0.0] * len(self.plates)
        if n and not p.fixed:
            for i, pl in enumerate(self.plates):
                VL, VH = V[(i - 1) % n], V[i % n]
                dL = VH - VL if n > 1 else 0.0
                shrink[i] = dL
                if dL < 0:
                    dt = min(dt, pl.L / -dL)
                cL = VL - pl.U                # consumption rate at the low end
                if cL > 0:
                    dist = self._dist_to_cont(pl.pieces, True)
                    if dist is not None:
                        dt = min(dt, max(dist, 0.0) / cL)
                cH = pl.U - VH                # consumption rate at the high end
                if cH > 0:
                    dist = self._dist_to_cont(pl.pieces, False)
                    if dist is not None:
                        dt = min(dt, max(dist, 0.0) / cH)
        dt = max(dt, 0.0)
        # 1. transport
        self._advance(dt, V)
        self.time += dt
        self.age -= dt
        # 2. collisions
        if n and not p.fixed:
            self._treat_events(shrink)
        # 3. behaviour laws (all ridges are reactivated first)
        for b in self.bounds:
            if b.kind == "R":
                b.active = True
        if not p.fixed:
            self._check_diving()
            self._check_breakup()
            self._check_backarc()
        # 4. velocities and consistency checks
        self.compute_velocities()
        if not p.fixed:
            self._verify_subductions()
        self._verify_ridges()
        # 5. thermal budget, Eq. 1: M C_p dT/dt = H - Q
        Q = self.heat_flow_TW()
        H = self.radiogenic_TW()
        self.T += (H - Q) * 1e12 * dt * SEC_YR / (p.M_earth * p.C_p)
        self.last_Q, self.last_H, self.last_dt = Q, H, dt
        return dt

    def _dist_to_cont(self, pieces, at_lo):
        """distance (deg) from a plate end to its nearest continent, or None"""
        d = 0.0
        seq = pieces if at_lo else reversed(pieces)
        for pc in seq:
            if isinstance(pc, Cont):
                return d
            d += pc.w
        return None

    # ======================================================================
    # material transport
    # ======================================================================
    def _advance(self, dt, V):
        """age everything by dt, lengthen slabs, and move material relative
        to the plate ends: accretion at ridges, consumption at trenches"""
        p = self.p
        n = len(self.bounds)
        for pl in self.plates:
            for pc in pl.pieces:
                if isinstance(pc, Ocean):
                    pc.a += dt
                else:
                    pc.life += dt
        if n == 0 or dt == 0.0:
            return
        # slabs lengthen at the convergence rate until slab_max_depth
        for k, b in enumerate(self.bounds):
            if b.kind == "T":
                dz = self.convergence(k) * dt * self.deg2m
                if b.Z < p.slab_max_depth:
                    b.Z = max(b.Z + dz, 0.0)
                self._update_slab_profile(b, dz)
        for i, pl in enumerate(self.plates):
            bL, bH = self.lo_b(i), self.hi_b(i)
            VL, VH = V[(i - 1) % n], V[i % n]
            if p.fixed:
                # with fixed boundaries only ridge-trench plates carry seafloor
                # through (code: moveAges, plates of type "RS")
                if {bL.kind, bH.kind} != {"R", "T"}:
                    continue
            # displacement of material relative to each plate end
            rel_lo = (pl.U - VL) * dt     # > 0: material leaves the low end
            rel_hi = (pl.U - VH) * dt     # > 0: material enters the high end
            # accretion first, then consumption, so that a plate fed at one
            # end and consumed at the other within one step is exact
            if rel_lo > 0:
                if bL.kind == "R":
                    self._accrete(pl, True, rel_lo, dt)
                else:
                    self._stretch(pl, True, rel_lo)
            if rel_hi < 0:
                if bH.kind == "R":
                    self._accrete(pl, False, -rel_hi, dt)
                else:
                    self._stretch(pl, False, -rel_hi)
            if rel_lo < 0:
                self._consume(pl, True, -rel_lo)
            if rel_hi > 0:
                self._consume(pl, False, rel_hi)
        if not p.fixed:
            self.x0 = (self.x0 + V[-1] * dt) % 360.0   # low end of plates[0]

    def _update_slab_profile(self, b, dz):
        """slab thickness record as in the code (subduction.cpp): entries move
        down with the slab, new entries are added at the top every dx (in
        depth) by linear interpolation between the current trench thickness
        and the top entry, and entries below slab_max_depth are dropped"""
        res = self.p.dx * self.deg2m
        z = b.prof_z + dz
        h = b.prof_h
        keep = z <= self.p.slab_max_depth
        if not keep.any():
            keep[-1] = True
        z, h = z[keep], h[keep]
        if z[0] > res:
            zn = z[0] - res * np.arange(int(z[0] // res), 0, -1)
            zn = zn[zn >= res * (1 - 1e-9)]
            if len(zn):
                hn = b.H_tr + (h[0] - b.H_tr) * zn / z[0]
                z = np.concatenate((zn, z))
                h = np.concatenate((hn, h))
        b.prof_z, b.prof_h = z, h

    def _accrete(self, pl, at_lo, w, dt):
        """add a strip of new seafloor of width w at a ridge end; ages run
        from 0 at the ridge to dt at the inner edge of the strip. Nodes closer
        than dx are thinned so the node count stays ~ width/dx."""
        dx = self.p.dx
        if at_lo:
            pc = pl.pieces[0]
            if isinstance(pc, Ocean):
                s = np.concatenate(([0.0], pc.s + w))
                a = np.concatenate(([0.0], pc.a))
                if len(s) >= 3 and s[2] - s[0] < dx:
                    s = np.delete(s, 1)
                    a = np.delete(a, 1)
                pc.s, pc.a = s, a
            else:                              # ridge next to a continent
                pl.pieces.insert(0, Ocean([0.0, w], [0.0, dt]))
        else:
            pc = pl.pieces[-1]
            if isinstance(pc, Ocean):
                s = np.append(pc.s, pc.w + w)
                a = np.append(pc.a, 0.0)
                if len(s) >= 3 and s[-1] - s[-3] < dx:
                    s = np.delete(s, -2)
                    a = np.delete(a, -2)
                pc.s, pc.a = s, a
            else:
                pl.pieces.append(Ocean([0.0, w], [dt, 0.0]))

    def _stretch(self, pl, at_lo, w):
        """gap opening at a trench end. Diverging trenches are removed by
        _verify_subductions, so this should not happen; it is counted in
        self.anomalies and the edge age is extended to keep the state valid."""
        if w > 1e-12:
            self.anomalies += 1
        pc = pl.pieces[0] if at_lo else pl.pieces[-1]
        if isinstance(pc, Ocean):
            if at_lo:
                pc.s = np.concatenate(([0.0], pc.s + w))
                pc.a = np.concatenate(([pc.a[0]], pc.a))
            else:
                pc.s = np.append(pc.s, pc.w + w)
                pc.a = np.append(pc.a, pc.a[-1])
        else:
            pc.w += w

    def _consume(self, pl, at_lo, w):
        """remove width w of seafloor from a plate end (subduction).
        Whole pieces are removed when consumed (a staple passes the trench).
        Consumption stops at a continent; the time step is chosen so that this
        happens exactly at the end of a step, and the collision is then
        treated in _treat_events. Returns the width that could not be
        consumed."""
        pieces = pl.pieces
        while w > 0:
            idx = 0 if at_lo else len(pieces) - 1
            pc = pieces[idx]
            if isinstance(pc, Cont):
                if w > 1e-7:
                    self.anomalies += 1
                return w
            if pc.w <= w + 1e-13:              # the whole piece goes
                w -= pc.w
                if len(pieces) == 1:           # plate vanishes: keep an empty piece
                    edge = pc.a[-1] if at_lo else pc.a[0]
                    pieces[0] = Ocean([0.0, 0.0], [edge, edge])
                    return 0.0
                pieces.pop(idx)
                continue
            if at_lo:                          # cut the piece at s = w
                k = int(np.searchsorted(pc.s, w, side="right"))
                acut = float(np.interp(w, pc.s, pc.a))
                s = np.concatenate(([0.0], pc.s[k:] - w))
                a = np.concatenate(([acut], pc.a[k:]))
                if len(s) > 2 and s[1] < 1e-12:
                    s, a = np.delete(s, 1), np.delete(a, 1)
            else:                              # cut the piece at s = width - w
                cut = pc.w - w
                k = int(np.searchsorted(pc.s, cut, side="left"))
                acut = float(np.interp(cut, pc.s, pc.a))
                s = np.append(pc.s[:k], cut)
                a = np.append(pc.a[:k], acut)
                if len(s) > 2 and s[-1] - s[-2] < 1e-12:
                    s, a = np.delete(s, -2), np.delete(a, -2)
            if len(s) == 1:
                s, a = np.array([0.0, s[0]]), np.array([a[0], a[0]])
            pc.s, pc.a = s, a
            return 0.0
        return 0.0

    # ======================================================================
    # topology changes
    # ======================================================================
    def _treat_events(self, shrink):
        """collisions at the end of a step (2020 Fig. 1d; code: treatCollisions)
        - a plate shrinking to zero length disappears; its two boundaries are
          removed and the neighbouring plates are welded with a staple
          (ridge-trench and trench-trench collisions)
        - a continent reaching a trench on the subducting side stops
          subduction there: the trench is removed and the plates are welded,
          merging continents if the overriding edge is continental"""
        vanished = [self.plates[i] for i, pl in enumerate(self.plates)
                    if shrink[i] < 0 and pl.L <= 1e-8]
        for pl in vanished:
            if pl not in self.plates or len(self.plates) < 2:
                continue
            i = self.plates.index(pl)
            b1, b2 = self.lo_b(i), self.hi_b(i)
            self._log("vanish")
            self._weld(self.bounds.index(b2))
            if b1 in self.bounds:
                self._weld(self.bounds.index(b1))
        changed = True
        while changed:
            changed = False
            for k, b in enumerate(self.bounds):
                if b.kind != "T":
                    continue
                n = len(self.plates)
                if b.sub == -1:
                    pl, at_lo = self.plates[k], False
                else:
                    pl, at_lo = self.plates[(k + 1) % n], True
                d = self._dist_to_cont(pl.pieces, at_lo)
                if d is not None and d <= 1e-8:
                    other = self.plates[(k + 1) % n] if b.sub == -1 else self.plates[k]
                    edge = other.pieces[0] if b.sub == -1 else other.pieces[-1]
                    self._log("collision_cc" if isinstance(edge, Cont) else "collision_ct")
                    self._weld(k)
                    changed = True
                    break

    def _log(self, kind):
        self.events.append((self.age / 1e6, kind))

    def _cleanup(self, pl):
        """drop zero-width ocean pieces inside a plate and merge touching
        continents. Zero-width pieces at plate ends are kept: they mark new
        seafloor sections at ridges."""
        pcs = pl.pieces
        out = []
        for j, pc in enumerate(pcs):
            if isinstance(pc, Ocean) and pc.w <= EPS_W and 0 < j < len(pcs) - 1:
                continue
            out.append(pc)
        merged = [out[0]]
        for pc in out[1:]:
            prev = merged[-1]
            if isinstance(pc, Cont) and isinstance(prev, Cont):
                merged[-1] = self._merge_conts(prev, pc)
            else:
                merged.append(pc)
        pl.pieces = merged

    def _cleanup_wrap(self):
        """single unbounded plate: its two ends meet, so treat that contact
        like an internal one"""
        pl = self.plates[0]
        while len(pl.pieces) > 1 and isinstance(pl.pieces[0], Ocean) and pl.pieces[0].w <= EPS_W:
            pl.pieces.pop(0)
        while len(pl.pieces) > 1 and isinstance(pl.pieces[-1], Ocean) and pl.pieces[-1].w <= EPS_W:
            pl.pieces.pop()
        if (len(pl.pieces) > 1 and isinstance(pl.pieces[0], Cont)
                and isinstance(pl.pieces[-1], Cont)):
            first = pl.pieces.pop(0)
            self.x0 = (self.x0 + first.w) % 360.0
            pl.pieces[-1] = self._merge_conts(pl.pieces[-1], first)

    def _merge_conts(self, c1, c2):
        """continent-continent collision: the new continent gets the
        width-weighted warming of the two, converted to an equivalent warming
        time (code: _collisionContinentContinent, computeLifetime0)"""
        W1, W2 = self._warming(c1), self._warming(c2)
        w = c1.w + c2.w
        new = Cont(w, 0.0, cid=self._new_id())
        Wn = (W1 * c1.w + W2 * c2.w) / w
        Tm, tw = self._warm_scales(w)
        r = min(Wn / Tm, 0.999999) if Tm > 0 else 0.0
        new.life = tw * math.atanh(r)
        return new

    def _weld(self, k):
        """remove bounds[k] and merge its two neighbouring plates"""
        n = len(self.bounds)
        if n == 1:                           # last boundary: one unbounded plate
            pl = self.plates[0]
            self.bounds = []
            self._cleanup(pl)
            self._cleanup_wrap()
            return
        i, j = k, (k + 1) % n
        A, B = self.plates[i], self.plates[j]
        merged = Plate(A.pieces + B.pieces, A.U)
        self._cleanup(merged)
        if j == 0:                           # merged plate now starts before x0
            self.x0 = (self.x0 - A.L) % 360.0
            self.plates[0] = merged
            del self.plates[i]
        else:
            self.plates[i] = merged
            del self.plates[j]
        del self.bounds[k]

    def _split(self, i, jpc, new_b):
        """split plate i between pieces[jpc-1] and pieces[jpc] and insert
        boundary new_b there"""
        pl = self.plates[i]
        A, B = pl.pieces[:jpc], pl.pieces[jpc:]
        if not self.bounds:                  # unbounded plate: re-anchor at new_b
            wA = sum(pc.w for pc in A)
            self.plates = [Plate(B + A)]
            self._cleanup(self.plates[0])
            self.x0 = (self.x0 + wA) % 360.0
            self.bounds = [new_b]
            return
        self.plates[i] = Plate(A, pl.U)
        self.plates.insert(i + 1, Plate(B, pl.U))
        self.bounds.insert(i, new_b)

    # ======================================================================
    # behaviour laws
    # ======================================================================
    def _tau_crit(self):
        """critical age with the code's random noise (code: _randomAge)"""
        t = self.tau_sub() / 1e6
        nz = self.p.tau_sub_noise
        if nz > 0:
            t = self.rng.uniform(t, t + nz) if t < nz else self.rng.uniform(t - nz, t + nz)
        return t * 1e6

    def _contacts(self, i):
        """internal contacts of plate i as (j, local position): contact
        between pieces[j-1] and pieces[j]"""
        pcs = self.plates[i].pieces
        out, x = [], 0.0
        for j in range(1, len(pcs)):
            x += pcs[j - 1].w
            out.append((j, x))
        return out

    def _si_candidates(self, A, B):
        """ordered (sub, age) candidates for subduction initiation at a
        contact A|B (A at lower angle). At a staple the higher-angle side is
        tried first, as in the code; at a passive margin the ocean subducts
        under the continent."""
        p = self.p
        out = []
        if isinstance(A, Ocean) and isinstance(B, Ocean):
            if p.si_at_staples:
                out = [(1, B.a[0]), (-1, A.a[-1])]
        elif isinstance(A, Cont) and isinstance(B, Ocean):
            if p.si_at_continents:
                out = [(1, B.a[0])]
        elif isinstance(A, Ocean) and isinstance(B, Cont):
            if p.si_at_continents:
                out = [(-1, A.a[-1])]
        return out

    def _try_si(self, i, j, xloc, sub, age):
        """create a trench at contact j of plate i if the driving force on
        the future subducting part points toward it (code:
        _realDivingSubduction). The new slab starts at depth
        init_depth_factor * plate thickness."""
        p = self.p
        pl = self.plates[i]
        if pl.L > 0 and (xloc < p.near_interface or pl.L - xloc < p.near_interface) \
                and self.bounds:
            return False
        H = self._H(age)
        nb = Boundary("T", sub=sub, Z=H * p.init_depth_factor, H0=H, bid=self._new_id())
        nb.prof_z = np.array([nb.Z])
        nb.prof_h = np.array([H])
        if self.bounds:
            tt = self._all_trench_terms()
            tt[id(nb)] = self._trench_terms(nb, H)
            A, B = pl.pieces[:j], pl.pieces[j:]
            if sub == 1:   # B subducts; the new trench is at its low end
                F1, _ = self._end_terms(nb, True, B, tt)
                F2, _ = self._end_terms(self.hi_b(i), False, B, tt)
                ok = (F1 + F2) < -1.0
            else:          # A subducts; the new trench is at its high end
                F1, _ = self._end_terms(self.lo_b(i), True, A, tt)
                F2, _ = self._end_terms(nb, False, A, tt)
                ok = (F1 + F2) > 1.0
            if not ok:
                return False
        self._split(i, j, nb)
        self._log("si")
        return True

    def _check_diving(self):
        """subduction initiation (section 2.2.4): at staples first, then at
        passive margins, where seafloor is older than the critical age. After
        each initiation the search restarts (code: _checkDiving)."""
        p = self.p
        for _ in range(200):
            done = False
            for want in ("staple", "margin"):
                for i, pl in enumerate(self.plates):
                    pcs = pl.pieces
                    for j, xloc in self._contacts(i):
                        A, B = pcs[j - 1], pcs[j]
                        is_staple = isinstance(A, Ocean) and isinstance(B, Ocean)
                        if is_staple != (want == "staple"):
                            continue
                        for sub, age in self._si_candidates(A, B):
                            if age >= self._tau_crit() and self._try_si(i, j, xloc, sub, age):
                                done = True
                                break
                        if done:
                            break
                    if done:
                        break
                if done:
                    break
            # the wrap contact of an unbounded single plate
            if not done and not self.bounds and len(self.plates[0].pieces) > 1:
                pcs = self.plates[0].pieces
                for sub, age in self._si_candidates(pcs[-1], pcs[0]):
                    if age >= self._tau_crit():
                        H = self._H(age)
                        nb = Boundary("T", sub=sub, Z=H * p.init_depth_factor,
                                      H0=H, bid=self._new_id())
                        nb.prof_z, nb.prof_h = np.array([nb.Z]), np.array([H])
                        self.bounds = [nb]
                        self._log("si")
                        done = True
                        break
            if not done:
                return

    def _warm_scales(self, w_deg):
        """maximum subcontinental excess temperature and warming time (yr)
        for a continent of width w (Appendix C, Eq. C7; heating per unit
        mass of mantle as in the code)"""
        p = self.p
        a = w_deg * self.deg2m
        ah = 0.5 * a / p.h_warm
        Pth = 2.0 * p.h_warm * p.alpha_um * p.rho_um_p * p.g / (ah ** 2 + ah ** -2)  # Eq. C4
        Hkg = self.radiogenic_TW() * 1e12 / p.M_mantle
        eta = self.eta(p.eta_um_p)
        Tmax = math.sqrt(Hkg * eta / (p.C_p * Pth))
        tw = math.sqrt(p.C_p * eta / (Hkg * Pth)) / SEC_YR
        return Tmax, tw

    def _warming(self, c):
        """current subcontinental excess temperature (K), Eq. C6"""
        Tmax, tw = self._warm_scales(c.w)
        return Tmax * math.tanh(c.life / tw)

    def cont_force(self, c):
        """subcontinental shear force (N/m), Eq. C8"""
        p = self.p
        a = c.w * self.deg2m
        ah = 0.5 * a / p.h_warm
        Pth = 2.0 * p.h_warm * p.alpha_um * p.rho_um_p * p.g / (ah ** 2 + ah ** -2)
        return 0.25 * a ** 2 / p.h_warm * Pth * self._warming(c)

    def _check_breakup(self):
        """continental breakup (section 2.2.3): when F > F_lim on a plate with
        the required trenches, a ridge opens at a random position within
        middle +/- breakup_spread * width; the two new continents restart
        their warming clocks"""
        p = self.p
        if not np.isfinite(p.F_lim):
            return
        for _ in range(100):
            if len(self.plates) < 2:
                return
            done = False
            for i, pl in enumerate(self.plates):
                ntr = [self.lo_b(i).kind, self.hi_b(i).kind].count("T")
                if p.breakup_requires == "both" and ntr < 2:
                    continue
                if ntr < 1:
                    continue
                for j, pc in enumerate(pl.pieces):
                    if isinstance(pc, Cont) and self.cont_force(pc) >= p.F_lim:
                        f = self.rng.uniform(0.5 - p.breakup_spread, 0.5 + p.breakup_spread)
                        wa = pc.w * f
                        c1 = Cont(wa, 0.0, self._new_id())
                        c2 = Cont(pc.w - wa, 0.0, self._new_id())
                        z1 = Ocean([0.0, 0.0], [0.0, 0.0])   # new seafloor on each side
                        z2 = Ocean([0.0, 0.0], [0.0, 0.0])
                        pl.pieces[j:j + 1] = [c1, z1, z2, c2]
                        self._split(i, j + 2, Boundary("R", bid=self._new_id()))
                        self._log("breakup")
                        done = True
                        break
                if done:
                    break
            if not done:
                return

    def _check_backarc(self):
        """back-arc opening (2020 Fig. 1b2; code: _checkRidgeCreation): if
        the driving force on the overriding plate points away from the trench,
        a ridge opens just behind the trench. A new zero-width plate (sliver)
        becomes the overriding plate; a staple (or the continent edge)
        separates the old upper plate from the new seafloor."""
        for _ in range(100):
            done = False
            tt = self._all_trench_terms()
            n = len(self.bounds)
            for k, b in enumerate(self.bounds):
                if b.kind != "T":
                    continue
                o = (k + 1) % n if b.sub == -1 else k      # overriding plate
                F = self.plate_drive(o, tt)
                away = F > 1.0 if b.sub == -1 else F < -1.0
                if not away:
                    continue
                sliver = Plate([Ocean([0.0, 0.0], [0.0, 0.0])])
                rb = Boundary("R", bid=self._new_id())
                op = self.plates[o]
                if b.sub == -1:   # overriding plate on the upper-angle side
                    if isinstance(op.pieces[0], Ocean):
                        op.pieces.insert(0, Ocean([0.0, 0.0], [0.0, 0.0]))
                    self.plates.insert(k + 1, sliver)
                    self.bounds.insert(k + 1, rb)
                else:             # overriding plate on the lower-angle side
                    if isinstance(op.pieces[-1], Ocean):
                        op.pieces.append(Ocean([0.0, 0.0], [0.0, 0.0]))
                    self.plates.insert(k + 1, sliver)
                    self.bounds[k] = rb
                    self.bounds.insert(k + 1, b)
                self._log("backarc")
                done = True
                break
            if not done:
                return

    def _verify_subductions(self):
        """a trench whose plates diverge is replaced by a staple (code:
        _verifySubductions); this mostly removes new trenches that fail"""
        for _ in range(100):
            bad = None
            for k, b in enumerate(self.bounds):
                if b.kind == "T" and self.convergence(k) < 0:
                    bad = k
                    break
            if bad is None:
                return
            self._log("si_abort")
            self._weld(bad)
            self.compute_velocities()

    def _verify_ridges(self):
        """a ridge whose plates converge becomes inactive for this step and
        the two plates move together (code: _verifyRidges)"""
        for _ in range(200):
            n = len(self.bounds)
            bad = None
            for k, b in enumerate(self.bounds):
                if b.kind == "R" and b.active and n > 1:
                    if self.plates[(k + 1) % n].U < self.plates[k].U - 1e-18:
                        bad = b
                        break
            if bad is None:
                return
            bad.active = False
            self.compute_velocities()

    # ======================================================================
    # diagnostics and output
    # ======================================================================
    def diagnostics(self):
        """dict of global quantities at the current time.
        urms/urmsL: plate RMS speed, unweighted/length-weighted (cm/yr);
        cont_rms/cont_rmsL: same for continents; flux: subduction area flux
        (km^2/yr); moving: False during single immobile plate phases."""
        n = len(self.bounds)
        cm = self.deg2m * 100.0              # cm per degree
        Ls = np.array([pl.L for pl in self.plates])
        Us = np.array([pl.U for pl in self.plates]) * cm
        urms = float(np.sqrt(np.mean(Us ** 2)))
        urmsL = float(np.sqrt(np.sum(Ls * Us ** 2) / 360.0))
        cw, cu = [], []
        for pl in self.plates:
            for pc in pl.pieces:
                if isinstance(pc, Cont):
                    cw.append(pc.w)
                    cu.append(pl.U * cm)
        cw, cu = np.array(cw), np.array(cu)
        crms = float(np.sqrt(np.mean(cu ** 2))) if len(cu) else 0.0
        crmsL = float(np.sqrt(np.sum(cw * cu ** 2) / cw.sum())) if len(cu) else 0.0
        flux = 0.0
        for k, b in enumerate(self.bounds):
            if b.kind == "T":
                c = self.convergence(k)
                if c > 0:
                    flux += c / 360.0 * self.surf * 1e-6
        nr = sum(1 for b in self.bounds if b.kind == "R")
        nt = sum(1 for b in self.bounds if b.kind == "T")
        return dict(age_Ma=self.age / 1e6, T=self.T, Q=self.heat_flow_TW(),
                    H=self.radiogenic_TW(), n_plates=len(self.plates) if n else 1,
                    n_ridges=nr, n_trenches=nt, n_cont=len(cw), urms=urms,
                    urmsL=urmsL, cont_rms=crms, cont_rmsL=crmsL, flux=flux,
                    mean_L_km=float(Ls.mean() * self.deg2m / 1e3),
                    moving=bool(np.any(np.abs(Us) > 1e-6)))

    def snapshot(self):
        """lightweight copy of the geometry for plotting"""
        xs = self.plate_x()
        plates, oceans, conts = [], [], []
        for i, pl in enumerate(self.plates):
            plates.append((xs[i], pl.L, pl.U * self.deg2m * 100.0))
            x = xs[i]
            for pc in pl.pieces:
                if isinstance(pc, Ocean):
                    oceans.append((x + pc.s, pc.a.copy()))
                else:
                    conts.append((x, pc.w))
                x += pc.w
        bnds = []
        bp = self.bound_pos()
        for k, b in enumerate(self.bounds):
            bnds.append((bp[k], b.kind, b.sub, b.Z, b.active))
        return dict(age_Ma=self.age / 1e6, T=self.T, Q=self.heat_flow_TW(),
                    plates=plates, oceans=oceans, conts=conts, bounds=bnds,
                    p=self.p)

    def check_invariants(self, tol=1e-6):
        """raise AssertionError if the state is inconsistent"""
        Ltot = sum(pl.L for pl in self.plates)
        assert abs(Ltot - 360.0) < tol, f"total length {Ltot}"
        if self.bounds:
            assert len(self.bounds) == len(self.plates)
        else:
            assert len(self.plates) == 1
        for pl in self.plates:
            assert pl.pieces, "empty plate"
            for j, pc in enumerate(pl.pieces):
                if isinstance(pc, Ocean):
                    assert len(pc.s) >= 2 and abs(pc.s[0]) < 1e-12
                    assert np.all(np.diff(pc.s) >= -1e-12), "unsorted nodes"
                    assert np.all(pc.a >= -1e-6), "negative age"
                else:
                    assert pc.w > 0
                    if j > 0:
                        assert not isinstance(pl.pieces[j - 1], Cont), "adjacent continents"
        for b in self.bounds:
            assert b.kind in ("R", "T")
            if b.kind == "T":
                assert b.sub in (-1, 1)
                assert b.Z >= 0

    def run(self, end_age_Ma=0.0, out_every_Myr=1.0, snap_ages=(),
            max_steps=10_000_000, check=False, callback=None):
        """integrate to end_age_Ma (Ma).

        out_every_Myr : interval of the diagnostics time series
        snap_ages     : ages (Ma) at which to store snapshots for plotting
        check         : verify invariants after every step (slow)
        callback      : f(model, diagnostics) called at each output
        returns dict of numpy arrays (keys of diagnostics()) plus 'events',
        'snaps' (dict age -> snapshot) and 'steps'
        """
        hist = []
        snaps = {}
        snap_ages = sorted(snap_ages, reverse=True)
        next_out = self.age / 1e6
        steps = 0
        zero_dt = 0
        while self.age / 1e6 > end_age_Ma and steps < max_steps:
            if self.age / 1e6 <= next_out + 1e-9:
                d = self.diagnostics()
                hist.append(d)
                next_out -= out_every_Myr
                if callback:
                    callback(self, d)
            while snap_ages and self.age / 1e6 <= snap_ages[0] + 1e-9:
                snaps[snap_ages.pop(0)] = self.snapshot()
            dt = self.step()
            zero_dt = zero_dt + 1 if dt < 1e-3 else 0
            if zero_dt > 1000:
                raise RuntimeError("time step stalled at %.3f Ma" % (self.age / 1e6))
            if check:
                self.check_invariants()
            steps += 1
        hist.append(self.diagnostics())
        while snap_ages:
            snaps[snap_ages.pop(0)] = self.snapshot()
        out = {k: np.array([h[k] for h in hist]) for k in hist[0]}
        out["events"] = list(self.events)
        out["snaps"] = snaps
        out["steps"] = steps
        return out


# ---------------------------------------------------------------------------
# initial conditions
# ---------------------------------------------------------------------------
def random_earth(params=None, T0=1875.0, age_Ma=3000.0, n_cont=4, seed=0,
                 min_cont_width=8.0, spread=(2.0, 8.0)):
    """random initial state, an approximation of 2020 section 3.3 (the
    original generator is not part of the released code):

    n_cont continents of random widths (Dirichlet, at least min_cont_width
    deg) cover cont_frac of the surface, separated by ocean basins of random
    widths. Each basin has a ridge at a random position (20-80% of its
    width) and a trench, with probability 0.5, at each continental edge
    (continent overriding; at least one trench in total). Seafloor ages
    increase linearly away from ridges at random half spreading rates drawn
    from `spread` (cm/yr), capped at 0.95 times the critical age. Initial
    slab depths are uniform in 200-1500 km."""
    p = params or Params()
    rng = np.random.default_rng(seed)
    tot_c = 360.0 * p.cont_frac
    while True:
        wc = rng.dirichlet(np.ones(n_cont)) * tot_c
        if wc.min() >= min_cont_width:
            break
    wg = rng.dirichlet(np.ones(n_cont)) * (360.0 - tot_c)
    x = rng.uniform(0, 360)
    conts, gaps = [], []
    for k in range(n_cont):
        conts.append((x % 360.0, wc[k]))
        x += wc[k]
        gaps.append((x, wg[k]))
        x += wg[k]
    tmp = Model(p, 0.0, [Plate([Ocean([0, 360], [0, 0])])], [], T0, age_Ma)
    cap = 0.95 * tmp.tau_sub()
    bounds, ridges = [], []
    trench_any = False
    edges = []
    for g, (g0, gw) in enumerate(gaps):
        r = g0 + gw * rng.uniform(0.2, 0.8)
        ul = rng.uniform(*spread) * 1e-2 / tmp.deg2m      # deg/yr
        uh = rng.uniform(*spread) * 1e-2 / tmp.deg2m
        ridges.append((g0, g0 + gw, r, ul, uh))
        bounds.append(dict(pos=r % 360.0, kind="R"))
        for side in (0, 1):
            if rng.uniform() < 0.5:
                pos = g0 if side == 0 else g0 + gw
                bounds.append(dict(pos=pos % 360.0, kind="T", sub=1 if side == 0 else -1,
                                   Z=rng.uniform(200e3, 1500e3)))
                trench_any = True
            else:
                edges.append((g, side))
    if not trench_any:
        g, side = edges[rng.integers(len(edges))]
        g0, gw = gaps[g]
        pos = g0 if side == 0 else g0 + gw
        bounds.append(dict(pos=pos % 360.0, kind="T", sub=1 if side == 0 else -1,
                           Z=rng.uniform(200e3, 1500e3)))

    def age_fn(xg):
        for g0, g1, r, ul, uh in ridges:
            for k in (-1, 0, 1):
                xx = xg + 360.0 * k
                if g0 - 1e-9 <= xx <= g1 + 1e-9:
                    a = (r - xx) / ul if xx < r else (xx - r) / uh
                    return min(a, cap)
        return 0.0

    return Model.from_layout(p, bounds, conts, age_fn, T0, age_Ma, seed=seed + 1)


def fixed_earth(n_cont=2, params=None, T0=2000.0, age_Ma=1150.0, age_trench=10.0,
                Z0=0.0):
    """symmetric fixed-boundary configuration of 2020 section 3.1 / Fig. 3:
    n_cont equal continents, each on an immobile plate that overrides two
    trenches, separated by two oceanic plates spreading from a central ridge
    (3 n_cont plates). Ages are linear from 0 at ridges to age_trench (Myr)
    at trenches. Boundaries do not move and continents do not break up."""
    p = (params or Params()).copy(fixed=True, F_lim=np.inf)
    wc = 360.0 * p.cont_frac / n_cont
    wo = (360.0 - 360.0 * p.cont_frac) / n_cont
    bounds, conts, ridges = [], [], []
    for k in range(n_cont):
        c0 = k * (wc + wo)
        conts.append((c0, wc))
        g0 = c0 + wc
        r = g0 + 0.5 * wo
        bounds.append(dict(pos=g0, kind="T", sub=1, Z=Z0))
        bounds.append(dict(pos=r, kind="R"))
        bounds.append(dict(pos=g0 + wo, kind="T", sub=-1, Z=Z0))
        ridges.append((g0, g0 + wo, r))

    def age_fn(xg):
        for g0, g1, r in ridges:
            for k in (-1, 0, 1):
                xx = xg + 360.0 * k
                if g0 - 1e-9 <= xx <= g1 + 1e-9:
                    return abs(xx - r) / (0.5 * wo) * age_trench * 1e6
        raise ValueError("position %g not in an ocean gap" % xg)

    return Model.from_layout(p, bounds, conts, age_fn, T0, age_Ma)


def earth_plates_test(params=None, T=1625.0):
    """single-plate speed as in 2020 Fig. 2: returns
    speed(L_km, age_trench_Myr, Z_km, subducting=True, age_rp_Myr=None) in
    deg/Myr for a plate bounded by a ridge and a trench. For an overriding
    plate, age and slab depth are those of the plate it overrides, and
    age_rp_Myr is its own maximum oceanic age (ridge push)."""
    p = params or Params()

    def speed(L_km, age_tr, Z_km, subducting=True, age_rp=None):
        m = Model(p, 0.0, [Plate([Ocean([0, 360], [0, 0])])], [], T, 0.0)
        b = Boundary("T", sub=-1 if subducting else 1, Z=Z_km * 1e3)
        H = m._H(age_tr * 1e6)
        b.prof_z, b.prof_h = np.array([b.Z]), np.array([H])
        SP, SS, eVS, eB = m._trench_terms(b, H)
        age_rp = age_tr if age_rp is None else age_rp
        F = m._rp(age_rp * 1e6) + SS + (SP if subducting else 0.0)
        U = F / (m._eta_H(L_km * 1e3 / m.deg2m) + eVS + eB)
        return U * SEC_YR * 1e6 / m.deg2m
    return speed
