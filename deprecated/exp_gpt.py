from __future__ import annotations

import glob
import os
from functools import cached_property
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from DarkNews import Cfourvec as Cfv  # type: ignore
from . import const, models
from .exp_dicts import EXPERIMENTS

# ----------------------------------------------------------------------
#  Cross‑sections & experiment‑wide constants
# ----------------------------------------------------------------------

JPARC_TAU_PER_POT: float = 4e-9
NUMI_TAU_PER_POT: float = 2.9e-7
SPS_TAU_PER_POT: float = 1.7e-6

XSEC_HARD = {"120GeV": 0.000677308, "400GeV": 0.0168063, "13.6TeV": 0.0}
XSEC_SOFT = {"120GeV": 38.4539, "400GeV": 39.8586, "13.6TeV": 102.253}
XSEC_TOT = {k: XSEC_HARD[k] + XSEC_SOFT[k] for k in XSEC_HARD}

# ----------------------------------------------------------------------
#  Lightweight Pythia‑file reader & event‑loader
# ----------------------------------------------------------------------


def _read_pythia(path: str, cols_needed: Sequence[str]) -> pd.DataFrame:
    """Parse a single Pythia8 ASCII file, keeping only *cols_needed*.

    All floating columns are returned as float32
    """

    # find header length (≤ 8 lines) by peeking the file start
    with open(path, "r", encoding="utf-8") as fh:
        header_len = sum(1 for _ in range(8) if fh.readline().startswith("#"))

    df = pd.read_csv(
        path,
        sep=r"\s+",
        # comment="#",
        skiprows=header_len,
        usecols=list(cols_needed),
        dtype={c: "float32" for c in cols_needed},
        engine="c",
    )
    return df


def load_events(file_pattern: str | Iterable[str]) -> pd.DataFrame:
    """Concatenate events from one or several Pythia8 files.

    *file_pattern* can be a glob ("/path/run_*") or an explicit iterable of
    paths. The function returns a *float32* DataFrame with normalised weights.
    """

    if isinstance(file_pattern, str):
        files = glob.glob(f"{file_pattern}_*.txt")
    else:
        files = [p for fp in file_pattern for p in glob.glob(f"{fp}_*.txt")]

    if not files:
        raise FileNotFoundError(f"No Pythia files found for pattern {file_pattern!r}")

    keep = ("E", "px", "py", "pz", "weights")
    frames = [_read_pythia(f, keep) for f in files]
    df = pd.concat(frames, ignore_index=True, copy=False)

    # normalise weights with float‑64 precision (weights column kept as f8)
    df["weights"] = df["weights"].astype("float64")
    df["weights"] /= df["weights"].sum()
    return df


# ----------------------------------------------------------------------
#  Core class – memory‑optimised version
# ----------------------------------------------------------------------


class Experiment:
    """Physics‑level container using ≈⅓ of the RAM of the original version."""

    __slots__ = (
        # detector spec
        "name",
        "L",
        "dZ",
        "norm",
        "Emin",
        "final_states",
        "R",
        "dX",
        "dY",
        "x0",
        "y0",
        "active_volume",
        "dtheta",
        "dphi",
        # tau sample
        "_p4_taus",
        "_weights",
        "nevents",
        # ALP caches
        "alp",
        "tau_BRs",
        "_p4_alp",
        "_p4_d1",
        "_p4_d2",
        "_mask",
    )

    # ------------------------------------------------------------------
    #  Construction & data storage
    # ------------------------------------------------------------------
    def __init__(
        self,
        pythia_pattern: str | Iterable[str],
        exp_cfg: dict,
        *,
        alp: models.ALP | None = None,
        duplicate_taus: int | None = None,
    ) -> None:
        # ingest tau sample ------------------------------------------------
        df = load_events(pythia_pattern)
        if duplicate_taus:
            df = pd.concat([df] * duplicate_taus, ignore_index=True, copy=False)

        self.nevents = len(df)
        self._weights = df["weights"].to_numpy("float64", copy=False)
        self._p4_taus = df[["E", "px", "py", "pz"]].to_numpy("float32", copy=False)
        del df  # free DataFrame early

        # detector geometry ------------------------------------------------
        required = {"L", "dZ", "norm", "Emin", "final_states"}
        missing = required - exp_cfg.keys()
        if missing:
            raise ValueError(f"exp_cfg missing keys: {missing}")

        # copy config into attributes
        for k, v in exp_cfg.items():
            setattr(self, k, v)
        if not hasattr(self, "name"):
            self.name = "unnamed"

        # derived angles
        if hasattr(self, "R"):
            self.dtheta = self.dphi = np.arctan(self.R / self.L)
        else:
            self.dtheta = np.arctan(self.dX / self.L)
            self.dphi = np.arctan(self.dY / self.L)

        # centre fallback
        self.x0 = getattr(self, "x0", 0.0)
        self.y0 = getattr(self, "y0", 0.0)

        # physics ----------------------------------------------------------
        self.alp = alp or models.ALP(0.5, 1e5)
        self.tau_BRs: dict[str, float] = {}
        self._p4_alp = self._p4_d1 = self._p4_d2 = self._mask = None

    # ------------------------------------------------------------------
    #  Lightweight accessors & cached views
    # ------------------------------------------------------------------
    @cached_property
    def p_taus(self) -> np.ndarray:  # |p| in LAB
        return Cfv.get_3vec_norm(self._p4_taus)

    @cached_property
    def ctheta_tau_lab(self) -> np.ndarray:
        return Cfv.get_cosTheta(self._p4_taus)

    @cached_property
    def phi_tau_lab(self) -> np.ndarray:
        return np.arctan2(self._p4_taus[:, 2], self._p4_taus[:, 1])

    # ------------------------------------------------------------------
    #  Vectorised ALP generation (single prod+decay for clarity)
    # ------------------------------------------------------------------
    def _generate_alp(self, prod="tau>mu+a", decay="mm") -> None:
        """Populate caches with ALP + daughter kinematics + event weights."""
        N = self.nevents
        a = self.alp

        # isotropic ALP in tau rest frame ---------------------------------
        phi = np.random.uniform(0, 2 * np.pi, N)
        cth = np.random.uniform(-1, 1, N)
        Ea = a.sample_Ea(prod, size=N).astype("float32")
        pa = np.sqrt(np.maximum(Ea**2 - a.m_a**2, 0)).astype("float32")
        p4_cm = Cfv.build_fourvec(Ea, pa, cth, phi)

        beta = -self.p_taus / self._p4_taus[:, 0]
        p4_alp = Cfv.Tinv(p4_cm, beta, self.ctheta_tau_lab, self.phi_tau_lab)
        self._p4_alp = p4_alp.astype("float32", copy=False)

        # daughters --------------------------------------------------------
        self._p4_d1, self._p4_d2 = self._sample_daughters(decay)

        # weights ----------------------------------------------------------
        self.tau_BRs[prod] = a.tau_BR(prod)
        self._weights = self._weights * self.tau_BRs[prod]

    def _sample_daughters(self, decay: str):
        a = self.alp
        N = self.nevents
        phi = np.random.uniform(0.0, 2 * np.pi, N)
        cth = np.random.uniform(-1.0, 1.0, N)

        m1, m2 = (models.LEPTON_MASSES[models.LEPTON_INDEX[d]] for d in decay)
        E1 = (a.m_a**2 + m1**2 - m2**2) / (2 * a.m_a)
        E2 = (a.m_a**2 - m1**2 + m2**2) / (2 * a.m_a)
        p = np.sqrt(max(E1**2 - m1**2, 0.0))

        p4_1_cm = Cfv.build_fourvec(E1, p, cth, phi)
        p4_2_cm = Cfv.build_fourvec(E2, -p, cth, phi)

        beta = -np.sqrt(1.0 - (a.m_a / self._p4_alp[:, 0]) ** 2)
        beta = np.clip(beta, -1.0, 0.0)

        ct_a = Cfv.get_cosTheta(self._p4_alp)
        phi_a = np.arctan2(self._p4_alp[:, 2], self._p4_alp[:, 1])

        p4_1 = Cfv.Tinv(p4_1_cm, beta, ct_a, phi_a)
        p4_2 = Cfv.Tinv(p4_2_cm, beta, ct_a, phi_a)
        return (
            p4_1.astype("float32", copy=False),
            p4_2.astype("float32", copy=False),
        )

    # ------------------------------------------------------------------
    #  Public interface -------------------------------------------------
    # ------------------------------------------------------------------
    def event_rate(self, *, regenerate: bool = False) -> float:
        """Return the expected visible‑signal yield for the configured ALP."""
        if regenerate or self._mask is None:
            self._generate_alp()
            self._mask = self._geom_mask()

        flux = self.norm * self._weights[self._mask]
        vis_br = self.alp.alp_visible_BR(self.final_states)
        prob = self.alp.prob_decay(self._p4_alp[self._mask, 0], self.L, self.dZ)
        return float(np.sum(flux * vis_br * prob))

    # ------------------------------------------------------------------
    #  Geometry & selection helpers ------------------------------------
    # ------------------------------------------------------------------
    def _geom_mask(self):
        N = self.nevents
        z = np.random.uniform(self.L, self.L + self.dZ, N)
        vx, vy = (self._p4_d1[:, i] / self._p4_d1[:, 0] for i in (1, 2))
        x1 = vx * z
        y1 = vy * z

        if hasattr(self, "R"):
            r2 = (x1 - self.x0) ** 2 + (y1 - self.y0) ** 2
            geom = r2 < self.R**2
        else:
            geom = (np.abs(x1 - self.x0) < self.dX / 2) & (
                np.abs(y1 - self.y0) < self.dY / 2
            )

        energy_ok = self._p4_d1[:, 0] > self.Emin / 2
        return geom & energy_ok

    # ------------------------------------------------------------------
    #  Minimal public access to cached heavy arrays --------------------
    # ------------------------------------------------------------------
    @property
    def p4_alp(self):
        return self._p4_alp

    @property
    def p4_daughter1(self):
        return self._p4_d1

    @property
    def p4_daughter2(self):
        return self._p4_d2

    @property
    def mask_in_acc(self):
        return self._mask


# ----------------------------------------------------------------------
#  Convenience: factory that builds an Experiment by name
# ----------------------------------------------------------------------


def from_name(name: str, pythia_pattern: str | Iterable[str], **kwargs) -> Experiment:
    """Instantiate an :class:`Experiment` from *EXPERIMENTS* by key."""
    if name not in EXPERIMENTS:
        raise KeyError(
            f"Unknown experiment name {name!r}. Available: {list(EXPERIMENTS)}"
        )
    cfg = {"name": name, **EXPERIMENTS[name]}
    return Experiment(pythia_pattern, cfg, **kwargs)


# ----------------------------------------------------------------------
#  Module export control
# ----------------------------------------------------------------------
__all__ = [
    "Experiment",
    "from_name",
    "load_events",
    "EXPERIMENTS",
]
