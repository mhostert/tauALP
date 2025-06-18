# Convert to DataFrame if requested
import pandas as pd
import numpy as np
from DarkNews import Cfourvec as Cfv
from . import const
from . import phase_space as ps

xsec_ccbar = {
    "120GeV": 2.2e-6,
    "400GeV": 20e-6,
    "13.6TeV": 0.5e-6,
}  # in cm^2
frag_fractions = {"Ds+": 0.081, "D+": 0.244, "D0": 0.606, "D-": 0.244}


def generate_Ds(p_beam=120, n_events=1000, a=2.0, b=1.0, n_exp=5):
    """
    Generate n 4-momenta of Ds mesons using the provided differential cross section shape.

    Parameters:
    - p_beam: beam energy in GeV
    - n_events: number of momenta to generate
    - a, b, n_exp: parameters of the differential cross-section for D production

    Returns:
    - List of 4-momenta [E, px, py, pz]
    """
    n_events = int(n_events)

    # u = np.random.uniform(0, 1, size=n_events)
    # xF_abs = 1 - u**(1 / (n_exp + 1))  # Inverse CDF of (1 - x)^n
    # x_F = xF_abs * np.random.choice([-1, 1], size=n_events)  # Randomly assign sign to xF

    x_F = np.random.uniform(-1, 1, size=n_events)

    p_T2 = np.random.exponential(
        scale=b, size=n_events
    )  # sample pT^2 from a rough expected distribution
    p_T = np.sqrt(p_T2)

    # Putting together the 4-momenta
    phi = np.random.uniform(0, 2 * np.pi, size=n_events)
    sqrt_s = np.sqrt(2 * np.sqrt(p_beam**2 + const.m_proton**2) * const.m_proton)
    pz_CM_max = sqrt_s / 2
    pz = x_F * pz_CM_max
    px = p_T * np.cos(phi)
    py = p_T * np.sin(phi)
    E = np.sqrt(px**2 + py**2 + pz**2 + const.m_charged_Ds**2)
    p4_CM = np.column_stack((E, px, py, pz))
    beta = (
        p_beam
        / (np.sqrt(p_beam**2 + const.m_proton**2) + const.m_proton)
        * np.ones(n_events)
    )
    p4 = Cfv.L(p4_CM, -beta)
    # Weight from the given differential cross section (unnormalized)
    weight = (1 - abs(x_F)) ** n_exp * np.exp(-(a * p_T + b * p_T2))

    return p4, weight / np.sum(weight)  # normalize the weights


def generate_taus(p_beam=120, n_events=1000, a=2.0, b=1.0, n_exp=5, as_dataframe=False):
    """
    Generate n 4-momenta of tau leptons using the provided differential cross section shape.

    Parameters:
    - p_beam: beam energy in GeV
    - n_events: number of momenta to generate
    - a, b, n_exp: parameters of the differential cross-section for D production

    Returns:
    - List of 4-momenta [E, px, py, pz]
    """

    # Generate Ds+ and Ds- momenta (so far they're the same)
    p4_Dsp, wp = generate_Ds(p_beam, n_events, a, b, n_exp)
    p4_Dsm, wm = generate_Ds(p_beam, n_events, a, b, n_exp)

    # concatenate
    p4_Ds = np.concatenate((p4_Dsp, p4_Dsm), axis=0)
    w = np.concatenate((wp, wm), axis=0)

    # Generate tau momenta from Ds momenta
    p4_tau = ps.decay_2_body(p4_Ds, const.m_charged_Ds, const.m_tau, 0.0)
    w = w * frag_fractions["Ds+"]

    if as_dataframe:
        df = pd.DataFrame(p4_tau, columns=["E", "px", "py", "pz"])
        df["weights"] = w
        df["E_mother"] = p4_Ds[:, 0]
        df["px_mother"] = p4_Ds[:, 1]
        df["py_mother"] = p4_Ds[:, 2]
        df["pz_mother"] = p4_Ds[:, 3]
        df["mother_pid"] = 431  # D+ meson
        df["pid"] = 15

        return df
    else:
        return p4_tau, w
