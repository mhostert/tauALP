import pandas as pd
import numpy as np

from scipy.stats import gamma

from DarkNews import Cfourvec as Cfv
from . import const
from . import phase_space as ps

xsec_ccbar = {
    "120GeV": 2.2e-6,
    "400GeV": 20e-6,
    "13.6TeV": 0.5e-6,
}  # in cm^2
frag_fractions = {"Ds+": 0.081, "D+": 0.244, "D0": 0.606, "D-": 0.244}


def get_xF(p4, p_beam, CoM=False):
    n_events = np.shape(p4)[0]
    p4_beam = np.zeros((n_events, 4))
    p4_beam[:, 0] = np.ones(n_events) * np.sqrt(const.m_proton**2 + p_beam**2)
    p4_beam[:, 3] = np.ones(n_events) * p_beam
    if CoM:
        p4_CM = p4
        p4_beam_CM = p4_beam
    else:
        # Boost to the center of mass frame
        beta = (
            p_beam
            / (np.sqrt(p_beam**2 + const.m_proton**2) + const.m_proton)
            * np.ones(n_events)
        )
        p4_CM = Cfv.L(p4, beta)
        p4_beam_CM = Cfv.L(p4_beam, beta)

    return p4_CM[:, -1] / p4_beam_CM[:, -1]


def get_pTSQR(p4):
    pT = Cfv.getXYnorm(p4)
    return pT**2


def get_pT(p4):
    pT = Cfv.getXYnorm(p4)
    return pT


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


def generate_taus_with_custom_method(
    params,
    p_beam=120,
    pT_max=30.0,
    CoM=False,
    as_dataframe=False,
    pid=15,
    n_events=1000,
    cone_force_acceptance=None,
    n_trials=None,
):
    """
    Generate n 4-momenta of Ds mesons using the provided differential cross section shape.

        cone_force_acceptance: [x0, y0, z0, R]
    Returns:
    - List of 4-momenta [E, px, py, pz]
    """
    n_events = int(n_events)

    n_generated = 0
    p4_tau = np.empty((0, 4))
    weights = np.empty(0)
    w_total = 0
    if n_trials is None:
        n_trials = n_events

    while n_generated < n_events:
        w_trial = np.ones(n_trials)

        # sample xF from a log-uniform distribution
        log10_x_F_abs = np.random.uniform(-4, 1, size=n_trials)
        x_F = 10**log10_x_F_abs * np.random.choice(
            [-1, 1], size=n_trials
        )  # Randomly assign sign to xF

        # final desired pdf for xF
        x_F_pdf_final = (
            params["r_1"] * np.exp(-params["a_1"] * abs(x_F) ** params["n_1"])
            + params["r_2"] * np.exp(-params["a_2"] * abs(x_F) ** params["n_2"])
            + params["r_3"] * np.exp(-params["a_3"] * abs(x_F) ** params["n_3"])
        )
        x_F_pdf_final /= np.trapz(x_F_pdf_final, x_F)  # normalize the pdf

        w_trial *= x_F_pdf_final * abs(x_F)

        # p_T = np.random.gamma(shape=params["mu1"], scale=params["lambda1"], size=n_trials)
        log10_p_T = np.random.uniform(-4, np.log10(pT_max), size=n_trials)
        p_T = 10**log10_p_T  # sample pT from a log-uniform distribution

        p_T_pdf_final = params["g_1"] * gamma.pdf(
            p_T ** params["m_1"],
            a=params["mu_1"],
            scale=params["lambda_1"],
        ) + params["g_2"] * gamma.pdf(
            p_T ** params["m_2"], a=params["mu_2"], scale=params["lambda_2"]
        )
        p_T_pdf_final /= np.trapz(p_T_pdf_final, p_T)  # normalize the pdf

        w_trial *= p_T_pdf_final * p_T

        # Putting together the 4-momenta
        phi = np.random.uniform(0, 2 * np.pi, size=n_trials)
        if CoM:
            sqrt_s = 2 * p_beam
        else:
            sqrt_s = np.sqrt(
                2 * np.sqrt(p_beam**2 + const.m_proton**2) * const.m_proton
            )
        pz_CM_max = sqrt_s / 2
        pz = x_F * pz_CM_max
        px = p_T * np.cos(phi)
        py = p_T * np.sin(phi)
        E = np.sqrt(px**2 + py**2 + pz**2 + const.m_tau**2)
        p4_CM = np.column_stack((E, px, py, pz))

        if CoM:
            p4_lab = p4_CM

        else:
            beta = (
                p_beam
                / (np.sqrt(p_beam**2 + const.m_proton**2) + const.m_proton)
                * np.ones(n_trials)
            )
            p4_lab = Cfv.L(p4_CM, -beta)

        # If force events in acceptance, count only accepted ones
        if isinstance(cone_force_acceptance, list):
            x0, y0, L, R = cone_force_acceptance
            v_tau = Cfv.get_3direction(p4_lab)
            x_tau_p0 = v_tau[:, 0] * L
            y_tau_p0 = v_tau[:, 1] * L

            accepted = (x_tau_p0 - x0) ** 2 + (y_tau_p0 - y0) ** 2 < R**2
            n_generated += accepted.sum()
            w_total += w_trial.sum()

        else:
            accepted = np.full(n_trials, True)
            n_generated += n_trials
            w_total += w_trial.sum()

        weights = np.concatenate([weights, w_trial[accepted]])
        p4_tau = np.concatenate([p4_tau, p4_lab[accepted]])
        w_accepted = weights.sum()

        efficiency = w_accepted / w_total

        if accepted.sum() > n_events:
            weights = weights[:n_events]
            p4_tau = p4_tau[:n_events]

    if as_dataframe:
        df = pd.DataFrame(p4_tau, columns=["E", "px", "py", "pz"])
        weights = weights / np.sum(weights) / 2  # normalize the weights
        df["weights"] = weights * efficiency
        df["pid"] = pid

        return df
    else:
        weights = weights / np.sum(weights) / 2  # normalize the weights
        return p4_tau, weights * efficiency


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
        df["mother_pid"] = np.zeros(len(p4_tau), dtype=int)
        df.loc[: len(p4_Dsp), "mother_pid"] = 431
        df.loc[len(p4_Dsp) :, "mother_pid"] = -431
        df["pid"] = np.zeros(len(p4_tau), dtype=int)
        df.loc[: len(p4_Dsp), "pid"] = 15
        df.loc[len(p4_Dsp) :, "pid"] = -15

        return df
    else:
        return p4_tau, w
