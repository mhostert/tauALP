import numpy as np
import pandas as pd

from DarkNews import const
from DarkNews import Cfourvec as Cfv

from . import models


# Experiments and their parameters
tau_per_POT = 5e-7
ICARUS_exp = {
    "name": "ICARUS",
    "L": 803e2,
    "theta0": 0.0968,
    "dX": 2 * 2.67e2,
    "dY": 2.86e2,
    "dZ": 17.00e2,
    "norm": tau_per_POT * 4.2e21,
    "Emin": 0.1,
    "final_states": ["ee", "em", "me", "mm"],
}
MicroBooNE_exp = {
    "name": "MicroBooNE",
    "L": 685e2,
    "theta0": 0.146,
    "dX": 2.26e2,
    "dY": 2.03e2,
    "dZ": 9.42e2,
    "norm": tau_per_POT * 4.2e21,
    "Emin": 0.1,
    "final_states": ["ee", "em", "me", "mm"],
}
NoVA_exp = {
    "name": "NOvA",
    "L": 990e2,
    "theta0": 14.6e-3,
    "dX": 3.9e2,
    "dY": 3.9e2,
    "dZ": 12.7e2,
    "norm": tau_per_POT * 4.2e21,
    "Emin": 0.1,
    "final_states": ["ee", "em", "me", "mm"],
}
tau_per_POT = 2.7e-6
CHARM_exp = {
    "name": "CHARM",
    "L": 480e2,
    "theta0": np.arctan(5e2 / 480e2),
    "dX": 3e2,
    "dY": 3e2,
    # "R": 1.5e2,
    "dZ": 35e2,
    "norm": tau_per_POT * 2.4e18,
    "Emin": 1,
    "final_states": ["ee", "em", "me", "mm"],
}
BEBC_exp = {
    "name": "BEBC",
    "L": 404e2,
    "theta0": 0,
    "dX": 3.57e2,
    "dY": 2.52e2,
    "dZ": 1.85e2,
    "norm": tau_per_POT * 2.72e18,
    "Emin": 0 * 1,
    "final_states": ["ee", "em", "me", "mm"],
}
NA62_exp = {
    "name": "NA62",
    "L": 79.4e2,
    "theta0": 0,
    "dX": 2e2,
    "dY": 2e2,
    "dZ": 78e2,
    "norm": tau_per_POT * 1.4e17,
    "Emin": 1,
    "final_states": ["ee", "mm"],
}
PROTO_DUNE_NP02_exp = {
    "name": "ProtoDUNE-NP02",
    "L": 677e2,
    "x0": np.arctan(3e2 / 677e2),
    "y0": np.arctan(1.5e2 / 677e2),
    "dX": 6e2,
    "dY": 7e2,
    "dZ": 6e2,
    "norm": tau_per_POT * 1.75e19,
    "Emin": 1,
    "final_states": ["ee", "em", "me", "mm"],
}
PROTO_DUNE_NP04_exp = {
    "name": "ProtoDUNE-NP04",
    "L": 723e2,
    "x0": np.arctan(3e2 / 677e2),
    "y0": np.arctan(1.5e2 / 677e2),
    "dX": 6e2,
    "dY": 7e2,
    "dZ": 6e2,
    "norm": tau_per_POT * 1.75e19,
    "Emin": 1,
    "final_states": ["ee", "em", "me", "mm"],
}
SHiP_exp = {
    "name": "SHiP",
    "L": 33.7e2,
    "theta0": 0,
    "dX": 2.5e2,
    "dY": 4.3e2,
    "dZ": 49.6e2,
    "norm": tau_per_POT * 6e20,
    "Emin": 1,
    "final_states": ["ee", "em", "me", "mm"],
}

sigma_tau = 25.95e-30  # cm^2
FASER_exp = {
    "name": "FASER",
    "L": 480e2,
    "theta0": np.arctan(6.5 / 480e2),
    "R": 0.1e2,
    "dZ": 1.5e2,
    "norm": 150e39 * sigma_tau,
    "Emin": 100,
    "final_states": ["ee", "em", "me", "mm"],
}
FASER2_exp = {
    "name": "FASER2",
    "L": 480e2,
    "theta0": 0,
    "R": 1e2,
    "dZ": 5e2,
    "norm": 3e42 * sigma_tau,
    "Emin": 100,
    "final_states": ["ee", "em", "me", "mm"],
}


def load_events(file_paths):
    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # Iterate over each file path, read the CSV file, and concatenate it to the existing DataFrame
    for file_path in file_paths:
        df_new = pd.read_csv(file_path, sep=" ").shift(axis=1).iloc[:, 1:]
        df = pd.concat([df, df_new], ignore_index=True)

    return df


class Experiment:
    def __init__(self, file_paths, exp_dic, alp=None):
        self.df_taus = load_events(file_paths)
        self.df_taus["weight"] = self.df_taus.weight / self.df_taus.weight.sum()

        # All experimental attributes
        required_keys = ["name", "L", "dZ", "norm", "Emin", "final_states"]
        for key, value in exp_dic.items():
            try:
                required_keys.pop(required_keys.index(key))
            except ValueError:
                pass
            setattr(self, key, value)
        if required_keys:
            raise ValueError(f"Missing keys: {required_keys}")

        if "R" in exp_dic:
            self.R = exp_dic["R"]
            self.dtheta = np.arctan(self.R / self.L)
            self.dphi = np.arctan(self.R / self.L)
        else:
            self.dX = exp_dic["dX"]
            self.dY = exp_dic["dY"]
            self.dtheta = np.arctan(self.dX / self.L)
            self.dphi = np.arctan(self.dY / self.L)

        if "theta0" in exp_dic:
            # detector center at the plane of the detector
            self.x0 = self.L * np.tan(self.theta0)
            self.y0 = 0
        else:
            # detector center at the plane of the detector
            self.x0 = exp_dic["x0"]
            self.y0 = exp_dic["y0"]

        if alp is not None:
            self.alp = alp
        else:
            self.alp = models.ALP(0.5, 1e-5)

        self.get_event_rate(self.alp)

    def get_alp_events_exclusive(self, alp, channel="e"):
        """
        Get ALP events for a given tau decay channel
        """

        self.nevents = len(self.df_taus)
        phi_alp = np.random.rand(self.nevents) * 2 * np.pi
        ctheta_alp = 2 * np.random.rand(self.nevents) - 1

        m_lepton = const.m_e if channel == "e" else const.m_mu if channel == "mu" else 0
        ECM_alp = (
            (const.m_tau**2 - m_lepton**2 + alp.m_a**2)
            / 2
            / const.m_tau
            * np.ones(self.nevents)
        )
        pCM_alp = np.sqrt(ECM_alp**2 - alp.m_a**2)

        p4_alp_CM = Cfv.build_fourvec(ECM_alp, pCM_alp, ctheta_alp, phi_alp)
        self.df_taus["p"] = np.sqrt(
            self.df_taus.px**2 + self.df_taus.py**2 + self.df_taus.pz**2
        )
        ctheta_tau_LAB = (self.df_taus.pz / self.df_taus.p).to_numpy()
        phitau_LAB = np.arctan2(self.df_taus.py, self.df_taus.px).to_numpy()
        beta = -(self.df_taus.p / self.df_taus.E).to_numpy()
        beta[beta < -1] = -1
        self.p4_alp = Cfv.Tinv(
            p4_alp_CM,
            beta,
            ctheta_tau_LAB,
            phitau_LAB,
        )

        self.weights = self.df_taus["weight"]
        self.weights = (
            self.weights
            if channel == "e"
            else (
                self.weights * alp.BR_tau_to_a_mu() / alp.BR_tau_to_a_e()
                if channel == "mu"
                else 0
            )
        )

        return self.p4_alp, self.weights

    def get_alp_events(self, alp):
        """
        Get ALP events for ALL tau decay channels
        """

        # Two branching ratios
        if alp.BR_tau_to_a_e() > 0 and alp.BR_tau_to_a_mu() > 0:
            p4_e, w_e = self.get_alp_events_exclusive(alp, channel="e")
            p4_mu, w_mu = self.get_alp_events_exclusive(alp, channel="mu")

            self.p4_alp = np.append(p4_e, p4_mu, axis=0)
            self.weights = np.append(w_e, w_mu)
            del w_e, w_mu, p4_mu, p4_e

        elif alp.BR_tau_to_a_e() > 0 and alp.BR_tau_to_a_mu() == 0:
            self.p4_alp, self.weights = self.get_alp_events_exclusive(alp, channel="e")

        elif alp.BR_tau_to_a_e() == 0 and alp.BR_tau_to_a_mu() > 0:
            self.p4_alp, self.weights = self.get_alp_events_exclusive(alp, channel="mu")

        # projected x,y at the plane of the detector
        self.p_alp = np.sqrt(self.p4_alp[:, 0] ** 2 - alp.m_a**2)
        self.x_alp = self.p4_alp[:, 1] / self.p_alp * self.L
        self.y_alp = self.p4_alp[:, 2] / self.p_alp * self.L

        return self.p4_alp, self.weights

    def get_alp_spectrum(self, alp, selection=True, bins=40, generate_events=True):
        """Get histogram of ALP momenta produced in a given tau decay channel

                * samples 4 momenta for tau decays to ALPs
                * calculates the acceptance of the detector for the ALP
                * returns the histogram of ALP momenta, the bin edges, and the geom acceptance


        Args:
            m_alp (_type_): _description_

        Returns:
            _type_: _description_
        """

        if generate_events:
            # Get ALP events for all channels
            self.p4_alp, self.weights = self.get_alp_events(alp)

        if hasattr(self, "theta0"):
            theta_alp = np.arccos(Cfv.get_cosTheta(self.p4_alp))
            self.mask_alp_in_acc = (self.theta0 - self.dtheta / 2 < theta_alp) & (
                theta_alp < self.theta0 + self.dtheta / 2
            )
            signal_selection = self.p4_alp[:, 0] > self.Emin
            self.eff = (
                self.dphi
                / np.pi
                * self.weights[signal_selection].sum()
                / self.weights.sum()
            )
        else:
            # NOTE: Selection of events in a square
            # NOTE: Assume a cuboid... need to extend for SHiP
            if hasattr(self, "R"):
                self.mask_alp_in_acc = (
                    (self.x_alp - self.x0) ** 2 + (self.y_alp - self.y0) ** 2
                ) < self.R**2
            else:
                self.mask_alp_in_acc = (np.abs(self.x_alp - self.x0) < self.dX / 2) & (
                    np.abs(self.y_alp - self.y0) < self.dY / 2
                )
            signal_selection = self.p4_alp[:, 0] > self.Emin
            self.eff = self.weights[signal_selection].sum() / self.weights.sum()

        # If less than 3 generated events were within acceptance, dont even try to compute rate
        if self.mask_alp_in_acc.sum() < 3:
            if isinstance(bins, int):
                p_bins = np.linspace(self.p_alp.min(), self.p_alp.max(), bins)
            elif isinstance(bins, np.ndarray):
                p_bins = bins
            self.geom_acceptance = 0
            return np.zeros(len(p_bins) - 1), p_bins

        # Else, histogram event rate in alp energy
        else:
            if isinstance(bins, int):
                p_bins = np.linspace(self.p_alp.min(), self.p_alp.max(), bins)
            elif isinstance(bins, np.ndarray):
                p_bins = bins

            if selection:
                h, p_bins = np.histogram(
                    self.p_alp[self.mask_alp_in_acc],
                    bins=p_bins,
                    weights=self.weights[self.mask_alp_in_acc],
                )
            else:
                h, p_bins = np.histogram(
                    self.p_alp,
                    bins=p_bins,
                    weights=self.weights,
                )
            self.geom_acceptance = (
                self.weights[self.mask_alp_in_acc].sum() / self.weights.sum()
            )

            return h / h.sum(), p_bins

    def get_event_rate(self, alp, selection=True):

        self.dPhidp, self.palp = self.get_alp_spectrum(alp, selection=selection)

        if np.array(self.palp).size > 1:
            self.dp = np.diff(self.palp)
            self.pc = self.palp[:-1] + self.dp / 2

            if selection:
                self.flux = (
                    alp.BR_tau_to_a_e() * self.norm * self.geom_acceptance * self.eff
                )
            else:
                self.flux = alp.BR_tau_to_a_e() * self.norm
            return self.flux * np.sum(
                self.dPhidp
                * alp.prob_decay(self.pc, self.L, self.dZ)
                * alp.visible_BR(self.final_states)
            )
        else:
            self.flux = (
                alp.BR_tau_to_a_e()
                * self.dPhidp
                * self.norm
                * self.geom_acceptance
                * self.eff
            )
            return (
                self.flux
                * alp.prob_decay(self.palp, self.L, self.dZ)
                * alp.visible_BR(self.final_states)
            )

    def reweight(self, alp1, alp2):

        new_rate = self.flux * (
            alp2.BR_tau_to_a_e()
            / alp1.BR_tau_to_a_e()
            * np.sum(
                self.dPhidp
                * alp2.prob_decay(self.pc, self.L, self.dZ)
                * alp2.visible_BR(self.final_states)
            )
        )
        return new_rate

    # def get_tau_events(df):
    #     tau_events = np.abs(df.pid) == 15
    #     df["p"] = np.sqrt(df.px**2 + df.py**2 + df.pz**2)
    #     nevents = tau_events.sum()
    #     df_taus = df[tau_events].reset_index(drop=True, inplace=False)

    #     m_alp = 0.5

    #     # Decay tau+/- to alp
    #     phi_alp = np.random.rand(nevents) * 2 * np.pi
    #     ctheta_alp = 2 * np.random.rand(nevents) - 1
    #     stheta_alp = np.sqrt(1 - ctheta_alp**2)
    #     ECM_alp = (
    #         (const.m_tau**2 - const.m_mu**2 + m_alp**2)
    #         / 2
    #         / const.m_tau
    #         * np.ones(nevents)
    #     )
    #     pCM_alp = np.sqrt(ECM_alp**2 - m_alp**2)

    #     p4_alp_CM = Cfv.build_fourvec(ECM_alp, pCM_alp, ctheta_alp, phi_alp)

    #     ctheta_tau_LAB = (df_taus.pz / df_taus.p).to_numpy()
    #     phitau_LAB = np.arctan2(df_taus.py, df_taus.px).to_numpy()
    #     p4_alp = Cfv.Tinv(
    #         p4_alp_CM, -(df_taus.p / df_taus.E).to_numpy(), ctheta_tau_LAB, phitau_LAB
    #     )

    #     theta_alp_deg = np.arccos(Cfv.get_cosTheta(p4_alp))
    #     phi_alp_deg = np.arctan2(p4_alp[:, 2], p4_alp[:, 1])
