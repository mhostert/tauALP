import numpy as np
import pandas as pd

from DarkNews import const
from DarkNews import Cfourvec as Cfv


def load_events(file_paths):
    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # Iterate over each file path, read the CSV file, and concatenate it to the existing DataFrame
    for file_path in file_paths:
        df_new = pd.read_csv(file_path, sep=" ").shift(axis=1).iloc[:, 1:]
        df = pd.concat([df, df_new], ignore_index=True)

    return df


class Experiment:
    def __init__(self, file_paths, name, L, theta0, dX, dY, dZ, norm):
        self.df_taus = load_events(file_paths)
        self.name = name
        self.L = L
        self.theta0 = theta0
        self.dX = dX
        self.dY = dY
        self.dZ = dZ
        self.norm = norm

        self.dtheta = np.arctan(dX / L)
        self.dphi = np.arctan(dY / L)

    def get_alp_spectrum(self, m_alp):
        """Get histogram of ALP momenta produced in a given tau decay channel

                * samples 4 momenta for tau decays to ALPs
                * calculates the acceptance of the detector for the ALP
                * returns the histogram of ALP momenta, the bin edges, and the geom acceptance


        Args:
            m_alp (_type_): _description_

        Returns:
            _type_: _description_
        """

        nevents = len(self.df_taus)
        phi_alp = np.random.rand(nevents) * 2 * np.pi
        ctheta_alp = 2 * np.random.rand(nevents) - 1
        ECM_alp = (
            (const.m_tau**2 - const.m_e**2 + m_alp**2)
            / 2
            / const.m_tau
            * np.ones(nevents)
        )
        pCM_alp = np.sqrt(ECM_alp**2 - m_alp**2)

        p4_alp_CM = Cfv.build_fourvec(ECM_alp, pCM_alp, ctheta_alp, phi_alp)
        self.df_taus["p"] = np.sqrt(
            self.df_taus.px**2 + self.df_taus.py**2 + self.df_taus.pz**2
        )
        ctheta_tau_LAB = (self.df_taus.pz / self.df_taus.p).to_numpy()
        phitau_LAB = np.arctan2(self.df_taus.py, self.df_taus.px).to_numpy()
        p4_alp = Cfv.Tinv(
            p4_alp_CM,
            -(self.df_taus.p / self.df_taus.E).to_numpy(),
            ctheta_tau_LAB,
            phitau_LAB,
        )

        theta_alp = np.arccos(Cfv.get_cosTheta(p4_alp))
        mask_alp_in_acc = (self.theta0 - self.dtheta / 2 < theta_alp) & (
            theta_alp < self.theta0 + self.dtheta / 2
        )

        # If less than 5 generated events were within acceptance, call it a day and return zeros
        if mask_alp_in_acc.sum() < 5:
            p_bins = np.linspace(p4_alp[:, 0].min(), p4_alp[:, 0].max(), 50)
            return np.zeros(len(p_bins) - 1), p_bins

        # Else, histogram event rate in alp energy
        else:
            p_bins = np.linspace(
                p4_alp[mask_alp_in_acc, 0].min(), p4_alp[mask_alp_in_acc, 0].max(), 50
            )

            h, p_bins = np.histogram(
                np.sqrt(p4_alp[:, 0] ** 2 - m_alp**2)[mask_alp_in_acc], bins=p_bins
            )

            # NOTE: CHECK -- is this really pi???
            self.eff_dphi = self.dphi / np.pi
            self.geom_acceptance = mask_alp_in_acc.sum() / nevents * self.eff_dphi
            return h / h.sum(), p_bins

    def get_event_rate(self, alp):

        dPhidp, palp = self.get_alp_spectrum(alp.m_a)

        if np.array(palp).size > 1:
            dp = np.diff(palp)
            pc = palp[:-1] + dp / 2
            return (
                alp.BR_tau_to_a_e()
                * np.sum(dp * dPhidp * alp.prob_decay(pc, self.L, self.dZ))
                * self.norm
                * self.geom_acceptance
            )
        else:
            return (
                alp.BR_tau_to_a_e()
                * dPhidp
                * alp.prob_decay(palp, self.L, self.dZ)
                * self.norm
                * self.geom_acceptance
            )

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
