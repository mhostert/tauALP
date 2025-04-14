import os
import numpy as np
import pandas as pd

from DarkNews import Cfourvec as Cfv

from . import const
from . import models


# Experiments and their parameters
JPARC_tau_per_POT = 4e-9
NuMI_tau_per_POT = 5e-7
SPS_tau_per_POT = 2.7e-6

ND280_exp = {
    "name": "ND280",
    "L": 280e2,
    "x0": 10.75e2,
    "y0": 0,
    "dX": 1.7e2,
    "dY": 1.96e2,
    "dZ": 3 * 56,
    "norm": JPARC_tau_per_POT * (12.34e20 + 6.29e20),
    "Emin": 0.05,
    "final_states": ["ee", "em", "me", "mm"],
}
ICARUS_exp = {
    "name": "ICARUS",
    "L": 803e2,
    "x0": 0.0968 * 803e2,
    "y0": 0,
    "dX": 2 * 2.67e2,
    "dY": 2.86e2,
    "dZ": 17.00e2,
    "norm": NuMI_tau_per_POT * 3e21,
    "Emin": 0.1,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}
MicroBooNE_exp = {
    "name": "MicroBooNE",
    "L": 685e2,
    "x0": 0.146 * 685e2,
    "y0": 0,
    "dX": 2.26e2,
    "dY": 2.03e2,
    "dZ": 9.42e2,
    "norm": NuMI_tau_per_POT * 2e21,
    "Emin": 0.1,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}
NoVA_exp = {
    "name": "NOvA",
    "L": 990e2,
    "x0": 14.6e-3 * 990e2,
    "y0": 0,
    "dX": 3.9e2,
    "dY": 3.9e2,
    "dZ": 12.7e2,
    "norm": NuMI_tau_per_POT * 4.2e21,
    "Emin": 0.1,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}
DUNE_exp = {
    "name": "DUNE-ND",
    "L": 574e2,
    "x0": 0,
    "y0": 0,
    "dX": 5e2,
    "dY": 5e2,
    "dZ": 5.88e2,
    "norm": NuMI_tau_per_POT * 10 * 1.1e21,
    "Emin": 1,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}

TwoByTwo_exp = {
    "name": "2x2 protoDUNE-ND",
    "L": 1035e2,
    "x0": 0,
    "y0": 0,
    "dX": 2 * 0.7e2,
    "dY": 1.4e2,
    "dZ": 2 * 0.7e2,
    "norm": NuMI_tau_per_POT * 1e21 * 0.87,
    "Emin": 0.1,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}
TwoByTwo_absorber_exp = {
    "name": "2x2 protoDUNE-ND absorber",
    "L": 1035e2 - 715e2,
    "x0": 0,
    "y0": 0,
    "dX": 2 * 0.7e2,
    "dY": 1.4e2,
    "dZ": 2 * 0.7e2,
    "norm": NuMI_tau_per_POT * 1e21 * 0.13,
    "Emin": 0.1,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}

ArgoNeuT_exp = {
    "name": "ArgoNeuT",
    "L": 1033e2 - 63,
    "x0": 0,
    "y0": 0,
    "dX": 0.40e2,
    "dY": 0.47e2,
    "dZ": 0.90e2 + 63,
    "norm": NuMI_tau_per_POT * 1.25e20 * 0.5 * 0.87,  # for efficiency,
    "Emin": 10,
    "final_states": ["mm"],
}
ArgoNeuT_absorber_exp = {
    "name": "ArgoNeuT_absorber",
    "L": 1033e2 - 63 - 715e2,
    "x0": 0,
    "y0": 0,
    "dX": 0.40e2,
    "dY": 0.47e2,
    "dZ": 0.90e2 + 63,
    "norm": NuMI_tau_per_POT * 1.25e20 * 0.5 * 0.13,  # for efficiency,
    "Emin": 10,
    "final_states": ["mm"],
}

CHARM_exp = {
    "name": "CHARM",
    "L": 480e2,
    "x0": 5e2,
    "y0": 0,
    "dX": 3e2,
    "dY": 3e2,
    # "R": 1.5e2,
    "dZ": 35e2,
    "norm": SPS_tau_per_POT * 2.4e18,
    "Emin": 1,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}
BEBC_exp = {
    "name": "BEBC",
    "L": 404e2,
    "x0": 0,
    "y0": 0,
    "dX": 3.57e2,
    "dY": 2.52e2,
    "dZ": 1.85e2,
    "norm": SPS_tau_per_POT * 2.72e18,
    "Emin": 1,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}
NA62_exp = {
    "name": "NA62",
    "L": 79.4e2,
    "x0": 0,
    "y0": 0,
    "dX": 2e2,
    "dY": 2e2,
    "dZ": 78e2,
    "norm": SPS_tau_per_POT * 1.4e17,
    "Emin": 1,
    "final_states": ["ee", "mm"],
}
PROTO_DUNE_NP02_exp = {
    "name": "ProtoDUNE-NP02",
    "L": 677e2,
    "x0": 3e2,
    "y0": 1.5e2,
    "dX": 6e2,
    "dY": 7e2,
    "dZ": 6e2,
    "norm": SPS_tau_per_POT * 1.75e19,
    "Emin": 1,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}
PROTO_DUNE_NP04_exp = {
    "name": "ProtoDUNE-NP04",
    "L": 723e2,
    "x0": 3e2,
    "y0": 1.5e2,
    "dX": 6e2,
    "dY": 7e2,
    "dZ": 6e2,
    "norm": SPS_tau_per_POT * 1.75e19,
    "Emin": 1,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}
SHiP_exp = {
    "name": "SHiP",
    "L": 33.7e2,
    "x0": 0,
    "y0": 0,
    "dX": 2.5e2,
    "dY": 4.3e2,
    "dZ": 49.6e2,
    "norm": SPS_tau_per_POT * 6e20,
    "Emin": 1,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}

sigma_tau = 25.95e-30  # cm^2
FASER_exp = {
    "name": "FASER",
    "L": 480e2,
    "x0": 6.5,
    "y0": 0,
    "R": 0.1e2,
    "dZ": 1.5e2,
    "norm": 150e39 * sigma_tau,
    "Emin": 100,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}
FASER2_exp = {
    "name": "FASER2",
    "L": 480e2,
    "x0": 0,
    "y0": 0,
    "R": 1e2,
    "dZ": 5e2,
    "norm": 3e42 * sigma_tau,
    "Emin": 100,
    "final_states": ["ee", "em", "me", "mm", "gg"],
}


def load_events(file_paths, as_dataframe=False, apply_BR_weights=True):
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    # Check if the file paths exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    PARENTS = [411, 431, 100443]
    BRANCHINGS = [1.20e-3, 5.36e-2, 3.1e-3]

    if as_dataframe:
        # Initialize an empty DataFrame
        df = pd.DataFrame()
        particle_count = 0
        event_count = 0
        # Iterate over each file path, read the CSV file, and concatenate it to the existing DataFrame
        for file_path in file_paths:
            df_new = pd.read_csv(file_path, sep=" ").shift(axis=1).iloc[:, 1:]
            if "tau_event_number" in df_new.columns:
                df_new.rename(
                    columns={"tau_event_number": "particle_count"}, inplace=True
                )
            this_event_count = df_new.event_number.max()
            this_particle_count = df_new.particle_count.max()

            df_new.event_number = df_new.event_number + event_count
            df_new.particle_count = df_new.particle_count + particle_count
            event_count += this_event_count
            particle_count += this_particle_count
            df = pd.concat([df, df_new], ignore_index=True)

        if apply_BR_weights:
            df["weights"] = np.zeros(len(df))
            for p, br in zip(PARENTS, BRANCHINGS):
                mask = np.abs(df["mother_pid"]) == p
                df.loc[mask, "weights"] = br
        else:
            df["weights"] = 1 / event_count

        return df
    else:
        data = np.vstack(
            [
                # OLD
                # event_number pid status mother1 mother2 daughter1 daughter2 E px py pz weight
                # np.loadtxt(file, skiprows=1, usecols=(1, 7, 8, 9, 10, 11))
                # NEW
                # event_number particle_count pid E px py pz mother_pid E_mother px_mother py_mother pz_mother
                np.loadtxt(file, skiprows=1, usecols=(2, 3, 4, 5, 6, 7, 7))
                for file in file_paths
            ]
        )
        data[:, -1] = np.zeros(np.shape(data)[0])
        if apply_BR_weights:
            for p, br in zip(PARENTS, BRANCHINGS):
                mask = np.abs(data[:, -2]) == p
                data[mask, -1] = br
        else:
            data[:, -1] = np.ones(np.shape(data)[0])

        return data


class Experiment:
    def __init__(self, file_paths, exp_dic, alp=None, duplicate_taus=None):
        """
        Initializes the experimental setup with the given parameters.

        Args:
            file_paths (list of str): List of file paths containing Pythia8 tau events.

            exp_dic (dict): Dictionary containing experimental attributes.

                    Required keys:
                    - name (str): Name of the experiment.
                    - L (float): Distance to the detector.
                    - dZ (float): Detector depth.
                    - norm (float): Normalization factor.
                    - Emin (float): Minimum energy threshold.
                    - final_states (list): List of ALP decay final states (e.g., ["ee", "em", "me", "mm","gg"]).

                Optional keys:

                    - R (float): Radius of the detector (used for circular detectors).
                    - dX (float): Width of the detector (used for rectangular detectors).
                    - dY (float): Height of the detector (used for rectangular detectors).
                    - theta0 (float): Angle of the detector center in the plane of the detector.
                    - x0 (float): X-coordinate of the detector center.
                    - y0 (float): Y-coordinate of the detector center.

            alp (object, optional): ALP (Axion-Like Particle) model instance.

                If not provided, a default ALP model is initialized.

        Raises:
            ValueError: If any required key is missing from `exp_dic`.

        Attributes:
            df_taus (DataFrame): DataFrame containing loaded event data.
            nevents (int): Number of events in the dataset.
            weight (Series): Normalized weights for the events.
            dtheta (float): Angular width of the detector in the theta direction.
            dphi (float): Angular width of the detector in the phi direction.
            x0 (float): X-coordinate of the detector center.
            y0 (float): Y-coordinate of the detector center.
            alp (object): ALP model instance.
        """

        df_taus = load_events(file_paths)
        if duplicate_taus is not None:
            if duplicate_taus >= 1:
                df_taus = np.repeat(df_taus, duplicate_taus, axis=0)
            else:
                df_taus = df_taus[: int(duplicate_taus * df_taus.shape[0]), :]

            # df_taus = pd.concat([df_taus] * duplicate_taus, ignore_index=True)

        self.nevents = np.shape(df_taus)[0]

        # self.tau_weights = df_taus.weight / df_taus.weight.sum()
        self.tau_weights = df_taus[:, -1] / df_taus[:, -1].sum()
        # self.p4_taus = np.array(
        #     [
        #         df_taus["E"],
        #         df_taus["px"],
        #         df_taus["py"],
        #         df_taus["pz"],
        #     ]
        # ).T
        self.p4_taus = np.array(
            [
                df_taus[:, 1],
                df_taus[:, 2],
                df_taus[:, 3],
                df_taus[:, 4],
            ]
        ).T

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
            self.alp = models.ALP(0.5, 1e5)
        self.tau_BRs = {}
        self.event_rate = self.get_event_rate(self.alp)

    def sample_alp_energy_spectrum(self, production_channel, alp):
        # ECM_alp = Cfv.random_generator(
        #     self.nevents, alp.Ea_min[production_channel], alp.Ea_max[production_channel]
        # )
        ECM_alp = np.random.uniform(
            alp.Ea_min[production_channel], alp.Ea_max[production_channel], self.nevents
        )
        return ECM_alp

    def generate_alp_events(self, alp, production_channel):
        """
        Generate ALP events for a given tau decay channel
        """

        # Flat dOmega = dcos(theta) * dphi
        phi_alp = np.random.uniform(0, 2 * np.pi, self.nevents)
        ctheta_alp = np.random.uniform(-1, 1, self.nevents)

        # Sampling ALP energy from different production channels
        ECM_alp = self.sample_alp_energy_spectrum(production_channel, alp)
        pCM_alp = np.zeros_like(ECM_alp)
        pCM_alp[ECM_alp > alp.m_a] = np.sqrt(
            ECM_alp[ECM_alp > alp.m_a] ** 2 - alp.m_a**2
        )

        # Build ALP 4 momenta
        p4_alp_CM = Cfv.build_fourvec(ECM_alp, pCM_alp, ctheta_alp, phi_alp)
        self.p_taus = Cfv.get_3vec_norm(self.p4_taus)
        ctheta_tau_LAB = Cfv.get_cosTheta(self.p4_taus)
        phitau_LAB = np.arctan2(self.p4_taus[:, 2], self.p4_taus[:, 1])
        beta = -self.p_taus / self.p4_taus[:, 0]
        beta[beta < -1] = -1
        self.p4_alp = Cfv.Tinv(
            p4_alp_CM,
            beta,
            ctheta_tau_LAB,
            phitau_LAB,
        )

        # Total tau branching ratio
        if production_channel == "tau>e+a" or production_channel == "tau>mu+a":
            # 2-body, then exactly calculable,
            self.tau_BRs[production_channel] = alp.tau_BR(production_channel)
        else:
            # 3 or 4-body, so perform a MC numerical integral using our own samples
            self.tau_BRs[production_channel] = np.sum(
                alp.tau_diff_BR(ECM_alp, production_channel) / self.nevents
            ) * (alp.Ea_max[production_channel] - alp.Ea_min[production_channel])

        self.weights = self.tau_weights * self.tau_BRs[production_channel]

        return self.p4_alp, self.weights

    def get_alp_events(self, alp=None):
        """
        Get ALP events for ALL tau decay channels
        """
        if alp is None:
            alp = self.alp
        production_channels = [
            "tau>e+a",
            "tau>mu+a",
            "tau>nu+pi+a",
            "tau>nu+rho+a",
            "tau>nu+nu+e+a",
            "tau>nu+nu+mu+a",
        ]
        p4_list = []
        weights_list = []
        channel_list = []

        for channel in production_channels:
            if alp.tau_BR(channel) > 0:
                p4, weights = self.generate_alp_events(alp, channel)
                p4_list.append(p4)
                weights_list.append(weights)
                channel_list.append(np.repeat(channel, len(weights)))

                # Break if LFV channels available and non-zero
                if p4_list and channel == "tau>mu+a":
                    break

        if p4_list:
            self.p4_alp = np.concatenate(p4_list, axis=0)
            self.weights = np.concatenate(weights_list)
            self.channel_list = np.concatenate(channel_list)
        else:
            self.p4_alp = np.array([])
            self.weights = np.array([])

        # 3-momentum absolute value
        # self.p_alp = np.zeros_like(self.p4_alp[:, 0])
        # mask = self.p4_alp[:, 0] > alp.m_a
        # self.p_alp[mask] = np.sqrt(self.p4_alp[mask, 0] ** 2 - alp.m_a**2)
        self.p_alp = Cfv.get_3vec_norm(self.p4_alp)

        # ALP velocity
        self.v_alp = Cfv.get_3direction(self.p4_alp)

        # projected x,y at the plane of the detector
        self.x_alp = self.v_alp[:, 0] * self.L
        self.y_alp = self.v_alp[:, 1] * self.L

        return self.p4_alp, self.weights, self.channel_list

    def get_alps_in_acceptance(self, generate_events=True, alp=None):

        if generate_events or hasattr(self, "p4_alp") is False:
            self.p4_alp, self.weights, self.channels = self.get_alp_events(alp=alp)

        # if hasattr(self, "theta0"):
        #     theta_alp = np.arccos(Cfv.get_cosTheta(self.p4_alp))
        #     self.mask_alp_in_acc = (self.theta0 - self.dtheta / 2 < theta_alp) & (
        #         theta_alp < self.theta0 + self.dtheta / 2
        #     )
        #     self.signal_selection = self.p4_alp[:, 0] > self.Emin
        #     self.eff = (
        #         self.dphi
        #         / np.pi
        #         * self.weights[self.signal_selection].sum()
        #         / self.weights.sum()
        #     )
        # else:

        # NOTE: Selection of events in a square
        # NOTE: Assume a cuboid... need to extend for SHiP
        if hasattr(self, "R"):
            self.mask_alp_in_acc = (
                ((self.x_alp - self.x0) ** 2 + (self.y_alp - self.y0) ** 2) < self.R**2
            ) & (self.p4_alp[:, 0] > self.Emin)
        else:
            self.mask_alp_in_acc = (
                (np.abs(self.x_alp - self.x0) < self.dX / 2)
                & (np.abs(self.y_alp - self.y0) < self.dY / 2)
                & (self.p4_alp[:, 0] > self.Emin)
            )
        self.eff = self.weights[self.mask_alp_in_acc].sum() / self.weights.sum()

        return (
            self.p4_alp[self.mask_alp_in_acc],
            self.weights[self.mask_alp_in_acc],
            self.channel_list[self.mask_alp_in_acc],
        )

    def get_alp_momentum_spectrum(
        self, alp=None, selection=True, bins=40, generate_events=True
    ):
        """Get histogram of ALP momenta produced in a given tau decay channel

                * returns the histogram of ALP momenta, the bin edges, and the geom acceptance
        Returns:
            np.ndarray: the area normalized histogram of ALP momenta
            np.ndarray: the bin edges
        """

        if generate_events:
            # Get ALP events for all channels
            self.p4_alp, self.weights, self.channels = self.get_alp_events(alp=alp)
            self.get_alps_in_acceptance(alp=alp)

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

            return h, p_bins

    def get_event_rate(self, alp, selection=True, generate_events=True):

        # self.dPhidp, self.palp = self.get_alp_momentum_spectrum(
        #     alp, selection=selection
        # )

        # self.dp = np.diff(self.palp)
        # self.pc = self.palp[:-1] + self.dp / 2
        if generate_events:
            self.get_alps_in_acceptance(generate_events=generate_events, alp=alp)

        self.flux = self.norm * self.weights[self.mask_alp_in_acc]
        self.p4_alp_in_acc = self.p4_alp[self.mask_alp_in_acc]
        self.channel_list_inc_acc = self.channel_list[self.mask_alp_in_acc]
        self.total_rate = np.sum(self.flux * self.get_signal_prob_decay(alp))
        return self.total_rate
        #     np.sum(
        #     self.dPhidp
        #     * alp.prob_decay(self.pc, self.L, self.dZ)
        # )

    def get_signal_prob_decay(self, alp, mask=None):
        """Calculate the probability of decay for a given ALP momentum and distance"""
        if mask is None:
            return alp.alp_visible_BR(self.final_states) * alp.prob_decay(
                self.p4_alp_in_acc[:, 0], self.L, self.dZ
            )
        else:
            return alp.alp_visible_BR(self.final_states) * alp.prob_decay(
                self.p4_alp_in_acc[:, 0][mask], self.L, self.dZ
            )

    def reweight(self, alp_old, alp_new):
        """Reweight the event rate for a new ALP decay constant

            NOTE: Both ALP *MUST* be the same mass and have the same 'c_lepton' matrix.

        Args:
            alp_old (alp.models.ALP): starting ALP
            alp_new (alp.models.ALP): desired ALP

        Returns:
            float: the new event rate
        """
        return (alp_old.f_a / alp_new.f_a) ** 2 * np.sum(
            self.flux * self.get_signal_prob_decay(alp_new)
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
