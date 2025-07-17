import os
import numpy as np
import pandas as pd
import glob

from DarkNews import Cfourvec as Cfv

from . import models


def read_pythia_file_with_attrs(file_path, n_header_lines=8):
    # Read header lines (first 8 lines)
    with open(file_path, "r") as f:
        header_lines = [next(f) for _ in range(n_header_lines)]

    attrs = {}
    # Parse header lines for attributes
    # We assume the header lines are formatted as "key: value"
    # These contain information about the generation of events, including the seed, version, and other parameters.
    # If the header lines are not present, we can skip this part
    if n_header_lines > 0:
        for line in header_lines:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                key = key.strip().lower().replace(" ", "_")  # clean the key
                key = key[2:]
                value = value.strip()

                # Try to convert to float or int if possible
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # leave as string

                attrs[key] = value

    # Now read the events (skipping header)
    df = (
        pd.read_csv(file_path, sep=" ", skiprows=n_header_lines)
        .shift(axis=1)
        .iloc[:, 1:]
    )

    # Attach attributes
    df.attrs.update(attrs)

    return df


def load_events(
    file_names,
    apply_tauBR_weights=True,
    apply_xsec_weights=True,
):
    """
    Load tau events from Pythia8 or local pandas files.
    """

    # In-house Ds->tau files
    if "parquet" in file_names:
        # Read parquet file
        df = pd.read_parquet(file_names)
        return df

    # Pythia8 files
    if not isinstance(file_names, list):
        files = glob.glob(f"{file_names}_*.txt")
    else:
        files = []
        for file_name in file_names:
            files += glob.glob(f"{file_name}_*.txt")

    # Check if the file paths exist
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise FileNotFoundError(f"No valid files matching {file_names}_*.txt")
    nfiles = len(files)
    if nfiles == 0:
        raise FileNotFoundError(f"No files found matching pattern: {file_names}_*.txt")

    PARENTS = [
        411,
        431,
        100443,
    ]
    BRANCHINGS = [1.20e-3, 5.36e-2, 3.1e-3]

    tau_soft = 0
    tau_hard = 0

    total_soft = 0
    total_hard = 0

    # Initialize an empty DataFrame
    particle_count = 0
    event_count = 0
    # Iterate over each file path, read the CSV file, and concatenate it to the existing DataFrame
    tot_tau_xsec = 0.0
    tot_xsec_soft = 0.0
    tot_xsec_hard = 0.0

    dfs = []  # collect dataframes here

    for file_path in files:

        if tau_hard > 500_000:
            continue  # already plenty of QCDhard events!
        try:
            # Handle the case where the file is from my local runs (no headers at the time)
            df_new = pd.read_csv(file_path, sep=" ").shift(axis=1).iloc[:, 1:]
        except pd.errors.ParserError:
            # Handle the case where the file is from the cluster runs (includes headers)
            df_new = read_pythia_file_with_attrs(file_path)

        # Obsolete "tau_event_number" column name
        if "tau_event_number" in df_new.columns:
            df_new.rename(columns={"tau_event_number": "particle_count"}, inplace=True)

        this_event_count = df_new.event_number.max()
        this_particle_count = df_new.particle_count.max()

        df_new.event_number = df_new.event_number + event_count
        df_new.particle_count = df_new.particle_count + particle_count
        event_count += this_event_count
        particle_count += this_particle_count

        tot_tau_xsec += df_new.attrs.get("tau_xsec_mb", 0.0) / len(files)

        # NOTE: setting all weights to 1.0
        if apply_xsec_weights:
            df_new["weights"] = np.ones(len(df_new))
            if df_new.attrs["mode"] == "hard":
                tau_hard += this_particle_count
                total_hard += this_event_count
                tot_xsec_hard += df_new.attrs.get("total_xsec_mb", 0.0) / len(files)
                # df_new["QCD"] = "h"
            elif df_new.attrs["mode"] == "soft":
                tau_soft += this_particle_count
                total_soft += this_event_count
                tot_xsec_soft += df_new.attrs.get("total_xsec_mb", 0.0) / len(files)
                # df_new["QCD"] = "s"
            else:
                raise ValueError(
                    f"Unknown file type: {file_path}. Expected 'hard' or 'soft' in the filename."
                )

            df_new["weights"] *= df_new.attrs.get("tau_xsec_mb", 0.0)

        if apply_tauBR_weights:
            for p, br in zip(PARENTS, BRANCHINGS):
                mask = np.abs(df_new["mother_pid"]) == p
                df_new.loc[mask, "weights"] *= br

        # Concatenate
        dfs.append(df_new)

    df = pd.concat(dfs, ignore_index=True)
    df.attrs.update(df_new.attrs)
    df.attrs["tau_xsec_mb"] = tot_tau_xsec
    df.attrs["total_xsec_mb"] = tot_xsec_soft + tot_xsec_hard
    if apply_xsec_weights:
        df["weights"] = df["weights"] / (
            total_hard * tot_xsec_hard + total_soft * tot_xsec_soft
        )
    return df


class Experiment:
    def __init__(
        self, file_paths, exp_dic, alp=None, duplicate_taus=None, savemem=True
    ):
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

        if isinstance(file_paths, list) or (
            isinstance(file_paths, str) and ".pd" not in file_paths
        ):
            df_taus = load_events(file_paths)
        elif isinstance(file_paths, str) and ".pd" in file_paths:
            df_taus = pd.read_parquet(file_paths)

        if duplicate_taus is not None:
            # if duplicate_taus >= 1:
            #     df_taus = np.repeat(df_taus, duplicate_taus, axis=0)
            # else:
            #     df_taus = df_taus[: int(duplicate_taus * df_taus.shape[0]), :]

            df_taus = pd.concat([df_taus] * duplicate_taus, ignore_index=True)

        self.nevents = len(df_taus)
        self.tau_weights = df_taus["weights"].to_numpy("float64", copy=False)
        self.tau_weights /= self.tau_weights.sum()
        self.p4_taus = df_taus[["E", "px", "py", "pz"]].to_numpy("float64", copy=False)

        # self.nevents = np.shape(df_taus)[0]
        # self.tau_weights = df_taus.weights / df_taus.weights.sum()
        # # self.tau_weights = df_taus[:, -1] / df_taus[:, -1].sum()
        # self.p4_taus = np.array(
        #     [
        #         df_taus["E"],
        #         df_taus["px"],
        #         df_taus["py"],
        #         df_taus["pz"],
        #     ]
        # ).T
        # self.p4_taus = np.array(
        #     [
        #         df_taus[:, 1],
        #         df_taus[:, 2],
        #         df_taus[:, 3],
        #         df_taus[:, 4],
        #     ]
        # ).T
        self.savemem = savemem
        if not savemem:
            self.df_taus = df_taus
        del df_taus

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

    def get_parent_p4(self):
        p4_parent = np.array(
            [
                self.df_taus["E_mother"],
                self.df_taus["px_mother"],
                self.df_taus["py_mother"],
                self.df_taus["pz_mother"],
            ]
        ).T
        return p4_parent

    def sample_alp_energy_spectrum(self, production_channel, alp, nevents=None):
        # ECM_alp = Cfv.random_generator(
        #     self.nevents, alp.Ea_min[production_channel], alp.Ea_max[production_channel]
        # )
        if nevents is None:
            nevents = self.nevents
        ECM_alp = np.random.uniform(
            alp.Ea_min[production_channel], alp.Ea_max[production_channel], nevents
        )
        return ECM_alp

    def sample_alp_daughter_4momenta(self, decay_channel, alp, nevents=None):

        if nevents is None:
            nevents = self.nevents

        # Flat dOmega = dcos(theta) * dphi
        phi_1 = np.random.uniform(0, 2 * np.pi, nevents)
        ctheta_1 = np.random.uniform(-1, 1, nevents)

        m1 = models.LEPTON_MASSES[models.LEPTON_INDEX[decay_channel[0]]]
        m2 = models.LEPTON_MASSES[models.LEPTON_INDEX[decay_channel[1]]]

        # Sampling ALP energy from different production channels
        ECM_1 = np.ones(nevents) * (alp.m_a**2 + m1**2 - m2**2) / (2 * alp.m_a)
        ECM_2 = np.ones(nevents) * (alp.m_a**2 - m1**2 + m2**2) / (2 * alp.m_a)

        pCM_1 = np.zeros_like(ECM_1)
        pCM_1[ECM_1 > m1] = np.sqrt(ECM_1[ECM_1 > m1] ** 2 - m1**2)

        # Build ALP 4 momenta
        p4_CM_1 = Cfv.build_fourvec(ECM_1, pCM_1, ctheta_1, phi_1)
        p4_CM_2 = Cfv.build_fourvec(ECM_2, -pCM_1, ctheta_1, phi_1)
        ctheta_alp_LAB = Cfv.get_cosTheta(self.p4_alp)
        phi_alp_LAB = np.arctan2(self.p4_alp[:, 2], self.p4_alp[:, 1])

        beta = -np.sqrt(1 - (alp.m_a / self.p4_alp[:, 0]) ** 2)
        beta[beta < -1] = -1
        p4_1 = Cfv.Tinv(
            p4_CM_1,
            beta,
            ctheta_alp_LAB,
            phi_alp_LAB,
        )
        p4_2 = Cfv.Tinv(
            p4_CM_2,
            beta,
            ctheta_alp_LAB,
            phi_alp_LAB,
        )

        return p4_1, p4_2

    def generate_alp_events(
        self, alp, production_channel, decay_channel, mask_taus=None
    ):
        """
        Generate ALP events for a given tau decay channel
        """

        n_taus = mask_taus.sum() if mask_taus is not None else self.nevents

        # Flat dOmega = dcos(theta) * dphi
        phi_alp = np.random.uniform(0, 2 * np.pi, n_taus)
        ctheta_alp = np.random.uniform(-1, 1, n_taus)

        # Sampling ALP energy from different production channels
        ECM_alp = self.sample_alp_energy_spectrum(
            production_channel, alp, nevents=n_taus
        )
        pCM_alp = np.zeros_like(ECM_alp)
        pCM_alp[ECM_alp > alp.m_a] = np.sqrt(
            ECM_alp[ECM_alp > alp.m_a] ** 2 - alp.m_a**2
        )

        # Build ALP 4 momenta
        p4_tau = self.p4_taus[mask_taus]
        p4_alp_CM = Cfv.build_fourvec(ECM_alp, pCM_alp, ctheta_alp, phi_alp)
        p_taus = Cfv.get_3vec_norm(p4_tau)
        ctheta_tau_LAB = Cfv.get_cosTheta(p4_tau)
        phitau_LAB = np.arctan2(p4_tau[:, 2], p4_tau[:, 1])
        beta = -p_taus / p4_tau[:, 0]
        beta[beta < -1] = -1

        self.p4_alp = Cfv.Tinv(
            p4_alp_CM,
            beta,
            ctheta_tau_LAB,
            phitau_LAB,
        )

        self.p4_1, self.p4_2 = self.sample_alp_daughter_4momenta(
            decay_channel=decay_channel, alp=alp, nevents=n_taus
        )

        # Total tau branching ratio
        if production_channel == "tau>e+a" or production_channel == "tau>mu+a":
            # 2-body, then exactly calculable,
            self.tau_BRs[production_channel] = alp.tau_BR(production_channel)
        else:
            # 3 or 4-body, so perform a MC numerical integral using our own samples
            self.tau_BRs[production_channel] = np.sum(
                alp.tau_diff_BR(ECM_alp, production_channel) / n_taus
            ) * (alp.Ea_max[production_channel] - alp.Ea_min[production_channel])

        self.weights = self.tau_weights[mask_taus] * self.tau_BRs[production_channel]

        return self.p4_alp, self.p4_1, self.p4_2, self.weights

    def get_alp_events(self, alp=None):
        """
        Get ALP events for ALL tau decay channels
        """
        if alp is None:
            alp = self.alp

        # Preload branching ratios
        br_decay_dict = {
            decay: vars(alp)[f"BR_a_to_{decay}"] for decay in self.final_states
        }

        # Compute channel weights
        dic_channel_weights = {
            (prod, decay): alp.tau_BR(prod) * br_decay_dict[decay]
            for prod in [
                "tau>e+a",
                "tau>mu+a",
                "tau>nu+pi+a",
                "tau>nu+rho+a",
                "tau>nu+nu+e+a",
                "tau>nu+nu+mu+a",
            ]
            for decay in self.final_states
        }

        # Normalize and filter
        total = sum(dic_channel_weights.values())
        dic_channel_weights = {
            k: v / total for k, v in dic_channel_weights.items() if (v / total) > 1e-3
        }

        # Assign events
        choices_of_taus = np.arange(self.nevents)
        np.random.shuffle(choices_of_taus)
        event_splits = {}
        start_idx = 0

        for (prod_channel, decay_channel), weight in dic_channel_weights.items():
            n = int(weight * self.nevents)
            selected = choices_of_taus[start_idx : start_idx + n]
            start_idx += n
            event_splits[(prod_channel, decay_channel)] = selected

        # First, compute number of events per channel
        event_splits = {}
        total_generated = 0
        choices_of_taus = np.arange(self.nevents)
        np.random.shuffle(choices_of_taus)
        start_idx = 0

        for (prod_channel, decay_channel), weight in dic_channel_weights.items():
            n = int(weight * self.nevents)
            selected = choices_of_taus[start_idx : start_idx + n]
            start_idx += n
            event_splits[(prod_channel, decay_channel)] = selected
            total_generated += n

        # Preallocate all outputs
        p4_alp = np.empty((total_generated, 4))
        p4_d1 = np.empty((total_generated, 4))
        p4_d2 = np.empty((total_generated, 4))
        weights = np.empty(total_generated)

        # Fill in arrays directly
        insert_idx = 0
        for (prod_channel, decay_channel), indices in event_splits.items():
            mask = np.zeros(self.nevents, dtype=bool)
            mask[indices] = True

            p_alp, p1, p2, w = self.generate_alp_events(
                alp, prod_channel, decay_channel, mask_taus=mask
            )

            n_events = len(w)
            p4_alp[insert_idx : insert_idx + n_events] = p_alp
            p4_d1[insert_idx : insert_idx + n_events] = p1
            p4_d2[insert_idx : insert_idx + n_events] = p2
            weights[insert_idx : insert_idx + n_events] = w
            insert_idx += n_events

        # Final assignment
        self.p4_alp = p4_alp
        self.p4_daughter1 = p4_d1
        self.p4_daughter2 = p4_d2
        self.weights = weights
        del p1, p2, weights, self.p4_1, self.p4_2

        # self.p4_alp = self.p4_daughter1 + self.p4_daughter2

        # 3-momentum absolute value
        # self.p_alp = np.zeros_like(self.p4_alp[:, 0])
        # mask = self.p4_alp[:, 0] > alp.m_a
        # self.p_alp[mask] = np.sqrt(self.p4_alp[mask, 0] ** 2 - alp.m_a**2)
        self.p_alp = Cfv.get_3vec_norm(self.p4_alp)

        # ALP velocity
        v_alp = Cfv.get_3direction(self.p4_alp)
        v_daughter1 = Cfv.get_3direction(self.p4_daughter1)
        v_daughter2 = Cfv.get_3direction(self.p4_daughter2)

        # random sample the ALP decay position according to its lifetime
        # dec_length = const.get_decay_rate_in_cm(self.alp.Gamma_a) * self.p_alp / alp.m_a
        # self.z_alp = self.L + np.random.exponential(scale=dec_length, size=np.shape(self.x_alp))

        # uniformly sample the ALP decay position inside detector (to be reweighted later)
        self.z_alp = np.random.uniform(
            self.L, self.L + self.dZ, size=np.shape(self.p_alp)
        )

        # projected x,y at the plane of the detector
        self.x_alp = v_alp[:, 0] * self.z_alp
        self.y_alp = v_alp[:, 1] * self.z_alp

        self.x_daughter1_exit = self.x_alp + v_daughter1[:, 0] * (
            self.L + self.dZ - self.z_alp
        )
        self.y_daughter1_exit = self.y_alp + v_daughter1[:, 1] * (
            self.L + self.dZ - self.z_alp
        )

        self.x_daughter2_exit = self.x_alp + v_daughter2[:, 0] * (
            self.L + self.dZ - self.z_alp
        )
        self.y_daughter2_exit = self.y_alp + v_daughter2[:, 1] * (
            self.L + self.dZ - self.z_alp
        )

        del v_alp, v_daughter1, v_daughter2

        return self.p4_alp, self.weights  # , self.channel_list

    def get_alps_in_acceptance(self, generate_events=True, alp=None):

        if generate_events or hasattr(self, "p4_alp") is False:
            self.get_alp_events(alp=alp)

        # NOTE: Selection of events in a square assuming a cuboid
        if hasattr(self, "R"):
            if self.active_volume:
                self.mask_alp_in_acc = (
                    (
                        (
                            (self.x_daughter1_exit - self.x0) ** 2
                            + (self.y_daughter1_exit - self.y0) ** 2
                        )
                        < self.R**2
                    )
                    & (
                        (
                            (self.x_daughter2_exit - self.x0) ** 2
                            + (self.y_daughter2_exit - self.y0) ** 2
                        )
                        < self.R**2
                    )
                    & (self.p4_daughter1[:, 0] > self.Emin / 2)
                    & (self.p4_daughter2[:, 0] > self.Emin / 2)
                )
            else:
                self.mask_alp_in_acc = (
                    (
                        ((self.x_alp - self.x0) ** 2 + (self.y_alp - self.y0) ** 2)
                        < self.R**2
                    )
                    & (self.p4_daughter1[:, 0] > self.Emin / 2)
                    & (self.p4_daughter2[:, 0] > self.Emin / 2)
                )

        else:
            if self.active_volume:
                self.mask_alp_in_acc = (
                    (
                        ((np.abs(self.x_daughter1_exit - self.x0)) < self.dX / 2)
                        & ((np.abs(self.y_daughter1_exit - self.y0)) < self.dY / 2)
                        & ((np.abs(self.x_daughter2_exit - self.x0)) < self.dX / 2)
                        & ((np.abs(self.y_daughter2_exit - self.y0)) < self.dY / 2)
                    )
                    & (self.p4_daughter1[:, 0] > self.Emin / 2)
                    & (self.p4_daughter2[:, 0] > self.Emin / 2)
                )
            else:
                self.mask_alp_in_acc = (
                    (np.abs(self.x_alp - self.x0) < self.dX / 2)
                    & (np.abs(self.y_alp - self.y0) < self.dY / 2)
                    & (self.p4_daughter1[:, 0] > self.Emin / 2)
                    & (self.p4_daughter2[:, 0] > self.Emin / 2)
                )

        self.eff = self.weights[self.mask_alp_in_acc].sum() / self.weights.sum()

        return (
            self.p4_alp[self.mask_alp_in_acc],
            self.weights[self.mask_alp_in_acc],
            # self.channel_list[self.mask_alp_in_acc],
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
            self.get_alp_events(alp=alp)
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
        # self.channel_list_inc_acc = self.channel_list[self.mask_alp_in_acc]
        self.total_rate = np.sum(self.flux * self.get_signal_prob_decay(alp))

        if self.savemem:
            del self.p4_alp
            del self.p4_daughter1
            del self.p4_daughter2
            del self.p_alp

        return self.total_rate

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
