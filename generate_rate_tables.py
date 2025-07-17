import multiprocessing as mp
import numpy as np

from alp import exp, sim_tools, const
from alp.exp_dicts import EXPERIMENTS
from tqdm.contrib.concurrent import process_map

# n_events = 100_000
# NUMI_files = f"tau_events/df_NuMI_{n_events}_custom_parametrization.parquet"
# SPS_files = f"tau_events/df_SPS_{n_events}_custom_parametrization.parquet"
# LHC_files = f"tau_events/df_LHC_{n_events}_custom_parametrization.parquet"

NUMI_files = "pythia8_events/numi_120GeV.parquet"
SPS_files = "pythia8_events/sps_400GeV.parquet"
LHC_files = "pythia8_events/lhc_13.6TeV.parquet"


# Creating the experimental classes
ICARUS = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["ICARUS"], duplicate_taus=3)
MICROBOONE = exp.Experiment(
    NUMI_files, exp_dic=EXPERIMENTS["MicroBooNE"], duplicate_taus=20
)
NOVA = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["NoVA"], duplicate_taus=2)

DUNE = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["DUNE"])
TWOBYTWO = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["TwoByTwo"])
TWOBYTWO_ABSORBER = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["TwoByTwo_absorber"])

ARGONEUT = exp.Experiment(
    NUMI_files, exp_dic=EXPERIMENTS["ArgoNeuT"], duplicate_taus=20
)
ARGONEUT_absorber = exp.Experiment(
    NUMI_files, exp_dic=EXPERIMENTS["ArgoNeuT_absorber"], duplicate_taus=20
)

CHARM = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS["CHARM"])
BEBC = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS["BEBC"])
# NA62 = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS['NA62'])
SHIP = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS["SHiP"])

PROTODUNE_NP02 = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS["PROTO_DUNE_NP02"])
PROTODUNE_NP04 = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS["PROTO_DUNE_NP04"])

FASER = exp.Experiment(LHC_files, exp_dic=EXPERIMENTS["FASER"])
FASER2 = exp.Experiment(LHC_files, exp_dic=EXPERIMENTS["FASER2"])

NPOINTS = 100
MAX_WORKERS = mp.cpu_count() - 3


def simulate(args):
    exp_case, kwargs = args
    _ = sim_tools.make_rate_table(exp_case, save=True, **kwargs)


def run_simulations(exp_list, BP_NAME, **kwargs):
    print(f"Running simulations for {BP_NAME}...")
    args_list = [(exp_case, kwargs) for exp_case in exp_list]
    process_map(simulate, args_list, max_workers=MAX_WORKERS)


if __name__ == "__main__":

    mp.set_start_method("spawn")

    # """
    # LFC   FA vs ma
    # """
    # BP_NAME = "LFC_universal_jul17"
    # c_lepton = np.diag([1, 1, 1])

    # kwargs = {
    #     "inv_fa_range": [1e-6, 1],
    #     "ma_range": [2e-3, 0.25],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = [CHARM, BEBC, SHIP, FASER2, PROTODUNE_NP02, PROTODUNE_NP04, DUNE]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFC   FA vs ma -- e-tau coupled
    # """
    # BP_NAME = "LFC_etau_jul17"
    # c_lepton = np.diag([1, 0, 1])

    # kwargs = {
    #     "inv_fa_range": [1e-5, 1],
    #     "ma_range": [2e-3, const.m_tau * 1.1],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = [CHARM, BEBC, SHIP, FASER2, PROTODUNE_NP02, PROTODUNE_NP04, DUNE]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFC   FA vs ma -- tauphilic
    # """
    # BP_NAME = "LFC_tauphilic_jul17"
    # c_lepton = np.diag([0, 0, 1])

    # kwargs = {
    #     "inv_fa_range": [1e-4, 10],
    #     "ma_range": [2e-3, const.m_tau * 1.1],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = [CHARM, BEBC, SHIP, FASER2, PROTODUNE_NP02, PROTODUNE_NP04, DUNE]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFV ANARCHY   FA vs ma
    # """
    # BP_NAME = "anarchy_jul17"
    # lamb = 1
    # c_lepton = np.ones((3, 3))

    # kwargs = {
    #     "inv_fa_range": [1e-10, 1e-3],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = [
    #     ICARUS,
    #     # MICROBOONE,
    #     NOVA,
    #     CHARM,
    #     BEBC,
    #     SHIP,
    #     FASER,
    #     FASER2,
    #     PROTODUNE_NP02,
    #     PROTODUNE_NP04,
    #     DUNE,
    # ]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    # LFV HIEREACHY  FA vs ma
    # """
    # for lam in [0.05, 0.01, 0.005, 0.001]:
    #     BP_NAME = f"hierarchy_lambda_{lam}_clep_1_jul17"
    #     lamb = lam
    #     C_LEPTON = 1
    #     c_lepton = C_LEPTON * np.array(
    #         [[1, lamb**2, lamb], [lamb**2, 1, lamb], [lamb, lamb, 1]]
    #     )

    #     kwargs = {
    #         "inv_fa_range": [1e-10, 1e-3],
    #         "name": BP_NAME,
    #         "c_lepton": c_lepton,
    #         "Npoints": NPOINTS,
    #     }

    #     exp_list = [
    #         ICARUS,
    #         # MICROBOONE,
    #         NOVA,
    #         CHARM,
    #         BEBC,
    #         SHIP,
    #         FASER,
    #         FASER2,
    #         PROTODUNE_NP02,
    #         PROTODUNE_NP04,
    #         DUNE,
    #     ]
    #     run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFV ANARCHY   FA vs ma
    # """
    # BP_NAME = "anarchy_jul17"
    # lamb = 1
    # c_lepton = np.ones((3, 3))

    # kwargs = {
    #     "inv_fa_range": [1e-9, 1e-3],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = [
    #     MICROBOONE,
    # ]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    # LFV HIEREACHY  FA vs ma
    # """
    # for lam in [0.05, 0.01, 0.005, 0.001]:
    #     BP_NAME = f"hierarchy_lambda_{lam}_clep_1_jul17"
    #     lamb = lam
    #     C_LEPTON = 1
    #     c_lepton = C_LEPTON * np.array(
    #         [[1, lamb**2, lamb], [lamb**2, 1, lamb], [lamb, lamb, 1]]
    #     )

    #     kwargs = {
    #         "inv_fa_range": [1e-9, 1e-3],
    #         "name": BP_NAME,
    #         "c_lepton": c_lepton,
    #         "Npoints": NPOINTS,
    #     }

    #     exp_list = [
    #         MICROBOONE,
    #     ]
    #     run_simulations(exp_list, BP_NAME, **kwargs)

    """'
        LFV Ra=5 FA vs ma
    """
    BP_NAME = "LFV_Ra_5_jul17"
    c_lepton = np.array([[5, 2, 2], [2, 5, 2], [2, 2, 5]])

    kwargs = {
        "inv_fa_range": [1e-9, 1e-5],
        "ma_range": [2.2e-1, 1.7],
        "name": BP_NAME,
        "c_lepton": c_lepton,
        "Npoints": NPOINTS,
    }

    # exp_list = [ARGONEUT, ARGONEUT_absorber, CHARM, BEBC]
    exp_list = [ARGONEUT, ARGONEUT_absorber]
    run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFV Ra=1/3 FA vs ma
    # """
    # BP_NAME = "LFV_Ra_1o3_jul17"
    # c_lepton = np.array([[0.333, 2, 2], [2, 0.333, 2], [2, 2, 0.333]])

    # kwargs = {
    #     "inv_fa_range": [1e-9, 1e-5],
    #     "ma_range": [2.2e-1, 1.7],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = [ARGONEUT, ARGONEUT_absorber, CHARM, BEBC]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """
    #     LFV ANARCHY   FA vs ma
    # """
    # BP_NAME = "anarchy_vMC"
    # lamb = 1
    # c_lepton = np.ones((3, 3))

    # kwargs = {
    #     "inv_fa_range": [1e-10, 1e-3],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = []
    # for exp_name in ["MicroBooNE"]:
    #     for ncase in ["nlow", "nhigh"]:
    #         for bcase in ["blow", "bhigh"]:
    #             EXPERIMENTS[exp_name]["name"] = f"{exp_name}_{ncase}_{bcase}"

    #             NOVA_case = exp.Experiment(
    #                 f"tau_events/df_120GeV_1e6_{ncase}_{bcase}.parquet",
    #                 exp_dic=EXPERIMENTS[exp_name],
    #             )

    #             exp_list.append(NOVA_case)

    # # Run with Pythia8 events:
    # EXPERIMENTS[exp_name]["name"] = f"{exp_name}_pythia8"
    # new_case = exp.Experiment(
    #     NUMI_files,
    #     exp_dic=EXPERIMENTS[exp_name],
    # )
    # exp_list.append(new_case)

    # for exp_name in ["CHARM"]:
    #     for ncase in ["nlow", "nhigh"]:
    #         for bcase in ["blow", "bhigh"]:
    #             EXPERIMENTS[exp_name]["name"] = f"{exp_name}_{ncase}_{bcase}"

    #             NOVA_case = exp.Experiment(
    #                 f"tau_events/df_400GeV_1e6_{ncase}_{bcase}.parquet",
    #                 exp_dic=EXPERIMENTS[exp_name],
    #             )

    #             exp_list.append(NOVA_case)

    # # Run with Pythia8 events:
    # EXPERIMENTS[exp_name]["name"] = f"{exp_name}_pythia8"
    # new_case = exp.Experiment(
    #     NUMI_files,
    #     exp_dic=EXPERIMENTS[exp_name],
    # )
    # exp_list.append(new_case)

    # run_simulations(exp_list, BP_NAME, **kwargs)
