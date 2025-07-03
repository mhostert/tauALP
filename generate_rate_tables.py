import multiprocessing as mp
import numpy as np

from alp import exp, sim_tools
from alp.exp_dicts import EXPERIMENTS
from tqdm.contrib.concurrent import process_map

# Pythia8 tau events
# NUMI_files = [
#     f"pythia8_events/tau_events_120GeV_HardOff_pT0.0001_{i}.txt" for i in range(0, 8)
# ] + [
#     f"pythia8_events/tau_events_120GeV_HardOff_pT0.0001_v2_{i}.txt" for i in range(0, 8)
# ]

# SPS_files = [
#     f"pythia8_events/tau_events_120GeV_HardOff_pT0.0001_{i}.txt" for i in range(0, 8)
# ]
# LHC_files = [f"pythia8_events/tau_events_LHC_13.6TeV_v6_{i}.txt" for i in range(0, 8)]

# NUMI_files = "tau_events/df_120GeV.parquet"  # In-house MC
# SPS_files = "tau_events/df_400GeV.parquet"  # In-house MC
# LHC_files = "pythia8_cluster/pythia8_events_pT10GeV/soft_LHC_13.6TeV_pT13.6TeV"  # Pythia8 events

NUMI_files = [
    "pythia8_events/soft_120_GeV",
    "pythia8_events/soft_120_GeV_3e3",
    "pythia8_events/soft_NuMI_120GeV_pt1TeV",
    "pythia8_events/soft_NuMI_120GeV_pt1TeV_nopThatmin",
]
SPS_files = [
    "pythia8_events/soft_SPS_400GeV_pt1TeV",
    "pythia8_events/soft_SPS_400GeV_pt1TeV_v2",
    "pythia8_events/soft_SPS_400GeV_pt1TeV_v3",
]
LHC_files = ["pythia8_events/soft_test_LHC13.6TeV_pt1TeV_weighted"]


# Creating the experimental classes
ICARUS = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["ICARUS"], duplicate_taus=1)
MICROBOONE = exp.Experiment(
    NUMI_files, exp_dic=EXPERIMENTS["MicroBooNE"], duplicate_taus=1
)
NOVA = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["NoVA"], duplicate_taus=1)

DUNE = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["DUNE"])
TWOBYTWO = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["TwoByTwo"])
TWOBYTWO_ABSORBER = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["TwoByTwo_absorber"])

ARGONEUT = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["ArgoNeuT"])
ARGONEUT_absorber = exp.Experiment(NUMI_files, exp_dic=EXPERIMENTS["ArgoNeuT_absorber"])

CHARM = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS["CHARM"])
BEBC = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS["BEBC"])
# NA62 = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS['NA62'])
SHIP = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS["SHiP"])

PROTODUNE_NP02 = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS["PROTO_DUNE_NP02"])
PROTODUNE_NP04 = exp.Experiment(SPS_files, exp_dic=EXPERIMENTS["PROTO_DUNE_NP04"])

FASER = exp.Experiment(LHC_files, exp_dic=EXPERIMENTS["FASER"])
FASER2 = exp.Experiment(LHC_files, exp_dic=EXPERIMENTS["FASER2"])

NPOINTS = 26


def simulate(args):
    exp_case, kwargs = args
    _ = sim_tools.make_rate_table(exp_case, save=True, **kwargs)


def run_simulations(exp_list, BP_NAME, **kwargs):
    print(f"Running simulations for {BP_NAME}...")
    args_list = [(exp_case, kwargs) for exp_case in exp_list]
    process_map(simulate, args_list, max_workers=6)


if __name__ == "__main__":

    mp.set_start_method("spawn")

    """'
        LFV ANARCHY   FA vs ma
    """
    BP_NAME = "anarchy_vMC"
    lamb = 1
    c_lepton = np.ones((3, 3))

    kwargs = {
        "inv_fa_range": [1e-10, 1e-3],
        "name": BP_NAME,
        "c_lepton": c_lepton,
        "Npoints": NPOINTS,
    }

    exp_list = []
    # for ncase in ["nlow", "nhigh"]:
    #     for bcase in ["blow", "bhigh"]:
    #         EXPERIMENTS["NoVA"]["name"] = f"NOVA_{ncase}_{bcase}"

    #         NOVA_case = exp.Experiment(
    #             f"tau_events/df_120GeV_1e5_{ncase}_{bcase}.parquet",
    #             exp_dic=EXPERIMENTS["NoVA"],
    #         )

    #         exp_list.append(NOVA_case)

    # Run with Pythia8 events:
    EXPERIMENTS["NoVA"]["name"] = f"NOVA_pythia8"
    NOVA_case = exp.Experiment(
        NUMI_files,
        exp_dic=EXPERIMENTS["NoVA"],
    )
    exp_list.append(NOVA_case)

    run_simulations(exp_list, BP_NAME, **kwargs)

    # exp_list = [CHARM, BEBC, SHIP, FASER2, PROTODUNE_NP02, PROTODUNE_NP04, DUNE]

    # """
    #     LFC   FA vs ma
    # """
    # BP_NAME = "LFC_universal_vMC"
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
    # BP_NAME = "LFC_etau"
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
    # BP_NAME = "LFC_tauphilic"
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
    # BP_NAME = "anarchy_vMC"
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
    #     MICROBOONE,
    #     NOVA,
    #     CHARM,
    #     BEBC,
    #     SHIP,
    #     FASER,
    #     FASER2,
    #     PROTODUNE_NP02,
    #     PROTODUNE_NP04,
    #     DUNE,
    #     # TWOBYTWO,
    #     # TWOBYTWO_ABSORBER,
    # ]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFV HIEREACHY  FA vs ma
    # """
    # for lam in [0.05, 0.01, 0.005, 0.001]:
    #     BP_NAME = f"hierarchy_lambda_{lam}_clep_1"
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
    #         MICROBOONE,
    #         NOVA,
    #         CHARM,
    #         BEBC,
    #         SHIP,
    #         FASER,
    #         FASER2,
    #         PROTODUNE_NP02,
    #         PROTODUNE_NP04,
    #         DUNE,
    #         TWOBYTWO,
    #     ]
    #     run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFV ANARCHY  INVFA vs Bvis
    # """
    # for ma in [0.3, 1.0]:
    #     BP_NAME = f"anarchy_inv_ma_{ma}"
    #     lamb = 1
    #     c_lepton = np.array([[1, lamb**2, lamb], [lamb**2, 1, lamb], [lamb, lamb, 1]])

    #     kwargs = {
    #         "inv_fa_range": [1e-10, 1e-3],
    #         "Bvis_range": [1e-8, 1],
    #         "name": BP_NAME,
    #         "c_lepton": c_lepton,
    #         "ma_fixed": ma,
    #         "Npoints": 201,
    #     }

    #     sim_tools.make_rate_table_invfa_vs_Bvis(ICARUS, save=True, **kwargs)
    #     sim_tools.make_rate_table_invfa_vs_Bvis(MICROBOONE, save=True, **kwargs)
    #     sim_tools.make_rate_table_invfa_vs_Bvis(NOVA, save=True, **kwargs)
    #     sim_tools.make_rate_table_invfa_vs_Bvis(CHARM, save=True, **kwargs)
    #     sim_tools.make_rate_table_invfa_vs_Bvis(BEBC, save=True, **kwargs)
    #     sim_tools.make_rate_table_invfa_vs_Bvis(SHIP, save=True, **kwargs)
    #     sim_tools.make_rate_table_invfa_vs_Bvis(FASER, save=True, **kwargs)
    #     sim_tools.make_rate_table_invfa_vs_Bvis(FASER2, save=True, **kwargs)
    #     sim_tools.make_rate_table_invfa_vs_Bvis(PROTODUNE_NP02, save=True, **kwargs)
    #     sim_tools.make_rate_table_invfa_vs_Bvis(PROTODUNE_NP04, save=True, **kwargs)

    # """'
    #     LFV Ra=5 FA vs ma
    # """
    # BP_NAME = "LFV_Ra_5"
    # c_lepton = np.array([[5, 2, 2], [2, 5, 2], [2, 2, 5]])

    # kwargs = {
    #     "inv_fa_range": [1e-9, 1e-5],
    #     "ma_range": [1e-1, 1.7],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = [ARGONEUT, ARGONEUT_absorber]
    # run_simulations(exp_list, BP_NAME, **kwargs)

    # """'
    #     LFV Ra=1/3 FA vs ma
    # """
    # BP_NAME = "LFV_Ra_1o3"
    # c_lepton = np.array([[0.333, 2, 2], [2, 0.333, 2], [2, 2, 0.333]])

    # kwargs = {
    #     "inv_fa_range": [1e-9, 1e-5],
    #     "ma_range": [1e-1, 1.7],
    #     "name": BP_NAME,
    #     "c_lepton": c_lepton,
    #     "Npoints": NPOINTS,
    # }

    # exp_list = [ARGONEUT, ARGONEUT_absorber]
    # run_simulations(exp_list, BP_NAME, **kwargs)
