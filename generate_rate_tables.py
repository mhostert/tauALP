import multiprocessing as mp
import numpy as np
from DarkNews import const

from alp import exp, models, sim_tools
from functools import partial
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

# Pythia8 tau events
NUMI_files = [
    f"pythia8_events/tau_events_120GeV_HardOff_pT0.0001_{i}.txt" for i in range(0, 8)
] + [
    f"pythia8_events/tau_events_120GeV_HardOff_pT0.0001_v2_{i}.txt" for i in range(0, 8)
]

SPS_files = [
    f"pythia8_events/tau_events_120GeV_HardOff_pT0.0001_{i}.txt" for i in range(0, 8)
]
LHC_files = [f"pythia8_events/tau_events_LHC_13.6TeV_v6_{i}.txt" for i in range(0, 8)]


# Creating the experimental classes
ICARUS = exp.Experiment(NUMI_files, exp_dic=exp.ICARUS_exp, duplicate_taus=5)
MICROBOONE = exp.Experiment(NUMI_files, exp_dic=exp.MicroBooNE_exp, duplicate_taus=5)
NOVA = exp.Experiment(NUMI_files, exp_dic=exp.NoVA_exp, duplicate_taus=3)

DUNE = exp.Experiment(NUMI_files, exp_dic=exp.DUNE_exp, duplicate_taus=3)
TWOBYTWO = exp.Experiment(NUMI_files, exp_dic=exp.TwoByTwo_exp, duplicate_taus=3)
TWOBYTWO_ABSORBER = exp.Experiment(
    NUMI_files, exp_dic=exp.TwoByTwo_absorber_exp, duplicate_taus=3
)


ARGONEUT = exp.Experiment(NUMI_files, exp_dic=exp.ArgoNeuT_exp, duplicate_taus=10)
ARGONEUT_absorber = exp.Experiment(
    NUMI_files, exp_dic=exp.ArgoNeuT_absorber_exp, duplicate_taus=10
)

CHARM = exp.Experiment(SPS_files, exp_dic=exp.CHARM_exp)
BEBC = exp.Experiment(SPS_files, exp_dic=exp.BEBC_exp)
NA62 = exp.Experiment(SPS_files, exp_dic=exp.NA62_exp)
SHIP = exp.Experiment(SPS_files, exp_dic=exp.SHiP_exp, duplicate_taus=0.2)

PROTODUNE_NP02 = exp.Experiment(SPS_files, exp_dic=exp.PROTO_DUNE_NP02_exp)
PROTODUNE_NP04 = exp.Experiment(SPS_files, exp_dic=exp.PROTO_DUNE_NP04_exp)

FASER = exp.Experiment(LHC_files, exp_dic=exp.FASER_exp)
FASER2 = exp.Experiment(LHC_files, exp_dic=exp.FASER2_exp)

NPOINTS = 201


# def simulate(exp_case, **kwargs):
#     _ = sim_tools.make_rate_table(exp_case, save=True, **kwargs)


# def run_simulations(exp_list, **kwargs):
#     with Pool() as pool:
#         pool.map(partial(simulate, **kwargs), exp_list)


def simulate(args):
    exp_case, kwargs = args
    _ = sim_tools.make_rate_table(exp_case, save=True, **kwargs)


def run_simulations(exp_list, BP_NAME, **kwargs):
    print(f"Running simulations for {BP_NAME}...")
    args_list = [(exp_case, kwargs) for exp_case in exp_list]
    process_map(simulate, args_list, max_workers=None)


if __name__ == "__main__":
    mp.set_start_method("spawn")  # or "fork" if you're on Linux

    exp_list = [CHARM, BEBC, SHIP, FASER2, PROTODUNE_NP02, PROTODUNE_NP04, DUNE]

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

    # """'
    #     LFC   FA vs ma
    # """
    # BP_NAME = "LFC_universal"
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

    """'
        LFV ANARCHY   FA vs ma
    """
    BP_NAME = "anarchy"
    lamb = 1
    c_lepton = np.array([[1, lamb**2, lamb], [lamb**2, 1, lamb], [lamb, lamb, 1]])

    kwargs = {
        "inv_fa_range": [1e-10, 1e-3],
        "name": BP_NAME,
        "c_lepton": c_lepton,
        "Npoints": NPOINTS,
    }

    exp_list = [
        ICARUS,
        MICROBOONE,
        NOVA,
        CHARM,
        BEBC,
        SHIP,
        FASER,
        FASER2,
        PROTODUNE_NP02,
        PROTODUNE_NP04,
        DUNE,
        TWOBYTWO,
        TWOBYTWO_ABSORBER,
    ]
    run_simulations(exp_list, BP_NAME, **kwargs)

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
