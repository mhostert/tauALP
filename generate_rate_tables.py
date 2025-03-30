import numpy as np

from alp import exp, models, sim_tools

# Pythia8 tau events
NUMI_files = [f"pythia8_events/tau_events_NuMI_120GeV_v3_{i}.txt" for i in range(0, 8)]
SPS_files = [f"pythia8_events/tau_events_SPS_400GeV_v3_{i}.txt" for i in range(0, 8)]
LHC_files = [f"pythia8_events/tau_events_LHC_13.6TeV_v6_{i}.txt" for i in range(0, 8)]


# Creating the experimental classes
ICARUS = exp.Experiment(NUMI_files, exp_dic=exp.ICARUS_exp)
MICROBOONE = exp.Experiment(NUMI_files, exp_dic=exp.MicroBooNE_exp)
NOVA = exp.Experiment(NUMI_files, exp_dic=exp.NoVA_exp)

CHARM = exp.Experiment(SPS_files, exp_dic=exp.CHARM_exp)
BEBC = exp.Experiment(SPS_files, exp_dic=exp.BEBC_exp)
NA62 = exp.Experiment(SPS_files, exp_dic=exp.NA62_exp)
SHIP = exp.Experiment(SPS_files, exp_dic=exp.SHiP_exp)

PROTODUNE_NP02 = exp.Experiment(SPS_files, exp_dic=exp.PROTO_DUNE_NP02_exp)
PROTODUNE_NP04 = exp.Experiment(SPS_files, exp_dic=exp.PROTO_DUNE_NP04_exp)

FASER = exp.Experiment(LHC_files, exp_dic=exp.FASER_exp)
FASER2 = exp.Experiment(LHC_files, exp_dic=exp.FASER2_exp)


"""'
    ANARCHY   FA vs ma
"""
BP_NAME = "anarchy"
lamb = 1
c_lepton = np.array([[1, lamb**2, lamb], [lamb**2, 1, lamb], [lamb, lamb, 1]])

kwargs = {
    "inv_fa_range": [1e-10, 1e-3],
    "name": BP_NAME,
    "c_lepton": c_lepton,
    "c_NN": 0,
    "mN": 0,
    "Npoints": 31,
}

sim_tools.make_rate_table(ICARUS, save=True, **kwargs)
sim_tools.make_rate_table(MICROBOONE, save=True, **kwargs)
sim_tools.make_rate_table(NOVA, save=True, **kwargs)
sim_tools.make_rate_table(CHARM, save=True, **kwargs)
sim_tools.make_rate_table(BEBC, save=True, **kwargs)
# sim_tools.make_rate_table(NA62,       save=True, **kwargs)
sim_tools.make_rate_table(SHIP, save=True, **kwargs)
sim_tools.make_rate_table(FASER, save=True, **kwargs)
sim_tools.make_rate_table(FASER2, save=True, **kwargs)
sim_tools.make_rate_table(PROTODUNE_NP02, save=True, **kwargs)
sim_tools.make_rate_table(PROTODUNE_NP04, save=True, **kwargs)

# """'
#     HIEREACHY  FA vs ma
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
#         "c_NN": 0,
#         "mN": 0,
#         "Npoints": 201,
#     }

#     sim_tools.make_rate_table(ICARUS, save=True, **kwargs)
#     sim_tools.make_rate_table(MICROBOONE, save=True, **kwargs)
#     sim_tools.make_rate_table(NOVA, save=True, **kwargs)
#     sim_tools.make_rate_table(CHARM, save=True, **kwargs)
#     sim_tools.make_rate_table(BEBC, save=True, **kwargs)
#     # sim_tools.make_rate_table(NA62,       save=True, **kwargs)
#     sim_tools.make_rate_table(SHIP, save=True, **kwargs)
#     sim_tools.make_rate_table(FASER, save=True, **kwargs)
#     sim_tools.make_rate_table(FASER2, save=True, **kwargs)
#     sim_tools.make_rate_table(PROTODUNE_NP02, save=True, **kwargs)
#     sim_tools.make_rate_table(PROTODUNE_NP04, save=True, **kwargs)


# """'
#     ANARCHY   INVFA vs Bvis
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
#         "c_NN": 0,
#         "mN": 0,
#         "ma_fixed": ma,
#         "Npoints": 201,
#     }

#     sim_tools.make_rate_table_invfa_vs_Bvis(ICARUS, save=True, **kwargs)
#     sim_tools.make_rate_table_invfa_vs_Bvis(MICROBOONE, save=True, **kwargs)
#     sim_tools.make_rate_table_invfa_vs_Bvis(NOVA, save=True, **kwargs)
#     sim_tools.make_rate_table_invfa_vs_Bvis(CHARM, save=True, **kwargs)
#     sim_tools.make_rate_table_invfa_vs_Bvis(BEBC, save=True, **kwargs)
#     # sim_tools.make_rate_table_invfa_vs_Bvis(NA62,       save=True, **kwargs)
#     sim_tools.make_rate_table_invfa_vs_Bvis(SHIP, save=True, **kwargs)
#     sim_tools.make_rate_table_invfa_vs_Bvis(FASER, save=True, **kwargs)
#     sim_tools.make_rate_table_invfa_vs_Bvis(FASER2, save=True, **kwargs)
#     sim_tools.make_rate_table_invfa_vs_Bvis(PROTODUNE_NP02, save=True, **kwargs)
#     sim_tools.make_rate_table_invfa_vs_Bvis(PROTODUNE_NP04, save=True, **kwargs)
