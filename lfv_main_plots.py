# %%
import numpy as np
import matplotlib.pyplot as plt

from alp.models import ALP
from alp import const
from alp import plot_tools as pt

# %%
fig, ax = pt.std_fig(figsize=(5, 5))

name = "anarchy"
lamb = 1
c_lepton = np.array([[1, lamb, lamb], [lamb, 1, lamb], [lamb, lamb, 1]])

pt.plot_other_limits(
    ax, c_lepton=c_lepton, linewidth=0.1, c_NN=0, mN=0.0, annotate=True
)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel(r"$1/f_a$ [GeV$^{-1}$]")
ax.set_xlabel(r"$m_a$ [GeV]")
ax.set_title(r"ALP lepton flavor agnostic ($c_{\ell_1 \ell_2}  = 1$)", fontsize=11)
#

ax.set_ylim(1e-10, 1e-3)
ax.set_xlim(1e-2, 2)
fig.savefig("plots/ALP_anarchy_otherlimits.pdf", bbox_inches="tight", dpi=300)

# %% [markdown]
# # Anarchical LFV

# %%
BP_NAME = "anarchy_vMC"
c_lepton = np.ones((3, 3))
pt.main_plot_LFV(
    BP_NAME,
    c_lepton,
    0,
    0,
    ymin=1e-11,
    figsize=(5, 6),
    loc="lower center",
    xmin=1e-2,
    linewidth=0.1,
    annotate=True,
    ncol=3,
    vlines=False,
)
pt.main_plot_LFV(
    BP_NAME,
    c_lepton,
    0,
    0,
    figsize=(5, 6),
    fa_power=1,
    ymin=1e-10,
    ymax=3e-8,
    xmin=0.1,
    xmax=1.8,
    ncol=3,
    loc="lower center",
    xscale="linear",
    legend=True,
    name_modifier="_fa4",
    vlines=False,
    linewidth=1,
)

# %%
BP_NAME = "anarchy"
c_lepton = np.ones((3, 3))
pt.main_plot_LFV(
    BP_NAME,
    c_lepton,
    0,
    0,
    ymin=1e-11,
    figsize=(5, 6),
    loc="lower center",
    xmin=1e-2,
    linewidth=0.1,
    annotate=True,
    ncol=3,
    vlines=False,
)
pt.main_plot_LFV(
    BP_NAME,
    c_lepton,
    0,
    0,
    figsize=(5, 6),
    fa_power=1,
    ymin=1e-10,
    ymax=3e-8,
    xmin=0.1,
    xmax=1.8,
    ncol=3,
    loc="lower center",
    xscale="linear",
    legend=True,
    name_modifier="_fa4",
    vlines=False,
    linewidth=1,
)

# %% [markdown]
# ## Hierarchical LFV

# %%
for lam in [0.05, 0.01, 0.005, 0.001]:
    BP_NAME = f"hierarchy_lambda_{lam}_clep_1"
    lamb = lam
    C_LEPTON = 1
    c_lepton = C_LEPTON * np.array([[1, lamb, lamb], [lamb, 1, lamb], [lamb, lamb, 1]])
    pt.main_plot_LFV(
        BP_NAME,
        c_lepton,
        0,
        0,
        ymin=1e-9,
        ymax=1e-3,
        loc="lower left",
        xmin=1e-2,
        linewidth=0.1,
        annotate=True,
        ncol=2,
        figsize=(5, 6),
        vlines=False,
    )

# %% [markdown]
# ## Invisible branching ratios

# %%
c_lepton = np.ones((3, 3))

BP_NAME = f"anarchy_inv_ma_0.3"
pt.make_Bvis_plot_LFV(BP_NAME, c_lepton, 0, 0, ma_fixed=0.3, ymin=1e-9, ymax=1e-4)

BP_NAME = f"anarchy_inv_ma_1.0"
pt.make_Bvis_plot_LFV(BP_NAME, c_lepton, 0, 0, ma_fixed=1.0, ymin=5e-10, ymax=1e-5)


# %% [markdown]
# ## Testing against Bertuzzo et al

# %%
BP_NAME = "LFV_Ra_5"
c_lepton = np.array([[5, 2, 2], [2, 5, 2], [2, 2, 5]])
fig, ax = pt.main_plot_LFV(
    BP_NAME,
    c_lepton,
    0,
    0,
    ymin=1e-9,
    ymax=1e-5,
    figsize=(5, 5),
    loc="upper center",
    xmin=1e-1,
    linewidth=0.1,
    annotate=True,
    ncol=2,
    title=r"{\bf LFV hierarchy} $\,\,\vert\,\, g_{\ell_1 \ell_2} = 1 \,\,\vert\,\, g_{\ell\ell} = 5$",
)


# %%
BP_NAME = "LFV_Ra_1o3"
c_lepton = np.array([[0.333, 2, 2], [2, 0.333, 2], [2, 2, 0.333]])
pt.main_plot_LFV(
    BP_NAME,
    c_lepton,
    0,
    0,
    ymin=1e-9,
    ymax=1e-5,
    figsize=(5, 5),
    loc="upper center",
    xmin=1e-1,
    linewidth=0.1,
    annotate=True,
    ncol=2,
    title=r"{\bf LFV hierarchy} $\,\,\vert\,\, g_{\ell_1 \ell_2} = 1 \,\,\vert\,\, g_{\ell\ell} = 1/3$",
)
