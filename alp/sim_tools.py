from contextlib import nullcontext
import numpy as np

from . import const

from alp import exp, models
from tqdm import tqdm


def make_rate_table(
    EXP,
    save=True,
    inv_fa_range=[0.5e-9, 1e-4],
    ma_range=[1e-2, const.m_tau - const.m_e * 1.01],
    Npoints=101,
    c_lepton=False,
    name="",
    show_progress=True,
):
    m_alps = np.geomspace(*ma_range, Npoints, endpoint=True)
    # higher resolution around the dimuon threshold
    if ma_range[0] < 2 * const.m_mu and ma_range[1] > 2 * const.m_mu:
        m_alps = np.append(m_alps, 2 * const.m_mu + np.linspace(-10e-3, 10e-3, 20))
        m_alps = np.append(
            m_alps, const.m_mu + const.m_e + np.linspace(-5e-3, 5e-3, 20)
        )
        m_alps = np.sort(m_alps)
        inv_fas = np.geomspace(*inv_fa_range, Npoints + 40, endpoint=True)
    else:
        inv_fas = np.geomspace(*inv_fa_range, Npoints, endpoint=True)

    MA, INV_FA = np.meshgrid(m_alps, inv_fas)

    # Make sure the initial experiment is set up with the right kind of ALP
    # This makes sure we can use reweighting to get the rates
    EXP.alp = alp_1 = models.ALP(0.5, 1e7, c_lepton=c_lepton)

    z = []
    total_iterations = len(m_alps) * len(inv_fas)
    progress_ctx = (
        tqdm(total=total_iterations, desc=f"Rate table for {EXP.name}")
        if show_progress
        else nullcontext()
    )

    # with tqdm(total=total_iterations, desc=f"Rate table for {EXP.name}") as pbar:
    with progress_ctx as pbar:
        for ma in m_alps:
            alp_1 = models.ALP(ma, 1e7, c_lepton=c_lepton)
            EXP.get_event_rate(alp_1)
            for inv_fa in inv_fas:
                alp_2 = models.ALP(ma, 1 / inv_fa, c_lepton=c_lepton)
                r = EXP.reweight(alp_1, alp_2)
                # r = EXP.get_event_rate(alp_2)
                z.append(r)
                if show_progress:
                    pbar.update(1)

    Z = np.reshape(z, MA.shape).T

    if save:
        np.save(f"data/{EXP.name}_rates{name}.npy", [MA, INV_FA, Z])
    return MA, INV_FA, Z


def make_rate_table_invfa_vs_Bvis(
    EXP,
    save=True,
    inv_fa_range=[0.5e-9, 1e-4],
    Bvis_range=[1e-5, 1],
    ma_fixed=0.5,
    Npoints=101,
    c_lepton=False,
    name="",
    c_NN=0,
    mN=0,
):
    inv_fas = np.geomspace(*inv_fa_range, Npoints, endpoint=True)
    Bvis_alps = np.geomspace(*Bvis_range, Npoints, endpoint=True)
    BVIS, INV_FA = np.meshgrid(Bvis_alps, inv_fas)

    z = []
    total_iterations = len(Bvis_alps) * len(inv_fas)
    with tqdm(
        total=total_iterations, desc=f"Rate table invfa vs Bvis for {EXP.name}"
    ) as pbar:
        for Bvis in Bvis_alps:
            alp_1 = models.ALP(
                ma_fixed, 1e7, Bvis=Bvis, c_lepton=c_lepton, c_NN=c_NN, mN=mN
            )
            EXP.get_event_rate(alp_1)
            for inv_fa in inv_fas:
                alp_2 = models.ALP(
                    ma_fixed, 1 / inv_fa, Bvis=Bvis, c_lepton=c_lepton, c_NN=c_NN, mN=mN
                )
                r = EXP.reweight(alp_1, alp_2)
                # r = EXP.get_event_rate(alp_2)
                z.append(r)
                pbar.update(1)

    Z = np.reshape(z, BVIS.shape).T
    # print(Z.shape, MA.shape, INV_FA.shape)

    if save:
        np.save(f"data/invfa_vs_Bvis_{EXP.name}_rates_{name}.npy", [BVIS, INV_FA, Z])
    return BVIS, INV_FA, Z
