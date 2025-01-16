import numpy as np
from scipy.integrate import quad, dblquad

from DarkNews import const

Gamma_tau = const.get_decay_rate_in_s(2.903e-13)  # GeV is the width of the Tau lepton
Gamma_mu = const.get_decay_rate_in_s(2.196e-6)  # GeV is the width of the Muon lepton
# def g_ps(x):
#     return 1 - 8 * x - 12 * x**2 * np.log(x) + 8 * x**3 - x**4
# Gamma_mu = (
#     const.m_mu**5 * const.Gf**2 / 192 / np.pi**3 * g_ps(const.m_e**2 / const.m_mu**2)
# )


def F_lepton_2body(m_parent, m_a, m_l):
    """Phase space function for l2 --> a l1

    Args:
        m_parent: parent lepton mass
        m_a: alp mass
        m_l: final state lepton mass

    """
    term1 = (1 + m_l / m_parent) ** 2
    term2 = (1 - m_l / m_parent) ** 2 - (m_a**2 / m_parent**2)
    term3 = np.sqrt(
        1
        - (2 * (m_l**2 + m_a**2) / m_parent**2)
        + ((m_l**2 - m_a**2) ** 2 / m_parent**4)
    )

    return np.where(m_parent > m_l + m_a, term1 * term2 * term3, 0)


def g1_analytic(r):
    return (
        2 * r ** (3 / 2) * np.sqrt(4 - r) * np.arccos(np.sqrt(r) / 2)
        + 1
        - 2 * r
        + r**2 * (3 - r) / (1 - r) * np.log(r)
    )


def g1_numeric(r):
    res, _ = quad(
        lambda x: 1
        - x
        + 2 * r * np.arctanh(((-1 + x) * x) / (2 + x * (-3 + 2 * r + x))),
        0,
        1,
    )
    return 2 * res


g1_numeric_vec = np.vectorize(g1_numeric)


def g1(ma, mi, f_a):
    r = (ma / mi) ** 2
    return np.piecewise(r, [r < 1, r >= 1], [g1_analytic, g1_numeric_vec])


def g2_analytic(ma, mi, f_a):
    Lam = 4 * np.pi * f_a
    delta2 = -3
    x = (ma / mi) ** 2 * (1 + 0j)
    return (
        2 * np.log(Lam**2 / mi**2)
        + 2 * delta2
        + 4
        - x**2 * np.log(x) / (x - 1)
        + (x - 1) * np.log(x - 1)
    )


def g2_numeric(ma, mi, f_a):
    Lambda_NP = 4 * np.pi * f_a
    delta2 = -3
    r = (ma / mi) ** 2
    res, _ = dblquad(
        lambda y, x: (
            -4 * delta2
            - (x * (1 - y)) / (x * (1 - y) + r * (1 - x - y))
            + (x * y) / (r * y - x * y)
            + 4 * np.log(mi**2 / Lambda_NP**2)
            + 2 * np.log(x * (1 - y) + r * (1 - x - y))
            # + 2 * np.log(r * y - x * y)
        ),
        0,
        1,
        0,
        lambda x: 1 - x,
    )

    return -res


g2_numeric_vec = np.vectorize(g2_numeric)


# def g2(ma, mi, f_a):
#     return np.piecewise(
#         ma, [ma < mi, ma >= mi], [g2_numeric_vec, g2_analytic], mi=mi, f_a=f_a
#     )


def g2(ma, mi, f_a):
    return g2_analytic(ma, mi, f_a)


def B1_loop(tau):
    f = np.piecewise(
        tau,
        [tau >= 1, tau < 1],
        [
            lambda tau: np.arcsin(1 / np.sqrt(tau)),
            lambda tau: np.pi / 2
            + 1j / 2 * np.log((1 + np.sqrt(1 - tau)) / (1 - np.sqrt(1 - tau))),
        ],
    )
    return 1 - tau * np.abs(f) ** 2


class ALP:
    def __init__(self, m_a, f_a, c_gg=0):
        self.f_a = f_a
        self.m_a = m_a
        self.Lambda_NP = 4 * np.pi * self.f_a

        # NOTE: democratic couplings
        self.c_lepton = np.ones((3, 3))

        self.c_gg = c_gg  # + self.c_gg_eff()
        self.c_BB = self.c_lepton.diagonal().sum() / 2

        # NOTE: Check the factor of 1/2
        self.c_WW = -(self.c_lepton.diagonal().sum()) / 2

        self.Gamma_a = (
            self.Gamma_a_to_ee()
            + self.Gamma_a_to_me()
            + self.Gamma_a_to_em()
            + self.Gamma_a_to_mm()
            + self.Gamma_a_to_gg()
        )

        # Channels are labeled as a -> l+ l-
        self.BR_a_to_ee = self.Gamma_a_to_ee() / self.Gamma_a
        self.BR_a_to_me = self.Gamma_a_to_me() / self.Gamma_a
        self.BR_a_to_em = self.Gamma_a_to_em() / self.Gamma_a
        self.BR_a_to_mm = self.Gamma_a_to_mm() / self.Gamma_a
        self.BR_a_to_gg = self.Gamma_a_to_gg() / self.Gamma_a

    def prob_decay(self, p_avg, L, dL):
        gamma = np.sqrt(p_avg**2 + self.m_a**2) / self.m_a
        beta = np.sqrt(1 - 1 / gamma**2)
        ell_dec = const.get_decay_rate_in_cm(self.Gamma_a) * gamma * beta
        return np.exp(-L / ell_dec) * (1 - np.exp(-dL / ell_dec))

    def c_gg_eff(self):
        return (
            self.c_lepton[0, 0] * B1_loop((2 * const.m_e / self.m_a) ** 2)
            + self.c_lepton[1, 1] * B1_loop((2 * const.m_mu / self.m_a) ** 2)
            + self.c_lepton[2, 2] * B1_loop((2 * const.m_tau / self.m_a) ** 2)
        )

    def F_alp_2body(self, m_l1, m_l2):
        """two body phase space of alp decays"""
        term1 = self.m_a * (m_l1 + m_l2) ** 2 / 32 / np.pi / self.f_a**2
        term2 = 1 - (m_l1 - m_l2) ** 2 / self.m_a**2
        term3 = (
            1
            - 2 * (m_l1**2 + m_l2**2) / self.m_a**2
            + (m_l1**2 - m_l2**2) ** 2 / self.m_a**4
        ) ** 0.5

        return np.where(self.m_a > m_l1 + m_l2, term1 * term2 * term3, 0.0).real

    def Gamma_a_to_ee(self):
        return self.c_lepton[0, 0] ** 2 * self.F_alp_2body(const.m_e, const.m_e)

    def Gamma_a_to_me(self):
        return self.c_lepton[0, 1] ** 2 * self.F_alp_2body(const.m_mu, const.m_e)

    def Gamma_a_to_em(self):
        return self.c_lepton[1, 0] ** 2 * self.F_alp_2body(const.m_e, const.m_mu)

    def Gamma_a_to_mm(self):
        return self.c_lepton[1, 1] ** 2 * self.F_alp_2body(const.m_mu, const.m_mu)

    def Gamma_a_to_gg(self):
        return 0
        #     * self.c_gg**2
        #     * self.m_a**3
        #     * const.alphaQED**2
        #     / 64
        #     / np.pi**3
        #     / self.f_a**2
        # )

    def BR_tau_to_a_mu(self):
        return (
            self.c_lepton[2, 1] ** 2
            / 64
            / np.pi
            * const.m_tau**3
            / self.f_a**2
            * F_lepton_2body(const.m_tau, self.m_a, const.m_mu)
            / Gamma_tau
        )

    def BR_tau_to_a_e(self):
        return (
            self.c_lepton[2, 0] ** 2
            / 64
            / np.pi
            * const.m_tau**3
            / self.f_a**2
            * F_lepton_2body(const.m_tau, self.m_a, const.m_e)
            / Gamma_tau
        )

    def BR_mu_to_a_e(self):
        return (
            self.c_lepton[1, 0] ** 2
            / 64
            / np.pi
            * const.m_mu**3
            / self.f_a**2
            * F_lepton_2body(const.m_mu, self.m_a, const.m_e)
            / Gamma_mu
        )
