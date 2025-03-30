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


def v_2body(m_parent, m_a, m_l):
    mask = m_parent > m_l + m_a
    if isinstance(m_a, np.ndarray):
        v = np.zeros_like(m_a)
        v[mask] = np.sqrt(
            1
            - (2 * (m_l**2 + m_a[mask] ** 2) / m_parent**2)
            + ((m_l**2 - m_a[mask] ** 2) ** 2 / m_parent**4)
        )
    else:
        if m_parent > m_l + m_a:
            v = np.sqrt(
                1
                - (2 * (m_l**2 + m_a**2) / m_parent**2)
                + ((m_l**2 - m_a**2) ** 2 / m_parent**4)
            )
        else:
            v = 0

    return v


def F_lepton_2body(m_parent, m_a, m_l):
    """Phase space function for l2 --> a l1

    Args:
        m_parent: parent lepton mass
        m_a: alp mass
        m_l: final state lepton mass

    """
    term1 = (1 + m_l / m_parent) ** 2
    term2 = (1 - m_l / m_parent) ** 2 - (m_a**2 / m_parent**2)
    term3 = v_2body(m_parent, m_a, m_l)

    return term1 * term2 * term3


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
    def __init__(
        self,
        m_a,
        f_a,
        c_gg=0,
        c_NN=0,
        mN=0,
        Bvis=1,
        c_lepton=None,
    ):
        self.f_a = f_a
        self.m_a = m_a
        self.mN = mN
        self.Lambda_NP = 4 * np.pi * self.f_a

        # NOTE: democratic couplings
        if c_lepton is None:
            self.c_lepton = np.ones((3, 3))
        else:
            self.c_lepton = c_lepton

        self.c_gg = c_gg  # + self.c_gg_eff()
        self.c_BB = self.c_lepton.diagonal().sum() / 2
        # NOTE: Check the factor of 1/2
        self.c_WW = -(self.c_lepton.diagonal().sum()) / 2

        # Heavy neutrino couplings
        self.c_NN = c_NN

        # Visible decay widths
        self.Gamma_a_vis = (
            self.Gamma_a_to_ee()
            + self.Gamma_a_to_me()
            + self.Gamma_a_to_em()
            + self.Gamma_a_to_mm()
            + self.Gamma_a_to_gg()
            + self.Gamma_a_to_NN()
        )
        # Invisible decay width
        self.Bvis = Bvis
        self.Binv = 1 - self.Bvis
        self.Gamma_a_inv = self.Gamma_a_vis * (1 - self.Bvis) / self.Bvis
        self.Gamma_a = self.Gamma_a_vis + self.Gamma_a_inv

        # Channels are labeled as a -> l+ l-

        if len(np.array([self.Gamma_a])) > 1:
            self.BR_a_to_ee = np.empty_like(self.Gamma_a)
            self.BR_a_to_me = np.empty_like(self.Gamma_a)
            self.BR_a_to_me = np.empty_like(self.Gamma_a)
            self.BR_a_to_em = np.empty_like(self.Gamma_a)
            self.BR_a_to_mm = np.empty_like(self.Gamma_a)
            self.BR_a_to_gg = np.empty_like(self.Gamma_a)
            self.BR_a_to_NN = np.empty_like(self.Gamma_a)
            self.BR_a_to_inv = np.empty_like(self.Gamma_a)

            self.BR_a_to_ee[self.Gamma_a > 0] = (
                self.Gamma_a_to_ee()[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_ee[self.Gamma_a <= 0] = 0
            self.BR_a_to_me[self.Gamma_a > 0] = (
                self.Gamma_a_to_me()[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_me[self.Gamma_a <= 0] = 0
            self.BR_a_to_em[self.Gamma_a > 0] = (
                self.Gamma_a_to_em()[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_em[self.Gamma_a <= 0] = 0
            self.BR_a_to_mm[self.Gamma_a > 0] = (
                self.Gamma_a_to_mm()[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_mm[self.Gamma_a <= 0] = 0
            self.BR_a_to_gg[self.Gamma_a > 0] = (
                self.Gamma_a_to_gg()[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_gg[self.Gamma_a <= 0] = 0
            self.BR_a_to_NN[self.Gamma_a > 0] = (
                self.Gamma_a_to_NN()[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_NN[self.Gamma_a <= 0] = 0

            self.BR_a_to_inv[self.Gamma_a > 0] = (
                self.Gamma_a_inv[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_inv[self.Gamma_a > 0] = 0

        else:
            self.BR_a_to_ee = self.Gamma_a_to_ee() / self.Gamma_a * (self.Gamma_a > 0)
            self.BR_a_to_me = self.Gamma_a_to_me() / self.Gamma_a * (self.Gamma_a > 0)
            self.BR_a_to_em = self.Gamma_a_to_em() / self.Gamma_a * (self.Gamma_a > 0)
            self.BR_a_to_mm = self.Gamma_a_to_mm() / self.Gamma_a * (self.Gamma_a > 0)
            self.BR_a_to_gg = self.Gamma_a_to_gg() / self.Gamma_a * (self.Gamma_a > 0)
            self.BR_a_to_NN = self.Gamma_a_to_NN() / self.Gamma_a * (self.Gamma_a > 0)
            self.BR_a_to_inv = self.Gamma_a_inv / self.Gamma_a * (self.Gamma_a > 0)

        self.Ea_min = {
            "tau>nu+rho+a": self.m_a,
            "tau>nu+pi+a": self.m_a,
            "tau>nu+nu+e+a": self.m_a,
            "tau>nu+nu+mu+a": self.m_a,
            "tau>e+a": (const.m_tau**2 + self.m_a**2 - const.m_e**2) / 2 / const.m_tau,
            "tau>mu+a": (const.m_tau**2 + self.m_a**2 - const.m_mu**2)
            / 2
            / const.m_tau,
        }

        self.Ea_max = {
            "tau>nu+rho+a": (const.m_tau**2 + self.m_a**2 - const.m_charged_rho**2)
            / 2
            / const.m_tau,
            "tau>nu+pi+a": (const.m_tau**2 + self.m_a**2 - const.m_charged_pion**2)
            / 2
            / const.m_tau,
            "tau>nu+nu+e+a": const.m_tau,
            "tau>nu+nu+mu+a": const.m_tau,
            "tau>e+a": (const.m_tau**2 + self.m_a**2 - const.m_e**2) / 2 / const.m_tau,
            "tau>mu+a": (const.m_tau**2 + self.m_a**2 - const.m_mu**2)
            / 2
            / const.m_tau,
        }

    def prob_decay(self, Ea, L, dL):
        gamma = Ea / self.m_a
        beta = np.sqrt(1 - 1 / gamma**2)
        ell_dec = const.get_decay_rate_in_cm(self.Gamma_a) * gamma * beta
        p = np.empty_like(ell_dec)
        p[ell_dec > 0] = np.exp(-L / ell_dec[ell_dec > 0]) * (
            1 - np.exp(-dL / ell_dec[ell_dec > 0])
        )
        p[ell_dec <= 0] = 0
        return p

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

        if isinstance(self.m_a, np.ndarray):
            Gamma = np.zeros_like(self.m_a)

            Gamma[self.m_a > m_l1 + m_l2] = (
                np.sqrt(
                    1
                    - 2 * (m_l1**2 + m_l2**2) / self.m_a[self.m_a > m_l1 + m_l2] ** 2
                    + (m_l1**2 - m_l2**2) ** 2 / self.m_a[self.m_a > m_l1 + m_l2] ** 4
                )
                * term1[self.m_a > m_l1 + m_l2]
                * term2[self.m_a > m_l1 + m_l2]
            )
        else:
            if self.m_a > m_l1 + m_l2:
                Gamma = (
                    np.sqrt(
                        1
                        - 2 * (m_l1**2 + m_l2**2) / self.m_a**2
                        + (m_l1**2 - m_l2**2) ** 2 / self.m_a**4
                    )
                    * term1
                    * term2
                )
            else:
                Gamma = 0

        return Gamma

    def Gamma_a_to_ee(self):
        return self.c_lepton[0, 0] ** 2 * self.F_alp_2body(const.m_e, const.m_e)

    def Gamma_a_to_me(self):
        return self.c_lepton[0, 1] ** 2 * self.F_alp_2body(const.m_mu, const.m_e)

    def Gamma_a_to_em(self):
        return self.c_lepton[1, 0] ** 2 * self.F_alp_2body(const.m_e, const.m_mu)

    def Gamma_a_to_mm(self):
        return self.c_lepton[1, 1] ** 2 * self.F_alp_2body(const.m_mu, const.m_mu)

    def Gamma_a_to_gg(self):
        return 0 * self.m_a
        #     * self.c_gg**2
        #     * self.m_a**3
        #     * const.alphaQED**2
        #     / 64
        #     / np.pi**3
        #     / self.f_a**2
        # )

    def Gamma_a_to_NN(self):
        return self.c_NN**2 * self.F_alp_2body(self.mN, self.mN)

    def alp_visible_BR(self, final_states):
        return (
            self.BR_a_to_ee * ("ee" in final_states)
            + self.BR_a_to_em * ("em" in final_states)
            + self.BR_a_to_mm * ("mm" in final_states)
            + self.BR_a_to_me * ("me" in final_states)
        )

    def tau_BR(self, production_channel):
        """Branching ratio for tau -> a + l"""
        if production_channel == "tau>e+a":
            return self.BR_tau_to_a_e()
        elif production_channel == "tau>mu+a":
            return self.BR_tau_to_a_mu()
        elif production_channel == "tau>nu+pi+a":
            return self.BR_tau_to_pi_nu_a()
        elif production_channel == "tau>nu+rho+a":
            return self.BR_tau_to_rho_nu_a()
        elif production_channel == "tau>nu+nu+e+a":
            return self.BR_tau_to_e_nu_nu_a()
        elif production_channel == "tau>nu+nu+mu+a":
            return self.BR_tau_to_mu_nu_nu_a()
        else:
            raise ValueError("Unknown production channel")

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

    def BR_tau_to_pi_nu_a(self):
        return self.c_lepton[2, 2] ** 2 * 1.5e-12 * (1e4 / self.f_a) ** 2

    def BR_tau_to_rho_nu_a(self):
        return self.c_lepton[2, 2] ** 2 * 2.3e-12 * (1e4 / self.f_a) ** 2

    def BR_tau_to_e_nu_nu_a(self):
        return self.c_lepton[2, 2] ** 2 * 1e-12 * (1e4 / self.f_a) ** 2

    def diff_BR_tau_to_pi_nu_a(self, Ea):
        """Differential decay rate for tau -> pi nu a"""

        mtau_barSQR = const.m_tau**2 + self.m_a**2 - 2 * const.m_tau * Ea
        dGammadE = (
            self.c_lepton[2, 2] ** 2
            * const.Gf**2
            * const.m_tau**2
            * const.fcharged_pion**2
            * np.abs(const.Vud) ** 2
            * (mtau_barSQR - const.m_charged_pion**2) ** 2
            / 256
            / np.pi**3
            / self.f_a**2
            / mtau_barSQR
            * (
                1
                - self.m_a**2
                * (const.m_tau**2 + mtau_barSQR)
                / (const.m_tau**2 - mtau_barSQR) ** 2
            )
            * v_2body(const.m_tau, np.sqrt(mtau_barSQR), self.m_a)
        )
        return (
            dGammadE
            * (Ea > self.Ea_min("tau>nu+pi+a"))
            * (Ea < self.Ea_max("tau>nu+pi+a"))
        ) / Gamma_tau

    def diff_BR_tau_to_rho_nu_a(self, Ea):
        """Differential decay rate for tau -> rho nu a"""
        mtau_barSQR = const.m_tau**2 + self.m_a**2 - 2 * const.m_tau * Ea
        dGammadE = (
            self.c_lepton[2, 2] ** 2
            * const.Gf**2
            * const.m_tau**2
            * const.fcharged_rho**2
            * np.abs(const.Vud) ** 2
            * (mtau_barSQR + 2 * const.m_charged_rho**2)
            * (1 - const.m_charged_rho**2 / mtau_barSQR) ** 2
            / 256
            / np.pi**3
            / self.f_a**2
            / mtau_barSQR
            * (
                1
                - self.m_a**2
                * (const.m_tau**2 + mtau_barSQR)
                / (const.m_tau**2 - mtau_barSQR) ** 2
            )
            * v_2body(const.m_tau, np.sqrt(mtau_barSQR), self.m_a)
        )
        return (
            dGammadE
            * (Ea > self.Ea_min("tau>nu+rho+a"))
            * (Ea < self.Ea_max("tau>nu+rho+a"))
        ) / Gamma_tau
