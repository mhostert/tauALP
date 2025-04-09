import numpy as np
from scipy.integrate import quad, dblquad

from . import const

lepton_masses = [const.m_e, const.m_mu, const.m_tau]
Gamma_tau = const.get_decay_rate_in_s(2.903e-13)  # GeV is the width of the Tau lepton
Gamma_mu = const.get_decay_rate_in_s(2.196e-6)  # GeV is the width of the Muon lepton
lepton_gammas = [0, Gamma_mu, Gamma_tau]
# def g_ps(x):
#     return 1 - 8 * x - 12 * x**2 * np.log(x) + 8 * x**3 - x**4
# Gamma_mu = (
#     const.m_mu**5 * const.Gf**2 / 192 / np.pi**3 * g_ps(const.m_e**2 / const.m_mu**2)
# )


def v_2body(m_parent, m_a, m_l):
    mask = (m_parent >= m_l + m_a) & (
        (
            1
            - (2 * (m_l**2 + m_a**2) / m_parent**2)
            + ((m_l**2 - m_a**2) ** 2 / m_parent**4)
        )
        > 0
    )
    if isinstance(m_a, np.ndarray):
        v = np.zeros_like(m_a)
        v[mask] = np.sqrt(
            1
            - (2 * (m_l**2 + m_a[mask] ** 2) / m_parent**2)
            + ((m_l**2 - m_a[mask] ** 2) ** 2 / m_parent**4)
        )
    else:
        if (m_parent >= m_l + m_a) & (
            (
                1
                - (2 * (m_l**2 + m_a**2) / m_parent**2)
                + ((m_l**2 - m_a**2) ** 2 / m_parent**4)
            )
            > 0
        ):
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


def l1_numeric(q, ma, ml):
    res, _ = dblquad(
        lambda y, x: (1 - x - y * x - 2 * y**2)
        / (x * (ma / ml) ** 2 + (1 - x - y * x) - q**2 / ml**2 * y * (1 - x - y)),
        a=0,
        b=1,
        gfun=lambda x: 0,
        hfun=lambda x: 1 - x,
    )
    return 2 * res


def l2_numeric(q, ma, ml):
    res, _ = dblquad(
        lambda y, x: (x * (1 - y))
        / ((1 - x - y) * (ma / ml) ** 2 - q**2 / ml**2 * y * (1 - x - y) + x * (1 - y))
        - y * x / (y * (ma / ml) ** 2 - q**2 / ml**2 * y * (1 - x - y) - x * y),
        a=0,
        b=1,
        gfun=lambda x: 0,
        hfun=lambda x: 1 - x,
    )
    return res


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


# def g2(ma, mi, f_a):
#     return np.piecewise(
#         ma, [ma < mi, ma >= mi], [g2_numeric_vec, g2_analytic], mi=mi, f_a=f_a
#     )

l1_numeric_vec = np.vectorize(l1_numeric)
l2_numeric_vec = np.vectorize(l2_numeric)
g1_numeric_vec = np.vectorize(g1_numeric)
g2_numeric_vec = np.vectorize(g2_numeric)


def g1(ma, mi):
    r = (ma / mi) ** 2
    return np.piecewise(r, [r < 1, r >= 1], [g1_analytic, g1_numeric_vec])


def g2(ma, mi, f_a):
    return g2_analytic(ma, mi, f_a)


def l1(q, ma, mi):
    return l1_numeric_vec(q, ma, mi)


def l2(q, ma, mi):
    return l2_numeric_vec(q, ma, mi)


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

        self.c_gg = c_gg + self.c_gg_eff()
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
            + self.Gamma_a_to_te()
            + self.Gamma_a_to_et()
            + self.Gamma_a_to_tm()
            + self.Gamma_a_to_mt()
            + self.Gamma_a_to_tt()
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
            self.BR_a_to_te = np.empty_like(self.Gamma_a)
            self.BR_a_to_et = np.empty_like(self.Gamma_a)
            self.BR_a_to_tm = np.empty_like(self.Gamma_a)
            self.BR_a_to_mt = np.empty_like(self.Gamma_a)
            self.BR_a_to_tt = np.empty_like(self.Gamma_a)
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
            self.BR_a_to_te[self.Gamma_a > 0] = (
                self.Gamma_a_to_te()[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_te[self.Gamma_a <= 0] = 0
            self.BR_a_to_et[self.Gamma_a > 0] = (
                self.Gamma_a_to_et()[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_et[self.Gamma_a <= 0] = 0

            self.BR_a_to_tm[self.Gamma_a > 0] = (
                self.Gamma_a_to_tm()[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_tm[self.Gamma_a <= 0] = 0
            self.BR_a_to_mt[self.Gamma_a > 0] = (
                self.Gamma_a_to_mt()[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_mt[self.Gamma_a <= 0] = 0

            self.BR_a_to_tt[self.Gamma_a > 0] = (
                self.Gamma_a_to_tt()[self.Gamma_a > 0] / self.Gamma_a[self.Gamma_a > 0]
            )
            self.BR_a_to_tt[self.Gamma_a <= 0] = 0

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
            self.BR_a_to_et = self.Gamma_a_to_et() / self.Gamma_a * (self.Gamma_a > 0)
            self.BR_a_to_te = self.Gamma_a_to_te() / self.Gamma_a * (self.Gamma_a > 0)
            self.BR_a_to_tm = self.Gamma_a_to_tm() / self.Gamma_a * (self.Gamma_a > 0)
            self.BR_a_to_mt = self.Gamma_a_to_mt() / self.Gamma_a * (self.Gamma_a > 0)
            self.BR_a_to_tt = self.Gamma_a_to_tt() / self.Gamma_a * (self.Gamma_a > 0)
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
            "tau>nu+nu+e+a": (const.m_tau**2 + self.m_a**2 - const.m_mu**2)
            / 2
            / const.m_tau,
            "tau>nu+nu+mu+a": (const.m_tau**2 + self.m_a**2 - const.m_mu**2)
            / 2
            / const.m_tau,
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
        # p[ell_dec > 0] = np.exp(-L / ell_dec[ell_dec > 0]) * (
        # 1 - np.exp(-dL / ell_dec[ell_dec > 0])
        # )
        p = np.exp(-L / ell_dec) * (1 - np.exp(-dL / ell_dec))
        # p[ell_dec <= 0] = 0
        return p

    def c_gg_eff(self):
        if isinstance(self.m_a, np.ndarray):
            c = np.zeros_like(self.m_a)
            c[self.m_a > 0] = (
                self.c_lepton[0, 0]
                * B1_loop((2 * const.m_e / self.m_a[self.m_a > 0]) ** 2)
                + self.c_lepton[1, 1]
                * B1_loop((2 * const.m_mu / self.m_a[self.m_a > 0]) ** 2)
                + self.c_lepton[2, 2]
                * B1_loop((2 * const.m_tau / self.m_a[self.m_a > 0]) ** 2)
            )
            return c
        else:
            if self.m_a:
                return (
                    self.c_lepton[0, 0] * B1_loop((2 * const.m_e / self.m_a) ** 2)
                    + self.c_lepton[1, 1] * B1_loop((2 * const.m_mu / self.m_a) ** 2)
                    + self.c_lepton[2, 2] * B1_loop((2 * const.m_tau / self.m_a) ** 2)
                )
            else:
                return 0

    def F_alp_2body(self, m_l1, m_l2):
        """two body phase space of alp decays"""
        term1 = self.m_a * (m_l1 + m_l2) ** 2 / 32 / np.pi / self.f_a**2
        if isinstance(self.m_a, np.ndarray):
            term2 = np.zeros_like(self.m_a)
            term2[self.m_a > m_l1 + m_l2] = (
                1 - (m_l1 - m_l2) ** 2 / self.m_a[self.m_a > m_l1 + m_l2] ** 2
            )
        else:
            if self.m_a > m_l1 + m_l2:
                term2 = 1 - (m_l1 - m_l2) ** 2 / self.m_a**2
            else:
                term2 = 0

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

    def Gamma_a_to_et(self):
        return self.c_lepton[0, 2] ** 2 * self.F_alp_2body(const.m_e, const.m_tau)

    def Gamma_a_to_te(self):
        return self.c_lepton[2, 0] ** 2 * self.F_alp_2body(const.m_tau, const.m_e)

    def Gamma_a_to_mt(self):
        return self.c_lepton[1, 2] ** 2 * self.F_alp_2body(const.m_mu, const.m_tau)

    def Gamma_a_to_tm(self):
        return self.c_lepton[2, 1] ** 2 * self.F_alp_2body(const.m_tau, const.m_mu)

    def Gamma_a_to_tt(self):
        return self.c_lepton[2, 2] ** 2 * self.F_alp_2body(const.m_tau, const.m_tau)

    def Gamma_a_to_gg(self):
        return (
            self.c_gg**2 * self.m_a**3 * const.alphaQED**2 / 64 / np.pi**3 / self.f_a**2
        )

    def Gamma_a_to_NN(self):
        return self.c_NN**2 * self.F_alp_2body(self.mN, self.mN)

    def alp_visible_BR(self, final_states):
        return (
            self.BR_a_to_ee * ("ee" in final_states)
            + self.BR_a_to_em * ("em" in final_states)
            + self.BR_a_to_mm * ("mm" in final_states)
            + self.BR_a_to_me * ("me" in final_states)
            + self.BR_a_to_et * ("et" in final_states)
            + self.BR_a_to_te * ("te" in final_states)
            + self.BR_a_to_tm * ("tm" in final_states)
            + self.BR_a_to_mt * ("mt" in final_states)
            + self.BR_a_to_tt * ("tt" in final_states)
            + self.BR_a_to_gg * ("gg" in final_states)
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
            raise ValueError(f"Unknown production channel {production_channel}")

    def tau_diff_BR(self, Ea, production_channel):
        """Differential Branching ratio for tau decay wrt Ealp"""
        if production_channel == "tau>nu+pi+a":
            return self.diff_BR_tau_to_pi_nu_a(Ea)
        elif production_channel == "tau>nu+rho+a":
            return self.diff_BR_tau_to_rho_nu_a(Ea)
        elif production_channel == "tau>nu+nu+e+a":
            return self.diff_BR_tau_to_nu_nu_e_a(Ea)
        elif production_channel == "tau>nu+nu+mu+a":
            return self.diff_BR_tau_to_nu_nu_mu_a(Ea)
        else:
            raise ValueError(f"Unknown production channel {production_channel}")

    def BR_tau_to_a_mu(self):
        return self.BR_li_to_lj_a(2, 1)

    def BR_tau_to_a_e(self):
        return self.BR_li_to_lj_a(2, 0)

    def BR_li_to_lj_a(self, l_i, l_j):
        return (
            self.c_lepton[l_i, l_j] ** 2
            / 64
            / np.pi
            * lepton_masses[l_i] ** 3
            / self.f_a**2
            * F_lepton_2body(lepton_masses[l_i], self.m_a, lepton_masses[l_j])
            / lepton_gammas[l_i]
        )

    """
    BEGIN approximate from Yohei's notes
    """

    def BR_tau_to_pi_nu_a(self):
        return self.c_lepton[2, 2] ** 2 * 1.5e-12 * (1e4 / self.f_a) ** 2

    def BR_tau_to_rho_nu_a(self):
        return self.c_lepton[2, 2] ** 2 * 2.3e-12 * (1e4 / self.f_a) ** 2

    def BR_tau_to_e_nu_nu_a(self):
        return self.c_lepton[2, 2] ** 2 * 1e-12 * (1e4 / self.f_a) ** 2

    def BR_tau_to_mu_nu_nu_a(self):
        return self.c_lepton[2, 2] ** 2 * 1e-12 * (1e4 / self.f_a) ** 2

    """
    END approximate from Yohei's notes
    """

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
            * (Ea > self.Ea_min["tau>nu+pi+a"])
            * (Ea < self.Ea_max["tau>nu+pi+a"])
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
            * (Ea >= self.Ea_min["tau>nu+rho+a"])
            * (Ea <= self.Ea_max["tau>nu+rho+a"])
        ) / Gamma_tau

    def diff_BR_tau_to_nu_nu_e_a(self, Ea):
        """Differential decay rate for tau -> rho nu a"""
        mtau_barSQR = const.m_tau**2 + self.m_a**2 - 2 * const.m_tau * Ea
        dGammadE = (
            self.c_lepton[2, 2] ** 2
            * const.Gf**2
            * const.m_tau**2
            * mtau_barSQR
            / 1536
            / np.pi**5
            / self.f_a**2
            * Iv_mvv_integral(Ea, self.m_a, mlep=const.m_e)
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
            * (Ea > self.Ea_min["tau>nu+nu+e+a"])
            * (Ea < self.Ea_max["tau>nu+nu+e+a"])
        ) / Gamma_tau

    def diff_BR_tau_to_nu_nu_mu_a(self, Ea):
        """Differential decay rate for tau -> rho nu a"""
        mtau_barSQR = const.m_tau**2 + self.m_a**2 - 2 * const.m_tau * Ea
        dGammadE = (
            self.c_lepton[2, 2] ** 2
            * const.Gf**2
            * const.m_tau**2
            * mtau_barSQR
            / 1536
            / np.pi**5
            / self.f_a**2
            * Iv_mvv_integral(Ea, self.m_a, mlep=const.m_mu)
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
            * (Ea > self.Ea_min["tau>nu+nu+mu+a"])
            * (Ea < self.Ea_max["tau>nu+nu+mu+a"])
        ) / Gamma_tau

    def BR_li_to_lj_gamma(self, l_i, l_j):
        """Branching ratio for li -> lj + gamma"""

        Gamma = (
            const.m_mu**3
            / 8
            / np.pi
            * (1 - (lepton_masses[l_j] / lepton_masses[l_i]) ** 2)
        )
        FFSQR = np.abs(
            self.F2_5(self.m_a, l_i=l_i, l_j=l_j) ** 2
            + np.abs(self.F2(self.m_a, l_i=l_i, l_j=l_j)) ** 2
        )
        return Gamma * FFSQR / Gamma_mu

    def BR_diff_li_to_lj_lk_lk(self, s12, s23, l_i, l_j, l_k):
        """Differential decay rate for mu -> e + a + gamma
        Args:
            s12: invariant mass of the e + a system
            s23: invariant mass of the a + gamma system
        """

        F2_s23 = self.F2(s23, l_i=l_i, l_j=l_j)
        F3_s23 = self.F3(s23, l_i=l_i, l_j=l_j)

        F2_5_s23 = self.F2_5(s23, l_i=l_i, l_j=l_j)
        F3_5_s23 = self.F3_5(s23, l_i=l_i, l_j=l_j)

        s13 = (
            self.m_a**2 + lepton_masses[l_i] ** 2 + lepton_masses[l_j] ** 2 - s12 - s23
        )

        # Denominator factors like (s23 - m_a^2 + i m_a Gamma_a)
        denom_s23 = s23 - self.m_a**2 + 1j * self.m_a * self.Gamma_a
        denom_s13_p = s13 - self.m_a**2 + 1j * self.m_a * self.Gamma_a
        denom_s13_m = s13 - self.m_a**2 - 1j * self.m_a * self.Gamma_a

        # Absolute square of denom_s23:
        abs_denom_s23_sq = abs(denom_s23) ** 2

        # Real part of something like (denom_s23 * denom_s13_m):
        re_s23_s13 = (denom_s23 * denom_s13_m).real

        #
        # --- Overall prefactors: (|ke_12|^2 + |kE_12|^2)*|c_ee|^2*(m_e^2 m_mu^2 / f_a^4)
        #
        # If ke_12 etc. are real in your notation, just do ke_12**2.
        # If they might be complex, you could do abs(ke_12)**2.
        #
        prefactor = (
            (self.c_lepton[l_i, l_j] ** 2 + self.c_lepton[l_j, l_i] ** 2)
            * abs(self.c_lepton[l_i, l_i]) ** 2
            * lepton_masses[l_j] ** 2
            * lepton_masses[l_i] ** 2
            / self.f_a**4
        )

        #
        # --- First big curly bracket from eq. (4.33) ---
        #
        #  2 * s23 * (s12 + s13) / |denom_s23|^2
        #  - s13*s23 / Re[...something...]
        #
        bracket_1 = 2.0 * s23 * (s12 + s13) / abs_denom_s23_sq - (
            s13 * s23 / (re_s23_s13)
        )  # etc. fill in exactly...

        part1 = prefactor * bracket_1

        #
        # --- Next big chunk: the " + 4 e^2 [...] " piece ---
        #
        # We'll just illustrate a few terms and you can fill out the rest.
        #
        # 4 * e^2 * [
        #    2*(s12 + s13) * Re[F2^*(s23)*F3(s23) + F2^5*(s23)*F3^5(s23)]
        #  + (1/s23)*(...)*(|F2(s23)|^2 + |F2^5(s23)|^2)
        #  + ...
        # ]
        #

        # Real part of ( F2_s23.conjugate()*F3_s23 + F2_5_s23.conjugate()*F3_5_s23 )
        re_F2F3 = (F2_s23.conjugate() * F3_s23 + F2_5_s23.conjugate() * F3_5_s23).real

        termA = 2.0 * (s12 + s13) * re_F2F3
        termB = (1.0 / s23) * (
            (lepton_masses[l_i] ** 2 * (s12 + s13) - 2.0 * s12 * s13)
            * (abs(F2_s23) ** 2 + abs(F2_5_s23) ** 2)
        )
        # etc.  Build them all up and sum:
        bracket_2 = termA + termB  # + the rest of your terms carefully transcribed

        part2 = 4.0 * const.eQED**2 * bracket_2

        #
        # --- Next chunk: the " + 2 e s23 me / f_a^2 c_ee Re[...] " piece ---
        #
        # For example:
        #
        # 2 * e * s23 * m_e / f_a^2 * c_ee * Re[ something with (ke_21 + kE_21)/(...) etc. ]
        #
        # Suppose you name that big bracket bracket_3:
        #
        bracket_3_real = 0.0  # fill in carefully
        part3 = (
            2.0
            * const.eQED
            * s23
            * lepton_masses[l_j]
            / (self.f_a**2)
            * self.c_lepton[l_j, l_j]
            * bracket_3_real
        )

        #
        # --- Finally, sum them up (still not adding the (1 <-> 2) piece) ---
        #
        M_sq_no_sym = part1 + part2 + part3

        return M_sq_no_sym

    # def BR_diff_li_to_lj_a_gamma(self, s12, s23, l_i, l_j):
    #     """Differential decay rate for li -> lj + a + gamma
    #     Args:
    #         s12: invariant mass of the (lj + gamma) system
    #         s23: invariant mass of the (gamma + a) system
    #     """
    #     denom = s12 * (self.m_a**2 - s12 - s23) ** 2
    #     curlyF = (
    #         self.m_a**6
    #         - s23**2 * (s12 + s23)
    #         - self.m_a**4 * (2 * lepton_masses[l_i] ** 2 + s12 + s23)
    #         + 2 * lepton_masses[l_i] ** 2 * (s12 + s23) * (2 * s12 + s23)
    #         - 2 * lepton_masses[l_i] ** 4 * (4 * s12 + s23)
    #         + self.m_a**2
    #         * (2 * lepton_masses[l_i] ** 4 + 4 * lepton_masses[l_i] ** 2 * s12 + s23**2)
    #     )
    #     curlyF /= denom

    #     return (
    #         const.alphaQED
    #         / 4
    #         / np.pi**2
    #         / 32
    #         / lepton_masses[l_i]
    #         * curlyF
    #         * (self.c_lepton[l_i, l_j] ** 2 + self.c_lepton[l_j, l_i] ** 2)
    #         / self.f_a**2
    #     )

    def BR_diff_li_to_lj_a_gamma_dEa(self, l_i, l_j, Ea):
        """Differential decay rate for li -> lj + a + gamma
        Args:
            x: 2 E_j/m_i
            y: 2 E_gamma/m_i
            z: 2 E_a/m_i
        """

        eta = self.m_a**2 / lepton_masses[l_i] ** 2
        z = 2 * Ea / lepton_masses[l_i]
        Integrand = (
            -0.5 * ((-10 + 4 * eta + 3 * z) * Sqrt(-4 * eta + Power(z, 2)))
            - (5 + (-2 + eta) * eta + (-4 + z) * z)
            * Log(-2 + z - Sqrt(-4 * eta + Power(z, 2)))
            + (5 + (-2 + eta) * eta + (-4 + z) * z)
            * Log(-2 + z + Sqrt(-4 * eta + Power(z, 2)))
        ) / (-1 - eta + z)

        return (
            lepton_masses[l_i] ** 3
            * const.alphaQED
            / 32
            / np.pi**2
            * Integrand
            * (2 / lepton_masses[l_i])
            * (Ea < lepton_masses[l_i] * (1 - eta))
            * (self.m_a < Ea)
            * (self.c_lepton[l_i, l_j] ** 2 + self.c_lepton[l_j, l_i] ** 2)
        )

    def BR_diff_li_to_lj_a_gamma_normed(self, x, y, l_i, l_j, m_a):
        """Differential decay rate for li -> lj + a + gamma
        Args:
            x: 2 E_j/m_i
            y: 2 E_gamma/m_i
        """
        eta = m_a**2 / lepton_masses[l_i] ** 2
        denom = y**2 * (1 - x - y - eta)
        num = y * (1 - x**2 - eta**2) - 2 * (1 - eta) * (1 - x - eta)

        if denom != 0:
            return (
                lepton_masses[l_i] ** 3
                * const.alphaQED
                / 32
                / np.pi**2
                * num
                / denom
                * (self.c_lepton[l_i, l_j] ** 2 + self.c_lepton[l_j, l_i] ** 2)
            )
        else:
            return 0

    def BR_li_to_lj_a_gamma(
        self, l_i, l_j, E_gamma_min=0.0, E_e_min=const.m_e, theta_eg_min=0.0
    ):
        """Branching ratio for l_i -> l_j + a + gamma"""

        eta = self.m_a**2 / lepton_masses[l_i] ** 2

        # if isinstance(self.m_a, np.ndarray) and self.m_a.ndim == 2:
        # Gamma_a = 0
        # else:
        Gamma_a, _ = dblquad(
            lambda x, y: self.BR_diff_li_to_lj_a_gamma_normed(x, y, l_i, l_j, self.m_a)
            * (y > 2 * E_gamma_min / lepton_masses[l_i])
            * (x > 2 * E_e_min / lepton_masses[l_i])
            * ((1 + 2 * (1 - x - y - eta) / x / y) < np.cos(theta_eg_min)),
            a=2 * E_gamma_min / lepton_masses[l_i],
            b=1 - eta,
            gfun=lambda y: max((1 - y - eta), 2 * E_e_min / lepton_masses[l_i]),
            hfun=lambda y: (1 - y - eta) / (1 - y),
        )

        return Gamma_a / self.f_a**2 / lepton_gammas[l_i]

    # def BR_li_to_lj_a_gamma(self, l_i, l_j):
    #     """Branching ratio for mu -> e + a + gamma"""

    #     Gamma_a, _ = dblquad(
    #         lambda s23, s12: self.BR_diff_li_to_lj_a_gamma(s12, s23, l_i, l_j),
    #         a=s12_min(lepton_masses[l_i], lepton_masses[l_j], 0, self.m_a),
    #         b=s12_max(lepton_masses[l_i], lepton_masses[l_j], 0, self.m_a),
    #         gfun=lambda s12: s23_min(
    #             s12, lepton_masses[l_i], lepton_masses[l_j], 0, self.m_a
    #         ),
    #         hfun=lambda s12: s23_max(
    #             s12, lepton_masses[l_i], lepton_masses[l_j], 0, self.m_a
    #         ),
    #     )
    #     return Gamma_a / Gamma_mu

    def F2_5(self, q, l_i, l_j):
        """
        Form factor for the transition l_i -> l_j + a
        Args:
            q: momentum transfer
            l_i: index of the first lepton
            l_j: index of the second lepton
        """
        return (
            -lepton_masses[l_i]
            * const.eQED
            / self.f_a**2
            / 16
            / np.pi**2
            * (self.c_lepton[l_i, l_j] + self.c_lepton[l_j, l_i])
            * (
                0.25 * self.c_lepton[l_i, l_i] * g1(q, lepton_masses[l_i])
                + const.alphaQED
                / 4
                / np.pi
                * self.c_gg
                * g2(q, lepton_masses[l_i], self.f_a)
            )
        )

    def F2(self, q, l_i, l_j):
        """
        Form factor for the transition l_i -> l_j + a
        Args:
            q: momentum transfer
            l_i: index of the first lepton
            l_j: index of the second lepton
        """
        return (
            -lepton_masses[l_i]
            * const.eQED
            / self.f_a**2
            / 16
            / np.pi**2
            * (self.c_lepton[l_i, l_j] - self.c_lepton[l_j, l_i])
            * (
                0.25 * self.c_lepton[l_i, l_i] * g1(q, lepton_masses[l_i])
                + const.alphaQED
                / 4
                / np.pi
                * self.c_gg
                * g2(q, lepton_masses[l_i], self.f_a)
            )
        )

    def F3(self, q, l_i, l_j):
        """
        Form factor for the transition l_i -> l_j + a
        Args:
            q: momentum transfer
            l_i: index of the first lepton
            l_j: index of the second lepton
        """
        return (
            -lepton_masses[l_i]
            * const.eQED
            / self.f_a**2
            / 16
            / np.pi**2
            * (self.c_lepton[l_i, l_j] - self.c_lepton[l_j, l_i])
            * (
                0.25 * self.c_lepton[l_i, l_i] * l1(q, lepton_masses[l_i], self.m_a)
                + const.alphaQED
                / 4
                / np.pi
                * self.c_gg
                * l2(q, lepton_masses[l_i], self.f_a)
            )
        )

    def F3_5(self, q, l_i, l_j):
        """
        Form factor for the transition l_i -> l_j + a
        Args:
            q: momentum transfer
            l_i: index of the first lepton
            l_j: index of the second lepton
        """
        return (
            -lepton_masses[l_i]
            * const.eQED
            / self.f_a**2
            / 16
            / np.pi**2
            * (self.c_lepton[l_i, l_j] + self.c_lepton[l_j, l_i])
            * (
                0.25 * self.c_lepton[l_i, l_i] * l1(q, lepton_masses[l_i], self.m_a)
                + const.alphaQED
                / 4
                / np.pi
                * self.c_gg
                * l2(q, lepton_masses[l_i], self.f_a)
            )
        )

    """
    BEGIN Estimates of Onia to ALPs
    """

    def BR_Psi2S_to_li_lj_a(self, l_i, l_j):
        """Branching ratio for Psi2S -> li + lj + a"""
        Brfactor = (
            self.c_lepton[l_i, l_j] ** 2
            / 16
            / np.pi**2
            * (lepton_masses[l_i] + lepton_masses[l_j]) ** 2
            / self.f_a**2
        ) * (self.m_a < 3.686 - lepton_masses[l_i] - lepton_masses[l_j])
        if l_i == 2 or l_j == 2:
            return Brfactor * 3e-3
        else:
            return Brfactor * 8e-3

    """
    END Estimates of Onia to ALPs
    """


def s23_max(s12, M, m1, m2, m3):
    """max of s23 = (p2+p3)^2 for given M, m1, m2, m3"""
    E2_star = (s12 - m1**2 + m2**2) / (2 * np.sqrt(s12))
    E3_star = (M**2 - s12 - m3**2) / (2 * np.sqrt(s12))

    return (E2_star + E3_star) ** 2 - (
        np.sqrt(E2_star**2 - m2**2) - np.sqrt(E3_star**2 - m3**2)
    ) ** 2


def s23_min(s12, M, m1, m2, m3):
    """min of s23 = (p2+p3)^2 for given M, m1, m2, m3"""
    E2_star = (s12 - m1**2 + m2**2) / (2 * np.sqrt(s12))
    E3_star = (M**2 - s12 - m3**2) / (2 * np.sqrt(s12))

    return (E2_star + E3_star) ** 2 - (
        np.sqrt(E2_star**2 - m2**2) + np.sqrt(E3_star**2 - m3**2)
    ) ** 2


def s12_max(M, m1, m2, m3):
    """max of s12 = (P - p3)^2 for given M, m1, m2, m3"""
    return (M - m3) ** 2


def s12_min(M, m1, m2, m3):
    """min of s12 = (P - p3)^2 for given M, m1, m2, m3"""
    return (m1 + m2) ** 2


def Power(x, y):
    return x**y


def Log(x):
    return np.log(x)


def Sqrt(x):
    return np.sqrt(x)


def Iv_mvv_integral(Ea, ma, mlep, mtau=const.m_tau):
    """Integral for the visible decay width of ALP decays

    Integral of (I * v) over mvvSQR

    From MATHEMATICA:

        v[a_, b_, c_] := Sqrt[1 - 2 (b^2 + c^2)/a^2 + (b^2 - c^2)^2/a^4];

        mtaubarSQR = mtau^2 + ma^2 - 2*mtau*Ea;

        mvvSQRMax = (Sqrt[mtaubarSQR] - mlep)^2;

        Ifunc[mtbarSQR_, mvvSQR_] := (1 + (mvvSQR - 2*mlep^2)/mtaubar^2 + (mlep^4 + mlep^2 mvvSQR - 2  mvvSQR^2)/mtaubar)*v[Sqrt[mtaubarSQR], Sqrt[mvvSQR], mlep];

    """

    return (
        Power(ma, 2)
        - 8 * Power(mlep, 2)
        - 2 * Ea * mtau
        + Power(mtau, 2)
        - Power(mlep, 8) / Power(Power(ma, 2) - 2 * Ea * mtau + Power(mtau, 2), 3)
        + (8 * Power(mlep, 6)) / Power(Power(ma, 2) - 2 * Ea * mtau + Power(mtau, 2), 2)
    ) / 2.0 + (
        6
        * Power(mlep, 4)
        * Log((Power(ma, 2) - 2 * Ea * mtau + Power(mtau, 2)) / Power(mlep, 2))
    ) / (
        Power(ma, 2) + mtau * (-2 * Ea + mtau)
    )


#
# Now we make a little helper that adds the (1 <-> 2) piece.
# That means we add M_squared(...) + M_squared(...) with s12<->s13 swapped.
# In many references, "1 <-> 2" indicates that one must symmetrize w.r.t.
# permutations of the final-state momenta, so you effectively add the
# expression again but with s12 and s13 swapped (and if the form-factor
# arguments likewise need swapping).
#


def M_squared_sym(
    s12,
    s13,
    s23,
    m_e,
    m_mu,
    f_a,
    m_a,
    Gamma_a,
    c_ee,
    ke_12,
    kE_12,
    e,
    # F2 etc. at both s23 and s13
    F2_s23,
    F2_5_s23,
    F3_s23,
    F3_5_s23,
    F2_s13,
    F2_5_s13,
    F3_s13,
    F3_5_s13,
):
    """
    Returns the *fully symmetrized* |M|^2 = expression + (1 <-> 2).
    """
    # The original piece:
    val = M_squared(
        s12,
        s13,
        s23,
        m_e,
        m_mu,
        f_a,
        m_a,
        Gamma_a,
        c_ee,
        ke_12,
        kE_12,
        e,
        F2_s23,
        F2_5_s23,
        F3_s23,
        F3_5_s23,
        F2_s13,
        F2_5_s13,
        F3_s13,
        F3_5_s13,
    )
    # The swapped piece (swap s12 <-> s13, also swap the form-factor inputs that
    # used to go with s13 vs. s12):
    val_swapped = M_squared(
        s13,
        s12,
        s23,
        m_e,
        m_mu,
        f_a,
        m_a,
        Gamma_a,
        c_ee,
        ke_12,
        kE_12,
        e,
        # notice we swap which F_*(sXX) we pass in
        F2_s23=F2_s23,
        F2_5_s23=F2_5_s23,
        F3_s23=F3_s23,
        F3_5_s23=F3_5_s23,
        F2_s13=F2_s12,
        F2_5_s13=F2_5_s12,
        F3_s13=F3_s12,
        F3_5_s13=F3_5_s12,
    )
    # sum them
    return val + val_swapped
