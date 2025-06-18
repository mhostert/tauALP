import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad

from DarkNews import Cfourvec as Cfv


def decay_2_body(p4_parent, m_P, m_1, m_2):
    """
    Generate daughter 4-momenta for P --> 1 + 2 decay for an arbitrary parent 4-momenta.
    """
    nevents = p4_parent.shape[0]

    # Flat dOmega = dcos(theta) * dphi
    phi = np.random.uniform(0, 2 * np.pi, nevents)
    ctheta = np.random.uniform(-1, 1, nevents)

    # Sampling ALP energy from different production channels
    ECM = (m_P**2 + m_1**2 - m_2**2) / 2 / m_P * np.ones(nevents)
    pCM = np.zeros_like(ECM)
    pCM[ECM > m_1] = np.sqrt(ECM[ECM > m_1] ** 2 - m_1**2)

    # Build ALP 4 momenta
    p4_1_CM = Cfv.build_fourvec(ECM, pCM, ctheta, phi)
    p_parent = Cfv.get_3vec_norm(p4_parent)
    ctheta_LAB = Cfv.get_cosTheta(p4_parent)
    phi_LAB = np.arctan2(p4_parent[:, 2], p4_parent[:, 1])
    beta = -p_parent / p4_parent[:, 0]
    beta[beta < -1] = -1
    p4_1 = Cfv.Tinv(
        p4_1_CM,
        beta,
        ctheta_LAB,
        phi_LAB,
    )

    return p4_1


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
    z = np.sqrt(1 / tau) * (1 + 0j)
    f = np.piecewise(
        z,
        [z < 1, z >= 1],
        [
            lambda z: np.arcsin(z),
            lambda z: np.pi / 2
            + 1j / 2 * np.log((z + np.sqrt(z**2 - 1)) / (z - np.sqrt(z**2 - 1))),
        ],
    )

    return 1 - tau * f**2
