import colorsys
import matplotlib.colors as mc
import numpy as np
from scipy.stats import chi2
from math import log10, floor, erf

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.tri as tri
from scipy.spatial.distance import pdist, squareform
from matplotlib import colors as mpl_colors
from matplotlib.collections import PatchCollection
from scipy.interpolate import splprep, splev

import scipy

from . import const
from alp.models import ALP


###########################
#
fsize = 11
fsize_annotate = 10

std_figsize = (1.2 * 3.7, 1.3 * 2.3617)
std_axes_form = [0.18, 0.16, 0.79, 0.76]

rcparams = {
    "axes.labelsize": fsize,
    "xtick.labelsize": fsize,
    "ytick.labelsize": fsize,
    "figure.figsize": std_figsize,
    "legend.frameon": False,
    "legend.loc": "best",
}
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
matplotlib.rcParams["hatch.linewidth"] = 0.3

CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CB_color_cycle)
rcParams.update(rcparams)

# settings for Mini Figs
TOTAL_RATE = False
INCLUDE_MB_LAST_BIN = False
STACKED = False
PLOT_FAMILY = False
PATH_PLOTS = "plots/event_rates/"

PAPER_TAG = r"HKZ\,2024"


##########################
#
def get_CL_from_sigma(sigma):
    return erf(sigma / np.sqrt(2))


def get_chi2vals_w_nsigmas(n_sigmas, ndof):
    return [chi2.ppf(get_CL_from_sigma(i), ndof) for i in range(n_sigmas + 1)]


def get_chi2vals_w_sigma(sigma, ndof):
    return chi2.ppf(get_CL_from_sigma(sigma), ndof)


def get_chi2vals_w_CL(CLs, ndof):
    return [chi2.ppf(cl, ndof) for cl in CLs]


def std_fig(ax_form=std_axes_form, figsize=std_figsize, rasterized=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)
    return fig, ax


def double_axes_fig(
    height=0.5,
    gap=0.1,
    axis_base=[0.14, 0.1, 0.80, 0.18],
    figsize=std_figsize,
    split_y=False,
    split_x=False,
    rasterized=False,
):
    fig = plt.figure(figsize=figsize)

    if split_y and not split_x:
        axis_base = [0.14, 0.1, 0.80, 0.4 - gap / 2]
        axis_appended = [0.14, 0.5 + gap / 2, 0.80, 0.4 - gap / 2]

    elif not split_y and split_x:
        axis_appended = [0.14, 0.1, 0.4 - gap / 2, 0.8]
        axis_base = [0.14 + 0.4 + gap / 2, 0.1, 0.4 - gap / 2, 0.8]

    else:
        axis_base[-1] = height
        axis_appended = axis_base + np.array(
            [0, height + gap, 0, 1 - 2 * height - gap - axis_base[1] - 0.07]
        )

    ax1 = fig.add_axes(axis_appended, rasterized=rasterized)
    ax2 = fig.add_axes(axis_base, rasterized=rasterized)
    ax1.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)

    return fig, [ax1, ax2]


def data_plot(ax, X, Y, xerr, yerr, zorder=2, label="data", **kwargs):
    return ax.errorbar(
        X,
        Y,
        yerr=yerr,
        xerr=xerr,
        marker="o",
        markeredgewidth=0.75,
        capsize=1,
        markerfacecolor="black",
        markeredgecolor="black",
        ms=1.75,
        lw=0.0,
        elinewidth=0.75,
        color="black",
        label=label,
        zorder=zorder,
        **kwargs,
    )


def step_plot(
    ax, x, y, lw=1, color="red", label="signal", where="post", dashes=(3, 0), zorder=3
):
    return ax.step(
        np.append(x, np.max(x) + x[-1]),
        np.append(y, 0.0),
        where=where,
        lw=lw,
        dashes=dashes,
        color=color,
        label=label,
        zorder=zorder,
    )


# Function to find the path that connects points in order of closest proximity
def nearest_neighbor_path(points):
    # Compute the pairwise distance between points
    dist_matrix = squareform(pdist(points))

    # Set diagonal to a large number to avoid self-loop
    np.fill_diagonal(dist_matrix, np.inf)

    # Start from the first point
    current_point = 0
    path = [current_point]

    # Find the nearest neighbor of each point
    while len(path) < len(points):
        # Find the nearest point that is not already in the path
        nearest = np.argmin(dist_matrix[current_point])
        # Add the nearest point to the path
        path.append(nearest)
        # Update the current point
        current_point = nearest
        # Mark the visited point so it's not revisited
        dist_matrix[:, current_point] = np.inf

    # Return the ordered path indices and the corresponding points
    ordered_points = points[path]
    return ordered_points


def plot_closed_region(points, logx=False, logy=False):
    x, y = points
    if logy:
        if (y == 0).any():
            raise ValueError("y values cannot contain any zeros in log mode.")
        sy = np.sign(y)
        ssy = (np.abs(y) < 1) * (-1) + (np.abs(y) > 1) * (1)
        y = ssy * np.log(y * sy)
    if logx:
        if (x == 0).any():
            raise ValueError("x values cannot contain any zeros in log mode.")
        sx = np.sign(x)
        ssx = (x < 1) * (-1) + (x > 1) * (1)
        x = ssx * np.log(x * sx)

    points = np.array([x, y]).T

    points_s = points - points.mean(0)
    angles = np.angle((points_s[:, 0] + 1j * points_s[:, 1]))
    points_sort = points_s[angles.argsort()]
    points_sort += points.mean(0)

    tck, u = splprep(points_sort.T, u=None, s=0.0, per=0, k=1)
    u_new = np.linspace(u.min(), u.max(), len(points[:, 0]))
    x_new, y_new = splev(u_new, tck, der=0)

    if logx:
        x_new = sx * np.exp(ssx * x_new)
    if logy:
        y_new = sy * np.exp(ssy * y_new)

    return x_new, y_new


def get_ordered_closed_region(points, logx=False, logy=False):
    x, y = points
    # check for nans
    if np.isnan(points).sum() > 0:
        raise ValueError("NaN's were found in input data. Cannot order the contour.")

    # check for repeated x-entries --
    # this is an error because
    x, mask_diff = np.unique(x, return_index=True)
    y = y[mask_diff]

    if logy:
        if (y == 0).any():
            raise ValueError("y values cannot contain any zeros in log mode.")
        sy = 1  # np.sign(y)
        ssy = 1  # (np.abs(y) < 1) * (-1) + (np.abs(y) > 1) * (1)
        y = ssy * np.log10(y * sy)
    if logx:
        if (x == 0).any():
            raise ValueError("x values cannot contain any zeros in log mode.")
        sx = 1  # np.sign(x)
        ssx = 1  # (x < 1) * (-1) + (x > 1) * (1)
        x = ssx * np.log10(x * sx)

    xmin, ymin = np.min(x), np.min(y)
    x, y = x - xmin, y - ymin

    points = np.array([x, y]).T
    # points_s     = (points - points.mean(0))
    # angles       = np.angle((points_s[:,0] + 1j*points_s[:,1]))
    # points_sort  = points_s[angles.argsort()]
    # points_sort += points.mean(0)

    # if np.isnan(points_sort).sum()>0:
    #     raise ValueError("NaN's were found in sorted points. Cannot order the contour.")
    # # print(points.mean(0))
    # # return points_sort
    # tck, u = splprep(points_sort.T, u=None, s=0.0, per=0, k=1)
    # # u_new = np.linspace(u.min(), u.max(), len(points[:,0]))
    # x_new, y_new = splev(u, tck, der=0)
    # # x_new, y_new = splev(u_new, tck, der=0)
    dist_matrix = squareform(pdist(points))

    # Set diagonal to a large number to avoid self-loop
    np.fill_diagonal(dist_matrix, np.inf)

    # Start from the first point
    current_point = 0
    path = [current_point]

    # Find the nearest neighbor of each point
    while len(path) < len(points):
        # Find the nearest point that is not already in the path
        nearest = np.argmin(dist_matrix[current_point])
        # Add the nearest point to the path
        path.append(nearest)
        # Update the current point
        current_point = nearest
        # Mark the visited point so it's not revisited
        dist_matrix[:, current_point] = np.inf

    # Return the ordered path indices and the corresponding points
    x_new, y_new = points[path].T
    x_new, y_new = x_new + xmin, y_new + ymin

    if logx:
        x_new = sx * 10 ** (ssx * x_new)
    if logy:
        y_new = sy * 10 ** (ssy * y_new)

    return x_new, y_new


def interp_grid(
    x,
    y,
    z,
    fine_gridx=False,
    fine_gridy=False,
    logx=False,
    logy=False,
    method="interpolate",
    smear_stddev=False,
):
    # default
    if not fine_gridx:
        fine_gridx = 100
    if not fine_gridy:
        fine_gridy = 100

    # log scale x
    if logx:
        xi = np.geomspace(np.min(x), np.max(x), fine_gridx)
    else:
        xi = np.linspace(np.min(x), np.max(x), fine_gridx)

    # log scale y
    if logy:
        yi = np.geomspace(np.min(y), np.max(y), fine_gridy)
    else:
        yi = np.linspace(np.min(y), np.max(y), fine_gridy)

    Xi, Yi = np.meshgrid(xi, yi)
    # if logy:
    #     Yi = 10**(-Yi)

    # triangulation
    if method == "triangulation":
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Zi = interpolator(Xi, Yi)

    elif method == "interpolate":
        Zi = scipy.interpolate.griddata(
            (x, y), z, (xi[None, :], yi[:, None]), method="linear", rescale=True
        )
    else:
        print(f"Method {method} not implemented.")

    # gaussian smear -- not recommended
    if smear_stddev:
        Zi = scipy.ndimage.filters.gaussian_filter(
            Zi, smear_stddev, mode="nearest", order=0, cval=0
        )

    return Xi, Yi, Zi


def round_sig(x, sig):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def sci_notation(
    num,
    sig_digits=1,
    precision=None,
    exponent=None,
    notex=False,
    optional_sci=False,
):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num != 0:
        if exponent is None:
            exponent = int(np.floor(np.log10(abs(num))))
        coeff = round(num / float(10**exponent), sig_digits)
        if coeff == 10:
            coeff = 1
            exponent += 1
        if precision is None:
            precision = sig_digits

        if optional_sci and np.abs(exponent) < optional_sci:
            string = rf"{round_sig(num, precision)}"
        else:
            string = r"{0:.{2}f}\times 10^{{{1:d}}}".format(coeff, exponent, precision)

        if notex:
            return string
        else:
            return f"${string}$"

    else:
        return r"0"


# https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


###########################
def get_cmap_colors(name, ncolors, cmin=0, cmax=1, reverse=False):
    try:
        cmap = plt.get_cmap(name)
    except ValueError:
        cmap = build_cmap(name, reverse=reverse)
    return cmap(np.linspace(cmin, cmax, ncolors, endpoint=True))


def build_cmap(color, reverse=False):
    cvals = [0, 1]
    colors = [color, "white"]
    if reverse:
        colors = colors[::-1]

    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    return mpl_colors.LinearSegmentedColormap.from_list("", tuples)


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(
                plt.Rectangle(
                    [
                        width / len(orig_handle.colors) * i - handlebox.xdescent,
                        -handlebox.ydescent,
                    ],
                    width / len(orig_handle.colors),
                    height,
                    facecolor=c,
                    edgecolor="none",
                )
            )

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


def plot_other_limits_Bvis(
    ax, ma_fixed, Bvis_range=[1e-12, 1], c_lepton=None, c_NN=0, mN=0, linewidth=0.25
):

    # BaBar Limit
    ma, qsi = np.genfromtxt("digitized/BABAR_leptophilic.dat", unpack=True)
    ax.fill_between(
        ma,
        (1 + (ma < const.m_tau - const.m_mu) * 1e100) * qsi / const.vev_EW,
        qsi / qsi,
        color="silver",
        linestyle="-",
    )
    ax.plot(
        ma,
        (1 + (ma < const.m_tau - const.m_mu) * 1e100) * qsi / const.vev_EW,
        color="black",
        linestyle="-",
        lw=linewidth,
    )
    # BELLE Limit
    ma, qsi = np.genfromtxt("digitized/Belle_leptophilic.dat", unpack=True)
    ax.fill_between(
        ma,
        (1 + (ma < const.m_tau - const.m_mu) * 1e100) * qsi / const.vev_EW,
        qsi / qsi,
        color="silver",
        linestyle="-",
    )
    ax.plot(
        ma,
        (1 + (ma < const.m_tau - const.m_mu) * 1e100) * qsi / const.vev_EW,
        color="black",
        linestyle="-",
        lw=linewidth,
    )

    Bvis = np.geomspace(Bvis_range[0], Bvis_range[1], 1000)
    inv_fa = np.geomspace(1e-9, 1e-2, 1000, endpoint=True)
    BVIS, INV_FA = np.meshgrid(Bvis, inv_fa)

    alps = ALP(ma_fixed, 1 / INV_FA, c_lepton=c_lepton, c_NN=c_NN, mN=mN, Bvis=BVIS)
    ################################################################
    # Belle-II (Tau -> mu alp)
    ma_limit, B_limit_90CL = np.genfromtxt(
        "digitized/BelleII_tau_to_mu_a.dat", unpack=True
    )
    B_limit_90CL_interp = np.interp(
        ma_fixed, ma_limit, B_limit_90CL, left=1e100, right=1e100
    )

    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 300, 100e100)  # 3 m travel

    BR_tau_mu_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_mu()
    c = ax.contourf(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=["silver"],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=["black"],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    # x_vis = []
    # y_vis = []
    # x_vis.append([c.collections[0].get_paths()[0].vertices[:,0]])
    # y_vis.append([c.collections[0].get_paths()[0].vertices[:,1]])

    ################################################################
    # Belle-II (Tau -> e alp)
    ma_limit, B_limit_90CL = np.genfromtxt(
        "digitized/BelleII_tau_to_e_a.dat", unpack=True
    )
    B_limit_90CL_interp = np.interp(
        ma_fixed, ma_limit, B_limit_90CL, left=1e100, right=1e100
    )

    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 300, 100e100)  # 3 m travel

    BR_tau_e_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_e()
    c = ax.contourf(
        BVIS,
        INV_FA,
        BR_tau_e_a,
        levels=[1, 1e100],
        colors=["silver"],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        BVIS,
        INV_FA,
        BR_tau_e_a,
        levels=[1],
        colors=["black"],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    BR_tau_ell_a = alps.BR_tau_to_a_e() + alps.BR_tau_to_a_mu()
    c = ax.contourf(
        BVIS,
        INV_FA,
        BR_tau_ell_a,
        levels=[1e-2, 1e100],
        colors=["silver"],
        alpha=1,
        zorder=0.2,
    )
    c = ax.contour(
        BVIS,
        INV_FA,
        BR_tau_ell_a,
        levels=[1e-2],
        colors=["black"],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> e- (alp -> mu+ mu-)) + (Tau- -> mu- (alp -> mu+ e-))
    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 2.7e-8
    BR_tau_mu_a = (
        1
        / B_limit_90CL
        * P_decay
        * (
            alps.BR_tau_to_a_e() * alps.BR_a_to_mm
            + alps.BR_tau_to_a_mu() * alps.BR_a_to_me
        )
    )
    c = ax.contourf(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=["grey"],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=["black"],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> mu- (alp -> mu+ mu-))
    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 2.1e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_mm
    c = ax.contourf(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=["grey"],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=["black"],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> mu- (alp -> mu+ e-))
    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 2.7e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_me
    c = ax.contourf(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=["grey"],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=["black"],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    # Belle (Tau- -> mu- (alp -> mu- e+))
    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 1.7e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_em
    c = ax.contourf(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=["grey"],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=["black"],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> e- (alp -> mu+ e-))
    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 1.5e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_me
    c = ax.contourf(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=["grey"],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        BVIS,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=["black"],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )


def plot_other_limits(
    ax,
    c_lepton=None,
    c_NN=0,
    mN=0,
    linewidth=0.25,
    vis_color="grey",
    inv_color="lightgrey",
    annotate=False,
    edgecolor="dimgrey",
):
    # Muon Limits
    x, y = np.genfromtxt("digitized/Jodidio_et_al_A.dat", unpack=True)
    y *= 2
    x *= 1e-9
    ax.fill_between(
        x,
        (1 / y) / c_lepton[0, 1],
        y / y,
        facecolor=inv_color,
        linestyle="-",
        zorder=0.1,
        edgecolor="None",
    )
    ax.plot(
        x,
        (1 / y) / c_lepton[0, 1],
        color="dimgrey",
        linestyle="-",
        lw=0.2,
        zorder=0.2,
    )

    # x, y = np.genfromtxt("digitized/TWIST_mue_A.dat", unpack=True)
    # y *= 2
    # x *= 1e-9
    # y = y[np.argsort(x)]
    # x = x[np.argsort(x)]
    # ax.fill_between(
    #     x,
    #     (1 / y) / c_lepton[0, 1],
    #     y / y,
    #     facecolor=inv_color,
    #     edgecolor="None",
    #     linestyle="-",
    #     zorder=0.1,
    # )
    # ax.plot(
    #     x,
    #     (1 / y) / c_lepton[0, 1],
    #     color=edgecolor,
    #     linestyle="-",
    #     lw=0.2,
    #     zorder=0.2,
    # )

    # SN1987A
    x, y = np.genfromtxt("digitized/Supernova_mumu.dat", unpack=True)
    x *= 1e-9
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]
    ax.fill_between(
        x,
        (1 / y) / c_lepton[1, 1],
        y / y,
        facecolor=lighten_color(inv_color, 0.5),
        edgecolor="None",
        linestyle="-",
        zorder=0.1,
    )
    ax.plot(
        x,
        (1 / y) / c_lepton[1, 1],
        color=lighten_color(edgecolor, 0.5),
        linestyle=(1, (6, 2)),
        lw=0.2,
        zorder=0.2,
    )

    # SN1987A ee
    x, y = np.genfromtxt("digitized/Supernova_ee.dat", unpack=True)
    x *= 1e-9
    x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
    ax.fill(
        x,
        (1 / y) / c_lepton[0, 0],
        facecolor=lighten_color(inv_color, 0.5),
        edgecolor="None",
        linestyle="-",
        zorder=0.1,
    )
    ax.plot(
        x,
        (1 / y) / c_lepton[0, 0],
        color=lighten_color(edgecolor, 0.5),
        linestyle=(1, (6, 2)),
        lw=0.2,
        zorder=0.2,
    )

    # SN1987A emu
    for v in ["v1", "v2"]:
        x, y = np.genfromtxt(f"digitized/Supernova_emu_{v}.dat", unpack=True)
        x *= 1e-3
        y *= 2 / (const.m_e + const.m_mu)
        x, y = get_ordered_closed_region([x, y], logx=True, logy=True)
        ax.fill(
            x,
            y / c_lepton[0, 1],
            facecolor=lighten_color(inv_color, 0.5),
            edgecolor="None",
            linestyle="-",
            zorder=0.1,
            alpha=0.75,
        )
        ax.plot(
            x,
            y / c_lepton[0, 1],
            color=lighten_color(edgecolor, 0.5),
            linestyle=(1, (6, 2)),
            lw=0.5,
            zorder=0.2,
        )

    # PIENU bounds
    for b in [
        "digitized/PIENU_mu_to_e_X.dat",
        "digitized/Derenzo_et_al_mu_to_e_X.dat",
        "digitized/Bilger_et_al_mu_to_e_X.dat",
    ]:
        mX, BR = np.genfromtxt(b, unpack=True)
        mX *= 1e-3
        alp = ALP(mX, 1.0, c_lepton=c_lepton, c_NN=c_NN, mN=mN)
        inv_fa = np.sqrt(BR / alp.BR_li_to_lj_a(1, 0))
        ax.fill_between(
            mX,
            inv_fa,
            inv_fa / inv_fa,
            edgecolor="None",
            facecolor=inv_color,
            linestyle="-",
            zorder=0.02,
        )
        ax.plot(
            mX,
            inv_fa,
            color="dimgrey",
            linestyle="-",
            lw=0.25,
            zorder=1,
        )

    pe, BR = np.genfromtxt("digitized/TWIST_mu_to_e_X.dat", unpack=True)
    pe *= 1e-3
    mX = np.sqrt(
        const.m_e**2 + const.m_mu**2 - 2 * const.m_mu * np.sqrt(pe**2 + const.m_e**2)
    )
    BR = np.append(BR, BR[np.argmin(mX)])
    mX = np.append(mX, 0)

    alp = ALP(mX, 1.0, c_lepton=c_lepton, c_NN=c_NN, mN=mN)
    inv_fa = np.sqrt(BR / alp.BR_li_to_lj_a(1, 0))
    ax.fill_between(
        mX,
        inv_fa,
        inv_fa / inv_fa,
        edgecolor="None",
        facecolor=inv_color,
        linestyle="-",
        zorder=0.01,
    )
    ax.plot(
        mX,
        inv_fa,
        color="dimgrey",
        linestyle="-",
        lw=0.25,
        zorder=0.2,
    )

    # ###########################################################
    # ma, y = np.genfromtxt("digitized/E137_Araki_et_al_LFV.dat", unpack=True)
    # inv_fa = y / const.m_e
    # x, y = get_ordered_closed_region([ma, inv_fa], logx=True, logy=True)
    # ax.plot(
    #     x,
    #     y,
    #     color="black",
    #     linestyle="-",
    #     zorder=3,
    # )

    ###########################################################
    ma = np.geomspace(1e-3, 3, 1000)
    inv_fa = np.geomspace(1e-9, 1e-2, 1000, endpoint=True)
    MA, INV_FA = np.meshgrid(ma, inv_fa)

    alps = ALP(MA, 1 / INV_FA, c_lepton=c_lepton, c_NN=c_NN, mN=mN, Bvis=1)
    ################################################################
    # Belle-II (Tau -> mu alp)
    ma_limit, B_limit_90CL = np.genfromtxt(
        "digitized/BelleII_tau_to_mu_a.dat", unpack=True
    )
    B_limit_90CL_interp = np.interp(MA, ma_limit, B_limit_90CL, left=1e100, right=1e100)

    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 300, 100e100)  # 3 m travel

    BR_tau_mu_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_mu()
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[inv_color],
        alpha=1,
        zorder=0.2,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    BR_tau_ell_a = alps.BR_tau_to_a_e() + alps.BR_tau_to_a_mu()
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_ell_a,
        levels=[1e-2, 1e100],
        colors=[inv_color],
        alpha=1,
        zorder=0.5,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_ell_a,
        levels=[1e-2],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle-II (Tau -> e alp)
    ma_limit, B_limit_90CL = np.genfromtxt(
        "digitized/BelleII_tau_to_e_a.dat", unpack=True
    )
    B_limit_90CL_interp = np.interp(MA, ma_limit, B_limit_90CL, left=1e100, right=1e100)

    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 300, 100e100)  # 3 m travel

    BR_tau_e_a = 1 / B_limit_90CL_interp * P_decay * alps.BR_tau_to_a_e()
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_e_a,
        levels=[1, 1e100],
        colors=[inv_color],
        alpha=1,
        zorder=0.2,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_e_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> e- (alp -> mu+ mu-)) + (Tau- -> mu- (alp -> mu+ e-))
    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 2.7e-8
    BR_tau_mu_a = (
        1
        / B_limit_90CL
        * P_decay
        * (
            alps.BR_tau_to_a_e() * alps.BR_a_to_mm
            + alps.BR_tau_to_a_mu() * alps.BR_a_to_me
        )
    )
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[vis_color],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> mu- (alp -> mu+ mu-))
    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 2.1e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_mm
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[vis_color],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> mu- (alp -> mu+ e-))
    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 2.7e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_me
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[vis_color],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    # Belle (Tau- -> mu- (alp -> mu- e+))
    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 1.7e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_em
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[vis_color],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )

    ################################################################
    # Belle (Tau- -> e- (alp -> mu+ e-))
    p_alp_avg = 10.58 / 4
    P_decay = alps.prob_decay(p_alp_avg, 0, 1)  # decays within 1 cm
    B_limit_90CL = 1.5e-8
    BR_tau_mu_a = 1 / B_limit_90CL * P_decay * alps.BR_tau_to_a_mu() * alps.BR_a_to_me
    c = ax.contourf(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1, 1e100],
        colors=[vis_color],
        alpha=1,
        zorder=0,
    )
    c = ax.contour(
        MA,
        INV_FA,
        BR_tau_mu_a,
        levels=[1],
        colors=[edgecolor],
        linestyles="-",
        linewidths=[linewidth],
        alpha=1,
        zorder=2,
    )
    if annotate:
        ax.annotate(
            r"Supernova $g_{\mu \mu}$",
            xy=(1.05e-2, 2.4e-8 / c_lepton[1, 1]),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        ax.annotate(
            r"Supernova $g_{ee}$",
            xy=(1.05e-2, 8e-8 / c_lepton[0, 0]),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        ax.annotate(
            r"Supernova $g_{e\mu}$",
            xy=(1.05e-2, 0.6e-8 / c_lepton[0, 1]),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        if 8e-10 / c_lepton[0, 1] < 5e-4:
            ax.annotate(
                r"$\mu\to e a_{\rm inv}$",
                xy=(1.05e-2, 1.0e-9 / c_lepton[0, 1]),
                xycoords="data",
                fontsize=9,
                horizontalalignment="left",
                verticalalignment="center",
                rotation=0,
                zorder=3,
                color="dimgrey",
            )
        ax.annotate(
            r"$\tau \to \ell + a_{\rm inv}$",
            xy=(1.05e-2, 3e-7 / c_lepton[0, 2]),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        ax.annotate(
            r"$\tau$ lifetime",
            xy=(1.05e-2, 9e-7 / c_lepton[0, 2]),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color=edgecolor,
        )
        ax.annotate(
            r"$\tau \to \ell + a_{\rm vis}$",
            xy=(0.55, 2.5e-7 / np.sqrt(c_lepton[1, 2])),
            xycoords="data",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
            zorder=3,
            color="black",
        )


def plot_hist_with_errors(
    ax, data, weights, bins, label, color, zorder=2, lw=1.5, nevents=1, ls="-"
):
    norm = np.max(weights)
    weights = weights / norm
    # Compute histogram and sum of squared weights per bin
    counts, bin_edges = np.histogram(data, bins=bins, weights=weights * nevents)
    sumw2, _ = np.histogram(data, bins=bins, weights=np.square(weights * nevents))

    # Bin centers for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_widths = np.diff(bin_edges)

    counts /= nevents
    sumw2 /= nevents**2
    # Plot main histogram
    ax.hist(
        data,
        bins=bins,
        weights=weights * norm,
        label=label,
        histtype="step",
        edgecolor=color,
        linestyle=ls,
        density=False,
        zorder=zorder,
        lw=lw,
    )

    # Plot error bars as bars around each bin
    ax.bar(
        bin_centers,
        2 * np.sqrt(sumw2) * norm,
        bottom=(counts - np.sqrt(sumw2)) * norm,
        width=bin_widths,
        edgecolor="None",
        facecolor=color,
        alpha=0.5,
        lw=0,
        zorder=zorder - 0.1,
    )
