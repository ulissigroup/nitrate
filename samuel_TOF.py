# Code base taken from https://github.com/goldsmith-lab/ptru-alloy-no3rr-activity.
# All credits to ﻿Wang, Z., Young, S. D., Goldsmith, B. R., & Singh, N. (2021).
# Supporting Information: Increasing electrocatalytic nitrate reduction activity
# by controlling adsorption through PtRu alloying. Journal of Catalysis, 395, 143–154.
# https://doi.org/10.1016/j.jcat.2020.12.031


import os
import glob
import pickle
import re
import copy
import datetime
from sklearn.metrics import mean_absolute_error as mae
from scipy.optimize import curve_fit
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core.structure import *
import pandas as pd
from pymatgen.core.periodic_table import Element
from matplotlib import colors
from matplotlib import cm

from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter

import tqdm
import json
import multiprocessing


def parse_mkmcxx_results(results_dir: str = ".", U: float = 0.2, n_jobs: int = 16, outfile_path: str = None):
    """
    Load the results from the specified MKMCXX results folder.

    Read in the MKMCXX results from a results folder created by Jin-Xun Liu's
    code. This will create a pickle file containing a list of objects, each
    containing the `mkmcxx.MicrokineticsSimulation.results()` output for that
    folder.

    Parameters
    ----------
    results_dir : str, optional
        The folder where the results should be located, by default ".". This
        folder should contain subfolders of the form "O_<EO>__N<EN>", where
        -1*<EO> and -1*<EN> are the binding energies of O and N in kj/mol. Each
        subfolder should contain an "output.log" file and a "run" folder
        containing the MKCMXX simulation results.
    U : float, optional
        The applied potential in V vs. RHE, by default 0.2 V.
    n_jobs : int, optional
        The number of multiprocessing jobs to use when parsing the results, by
        default 16.
    outfile_path : str, optional
        The path in which to save the output pickle file, by default
        "jxl-<potential>V-results_<date>.pckl"
    """

    if not outfile_path:
        outfile_path = "jxl-{U}V-results_{datetime.date.today().strftime('%d-%b-%Y').lower()}.pckl"

    # Verify that results folder exists
    if not os.path.exists(results_dir):
        raise RuntimeError("Folder {results_dir} does not exist.")

    # Get list of all directories where MKMCXX was run
    print("Searching directories...")
    data_dirs = glob.glob(os.path.join(results_dir, "*", "output.log"))
    print("Found {len(data_dirs)} data directories.")

    # Make simulation object for each folder
    completed_simulations = [
        make_mkmcxx_simulation(folder) for folder in tqdm.tqdm(data_dirs)
        ]

    # Read the results. Parallelize over many operations
    with multiprocessing.Pool(processes=16) as p:
        parsed_results = list(
            tqdm.tqdm(
                p.imap(read_results, completed_simulations),
                total=len(completed_simulations),
            )
        )

    # # Pickle to a results file
    #     print("Saving results to pickle file {outfile_path}...")
    #     save_to_file(
    #         parsed_results, outfile_path,
    #     )
    #     print("Saving done.")

    # Find directories where there were no results available
    dud_directories = [
        sim["directory"]
        for sim in tqdm.tqdm(
            filter(lambda sim: not sim["results"]["range_results"], parsed_results)
        )
        ]

    print("{len(dud_directories)} directories unexpectedly had no simulation results.")
    print(dud_directories)

    return True


def make_mkmcxx_simulation(folder_path: str):
    """
    Quick wrapper to make a `MicrokineticsSimulation` object
    representing a completed simulation directory.
    """

    # Define some constants
    ev_to_kjmol = 96.485  # kJ/mol per eV
    binding_energy_regex = re.compile(r".*O_(?P<EO>\d+)__N_(?P<EN>\d+).*")

    # Load dummy data to enable instantiation of dummy MKMCXX runs
    with open(os.path.expanduser("base-reaction-set.pckl"), "rb") as f:
        fake_reactions = pickle.load(f)

    # Set up runs
    fake_runs = [{"temp": 300, "time": 1e8, "abstol": 1e-8, "reltol": 1e-8}]

    # Enter some settings from the input.mkm file
    fake_settings = {
        "type": "sequencerun",
        "usetimestamp": False,
        "drc": 0,
        "reagents": ["NO3-", "Haqu"],
        "keycomponents": ["N2", "NO3-", "N2O"],
        "makeplots": 0,
    }

    # Get O and N binding energies
    result = {"U": U}
    try:
        match = binding_energy_regex.match(folder_path)
        result.update(
            {
                label: -float(energy) / ev_to_kjmol
                for label, energy in match.groupdict().items()
                }
        )
    except AttributeError:
        print("Invalid folder name. Must be of format .*O_<EO>__N_<EN>.*")
        raise RuntimeError

    # Instantiate simulation object and add to dictionary
    folder_name = os.path.dirname(folder_path)
    result.update({"directory": folder_name})
    sim = mkmcxx.MicrokineticSimulation(
        reactions=list(fake_reactions.values()),
        settings=fake_settings,
        runs=fake_runs,
        directory=folder_name,
        # run_directory=".",
        run_directory="run",
    )
    result.update({"sim": sim})

    return result


def get_rate_points(results: list, compound: str = "NO3-"):
    """
    Read rate information from the entire dataset, handling cases where the
    data does not exist

    Parameters
    ----------
    results : list
        The output of the `read_results` functions defined above.
    compound : str
        The key used to look up rate information in the
        `mkmcxx.MicrokineticsSimulation.get_results()
        ["range_results"]["derivatives"]` object

    Returns
    -------
    list
        A list of the form [{"EO": <EO>, "EN": <EN>, "ratelog": <ratelog>}, ...],
        where <EO> is the O binding energy in eV, <EN> is the N binding energy
        in eV, and <ratelog> is the base-10 logarithm of the formation/consumption
        rate of the compound specified in `compound_str`.
    """

    extracted_rate_points = []

    for result in tqdm.tqdm(results):
        try:
            # Try rounding rates to 6 sig figs.
            raw_rate = result["results"]["range_results"]["derivatives"][compound][0]
            rounded_rate = np.float("{raw_rate:.6e}")
            projection = {
                "EO": result["EO"],
                "EN": result["EN"],
                "ratelog": np.log10(-(rounded_rate)),
            }
        except (KeyError, IndexError):
            projection = None

        extracted_rate_points.append(projection)

    # Filter out any None values
    extracted_rate_points = list(filter(lambda x: x, extracted_rate_points))

    return extracted_rate_points


def get_drc_points(results: list, rxn: str):
    """
    Read rate information from the entire dataset, handling cases where the
    data does not exist

    Parameters
    ----------
    results : list
        The output of the `read_results` functions defined above.
    rxn : str
        The key used to look up rate information in the
        `mkmcxx.MicrokineticsSimulation.get_results()["drc_results"]["drc"]`
        object

    Returns
    -------
    list
        A list of the form [{"EO": <EO>, "EN": <EN>, "drc": <drc>}, ...], where
        <EO> is the O binding energy in eV, <EN> is the N binding energy in eV,
        and <drc> is the Campbell degree-of-rate-control coefficient for the
        reaction specified in `rxn`.
    """

    extracted_drc_points = []

    for result in tqdm.tqdm(results):
        try:
            projection = {
                "EO": result["EO"],
                "EN": result["EN"],
                "drc": np.float(result["results"]["drc_results"]["drc"].loc[rxn, 0]),
            }
        except (KeyError, IndexError):
            projection = None

        extracted_drc_points.append(projection)

    # Filter out any None values
    extracted_drc_points = list(filter(lambda x: x, extracted_drc_points))

    return extracted_drc_points


def make_drc_plot(data: list,rxn_str: str, ax: plt.Axes = None, outfile_name: str = None,
                  clip_threshold: float = 2.0, contour_levels: np.ndarray = np.r_[-2:2.1:0.1]):
    """Make a degree of rate control plot for the specified reaction.

    Parameters
    ----------
    data :
        List containing data of the form [{"EO": <EO>, "EN": <EN>, "drc":
        <drc>}, ...], where <EO> is the O binding energy in eV, <EN> is the N
        binding energy in eV, and <drc> is the Campbell degree of rate control
        coefficient.
    rxn_str : str
        The reaction to be written above the plot, in LaTeX format.
    ax : plt.Axes
        The Axes object on which to place this plot. If none, Figure and Axes
        objects will be created for you.
    outfile_name : str, optional
        Name of graphics file (including extension) to which to export the
        plotted graph, by default None. If None, no plot will be written to
        disk. Passed directly to `matplotlib.pyplot.Figure.savefig`. Has no
        effect if `ax` is specified.
    clip_threshold : float, optional
        Absolute value of the symmetric threshold around 0.0 to which to clip
        DRC data that exceeds that threshold, by default 2.0. For example, if
        the DRC threshold is set to 2.0, all DRC values greater than 2 or less
        than -2 will be set to +2 or -2, respectively.
    contour_levels : ndarray, int
        Array of level values to use when drawing the contours on the DRC
        contour plot. Passed directly to `matplotlib.pyplot.tricontour` and
        `matplotlib.pyplot.tricontourf`
    """

    from matplotlib import pylab as plt

    # Make data frame full of data
    df = pd.DataFrame.from_dict(data=data)
    # Filter out NaN/inf values
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    # Report number of points used for DRC
    n_points = len(df)
    print("Found {n_points} non-NaN/inf points to plot.")

    # Clip values outside of DRC \in [-2,2]
    mask = df[df["drc"] > clip_threshold].index
    df.loc[mask, "drc"] = clip_threshold

    mask = df[df["drc"] < -clip_threshold].index
    df.loc[mask, "drc"] = -clip_threshold

    # Reorder columns if necessary
    df = df.loc[:, ["EO", "EN", "drc"]]

    # Extract values
    values = [np.ravel(k) for k in np.hsplit(df.values, 3)]

    # Create DRC plot
    fig = None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
        print("Making my own plot")
    contourset = ax.tricontourf(*values, levels=contour_levels, cmap="jet")

    #     Plot the marker data
    #     marker_data_set.apply(func=lambda row: plot_marker_set(row, ax), axis=1)

    # Set x and y ticks to be the same
    ticks = np.r_[-6:-1:1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.axis([-6.7, -2.5, -7, -2])

    # Place reaction inside frame
    ax.annotate(
        xy=(0.5, 0.90),
        s=rxn_str,
        xycoords="axes fraction",
        fontsize="medium",
        ha="center",
        va="top",
    )
    ax.set_xlabel("O binding energy / eV")
    ax.set_ylabel("N binding energy / eV")
    ax.axis("square")

    # If user didn't specify axis, no figure will have been created. In this
    # case, add our own color bar and return the figure and axes objects.
    if fig:
        print("Adding my own colorbar")
        fig.colorbar(
            contourset,
            label="Degree of rate control factor",
        )
        if outfile_name:
            fig.savefig(outfile_name)

        return fig, ax
    else:
        # Return the contour set object so it can be used for a figure bar in
        # the larger figure
        return contourset


def make_tof_plot(data: list, compound_str: str, ax: plt.Axes = None, outfile_name: str = None, contour_levels: np.ndarray = np.r_[-36:3.1:3],):
    """Make a degree of rate control plot for the specified reaction.

    Parameters
    ----------
    data :
        List containing data of the form [{"EO": <EO>, "EN": <EN>, "rate":
        <rate>}, ...], where <EO> is the O binding energy in eV, <EN> is the N
        binding energy in eV, and <rate> is the Campbell degree of rate control
        coefficient.
    compound_str : str
        The name of the compound (in LaTeX syntax) to show in the title of the
        plot.
    ax : plt.Axes
        The Axes object on which to place this plot. If none, Figure and Axes
        objects will be created for you.
    outfile_name : str, optional
        Name of graphics file (including extension) to which to export the
        plotted graph, by default None. If None, no plot will be written to
        disk. Passed directly to `matplotlib.pyplot.Figure.savefig`. Has no
        effect if `ax` is specified.
    contour_levels : ndarray, int
        Array of level values to use when drawing the contours on the DRC
        contour plot. Passed directly to `matplotlib.pyplot.tricontour` and
        `matplotlib.pyplot.tricontourf`
    """

    # Make data frame full of data
    df = pd.DataFrame.from_dict(data=data)

    # Filter out NaN/inf values
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    # Report number of points used for DRC
    n_points = len(df)
    print("Found {n_points} non-NaN/inf points to plot.")

    # Reorder columns if necessary
    df = df.loc[:, ["EO", "EN", "ratelog"]]

    # Extract values
    values = [np.ravel(k) for k in np.hsplit(df.values, 3)]

    # Create TOF plot
    fig = None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
        print("Making my own plot")
    contourset = ax.tricontourf(
        *values, levels=contour_levels, cmap="Spectral_r")

    ax.set_title(
        fr"""Volcano plot: {compound_str} TOF""")
    ax.set_xlabel("O binding energy / eV")
    ax.set_ylabel("N binding energy / eV")

    # If user didn't specify axis, no figure will have been created. In this
    # case, add our own color bar and return the figure and axes objects.
    if fig:
        print("Adding my own colorbar")
        fig.colorbar(
            contourset,
            label=r"$\log_{10}(\mathrm{TOF})$ /  $\mathrm{s^{-1}}$",
        )
        if outfile_name:
            fig.savefig(outfile_name)

        return fig, ax
    else:
        # Return the contour set object so it can be used for a figure bar in
        # the larger figure
        return contourset

# Enumerate all compounds and reaction strings used
compound_strings = ["Haqu", "NO3-", "H2", "N2", "N2O", "H2O", "NO",
                    "NH3", "NO3*", "H*", "NO2*", "NO*", "N*", "NH*",
                    "NH2*", "NH3*", "O*", "N2*", "N2O*", "OH*", "H2O*", "*"]

rxn_strings = ["NO + * <-> NO*", "N2 + * <-> N2*", "N2O + * <-> N2O*",
               "H2O + * <-> H2O*", "NH3 + * <-> NH3*", "NO3- + * <-> NO3*",
               "NO3* + * <-> NO2* + O*", "NO2* + * <-> NO* + O*", "NO* + * <-> N* + O*",
               "2 N* <-> N2* + *", "2 NO* <-> N2O* + O*", "N2O* + * <-> N2* + O*",
               "N* + Haqu <-> NH*", "NH* + Haqu <-> NH2*", "NH2* + Haqu <-> NH3*",
               "O* + Haqu <-> OH*", "OH* + Haqu <-> H2O*", "Haqu + * <-> H*", "H2 + 2 * <-> 2 H*"]

rxn_labels_latex = {
    "NO + * <-> NO*"        : r"$NO + {}^* -> NO^*}$",
    "N2 + * <-> N2*"        : r"$N2 + {}^* -> N2^*}$",
    "N2O + * <-> N2O*"      : r"$N2O + {}^* -> N2O^*}$",
    "H2O + * <-> H2O*"      : r"$H2O + {}^* -> H2O^*}$",
    "NH3 + * <-> NH3*"      : r"$NH3 + {}^* -> NH3^*}$",
    "NO3- + * <-> NO3*"     : r"$NO3- + {}^* -> NO3^*}$",
    "NO3* + * <-> NO2* + O*": r"$NO3^* + {}^* -> NO2^* + O^*}$",
    "NO2* + * <-> NO* + O*" : r"$NO2^* + {}^* -> NO^* + O^*}$",
    "NO* + * <-> N* + O*"   : r"$NO^* + {}^* -> N^* + O^*}$",
    "2 N* <-> N2* + *"      : r"$2N^* -> N2^* + {}^* }$",
    "2 NO* <-> N2O* + O*"   : r"$2NO^* -> N2O^* + O^*}$",
    "N2O* + * <-> N2* + O*" : r"$N2O^* + {}^* -> N2^* + O^*}$",
    "N* + Haqu <-> NH*"     : r"$N^* + H+ + e- -> NH^*}$",
    "NH* + Haqu <-> NH2*"   : r"$NH^* + H+ + e- -> NH2^*}$",
    "NH2* + Haqu <-> NH3*"  : r"$NH2^* + H+ + e- -> NH3^*}$",
    "O* + Haqu <-> OH*"     : r"$O^* + H+ + e- -> OH^*}$",
    "OH* + Haqu <-> H2O*"   : r"$OH^* + H+ + e- -> H2O^*}$",
    "Haqu + * <-> H*"       : r"$H+ + e- + {}^* -> H^*}$",
    "H2 + 2 * <-> 2 H*"     : r"$H2 + 2 {}^* ->{} 2H^*}$",
}

def extrapolate_tof(Eads_N, Eads_O, data):
    v1 = np.array([Eads_N, Eads_O])
    dists = []
    for entry in data:
        v2 = np.array([entry['EN'], entry['EO']])
        dists.append(np.linalg.norm(v1-v2))
    return data[dists.index(min(dists))]['ratelog']

def get_distance_TOF(x, y, a,b,c,pt1,pt2):
    if y > pt2[1]:
        d = np.linalg.norm(np.array(pt2)-np.array([x,y]))

    elif y < pt1[1]:
        d = np.linalg.norm(np.array(pt1)-np.array([x,y]))
    else:
        a1 = -1/a 
        c1 = y - a1*x
        xl = (-1*c - -1*c1)/(-1*a1 - -1*a)
        yl = (c1*a - c*a1)/(-1*a1 - -1*a)
        vdiff = np.array([x,y]) - np.array([xl, yl])
        d = np.linalg.norm(vdiff)
    
    return d

# Selectivity
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

def select_N2(p):

    N2_pos = [[-6.649522876274468, -6.610989010989012],
              [-6.654777070063695, -6.511355311355311],
              [-5.8521453069223766, -4.964102564102564],
              [-5.4954480763397955, -4.96996336996337],
              [-4.155190032897039, -5.052014652014652],
              [-4.03781526329297, -5.163369963369964],
              [-4.001381209024521, -5.643956043956044],
              [-3.7510184083432487, -6.071794871794872],
              [-3.8578661253820488, -6.171428571428572],
              [-4.468883133851286, -6.452747252747253]]

    hull = Delaunay(np.array(N2_pos))
    return hull.find_simplex(p)>=0


def select_NH3(p):

    NH3_pos = [[-5.811320754716982, -5.861168384879726],
               [-6.042767295597485, -5.586254295532646],
               [-5.101886792452831, -4.393127147766323],
               [-4.875471698113207, -4.189690721649485],
               [-4.553459119496855, -4.55257731958763],
               [-4.548427672955976, -4.838487972508592],
               [-4.759748427672957, -5.536769759450172],
               [-5.076729559748427, -5.828178694158076]]

    hull = Delaunay(np.array(NH3_pos))
    return hull.find_simplex(p)>=0




# Load condensed plot data
import sys
cwd = os.getcwd()
sys.path.append(cwd.replace(cwd.split('/')[-1], ''))


# with open("/home/joyvan/repos/nitrate/NO3_heatmap_files/all-condensed-plot-data-h2o-corrected_24-jul-2020.pckl", "rb") as f:
#     condensed_plot_data = pickle.load(f)
