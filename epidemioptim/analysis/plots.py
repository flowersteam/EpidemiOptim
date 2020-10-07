import os

import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import ttest_ind
from epidemioptim.utils import get_stat_func, get_repo_path
font = {'weight':'bold', 'size'   : 16}
matplotlib.rc('font', **font)

LINE = 'mean'
ERR = 'sem'
COSTS_LABELS = ['# deaths', 'GDP cost (B)']
COST_LABELS2 = ['Health cost', 'Economic cost']
XLIM = (0, 60000)
YLIM = (0, 180)
X_STEP = 50
SCATTER_WIDTH = 150
LINEWIDTH = 4
ALPHA = 0.3
DPI = 100
RES_FOLDER = get_repo_path() + "/data/results/experiments/"
LINE, ERR_MIN, ERR_MAX = get_stat_func(line=LINE, err=ERR)
PLOT_STD = True
SWITCH = True
if SWITCH:
    XLIM, YLIM = YLIM, XLIM
    X_STEP = 1
    COST_LABELS2[0], COST_LABELS2[1] = COST_LABELS2[1], COST_LABELS2[0]
    COSTS_LABELS[0], COSTS_LABELS[1] = COSTS_LABELS[1], COSTS_LABELS[0]

def setup_figure(xlabel=COSTS_LABELS[0], ylabel=COSTS_LABELS[1], xlim=XLIM, ylim=YLIM, figsize=(18, 18)):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(LINEWIDTH)
    ax.spines['right'].set_linewidth(LINEWIDTH)
    ax.spines['bottom'].set_linewidth(LINEWIDTH)
    ax.spines['left'].set_linewidth(LINEWIDTH)
    ax.tick_params(width=int(LINEWIDTH * 1.5), direction='in', length=LINEWIDTH * 3, labelsize='small')
    artists = ()
    if xlabel:
        xlab = plt.xlabel(xlabel, fontweight='bold')
        artists += (xlab,)
    if ylabel:
        ylab = plt.ylabel(ylabel, fontweight='bold')
        artists += (ylab,)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    return artists, ax

def save_fig(path, artists):
    plt.savefig(os.path.join(path), bbox_extra_artists=artists, bbox_inches='tight', dpi=DPI)
    plt.close('all')

def compute_area_under_curve(points, max_costs, min_costs):
    points = (points - min_costs) / (max_costs - min_costs)
    min_y = points[:, 1].min()
    ind_sort_c1 = np.argsort(points[:, 0])
    points = points[ind_sort_c1, :]
    area = 0
    n_points = points.shape[0]
    for i in range(n_points - 1):
        delta_x = points[i + 1, 0] - points[i, 0]
        delta_y = points[i, 1] - min_y
        area += delta_x * delta_y
    return area

def pareto_plot(central, error):
    """
    Plots a Pareto front for a given algorithm. (Ellipses show errors in two dimensions).
    Parameters
    ----------
    central: 2D nd.array
        Coordinates of solutions from the Pareto front.
    error: 2D nd.array
        Error measures of solutions from the Pareto front.

    """
    n_points = central.shape[0]
    artists, ax = setup_figure()
    if PLOT_STD:
        ellipses = []
        for i in range(n_points):
            ellipses.append(Ellipse(xy=central[i],
                                    width=error[i][0],
                                    height=error[i][1],
                                    alpha=0.1))
        for e in ellipses:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('r')

    plt.scatter(central[:, 0], central[:, 1], s=SCATTER_WIDTH, color='r')
    return artists

def compute_max_costs(folder):
    list_algs = sorted(os.listdir(folder))
    os.makedirs(folder + '/res', exist_ok=True)
    max_costs = np.zeros([2])
    min_costs = np.array([np.inf, np.inf])
    for alg in list_algs:
        if 'res' not in alg:
            alg_folder = folder + alg + '/'
            list_folds = sorted(os.listdir(alg_folder))
            for fold in list_folds:
                trial_folder = alg_folder + fold + '/'
                if 'res' not in fold:
                    try:
                        with open(trial_folder + 'res_eval2.pk', 'rb') as f:
                            res = pickle.load(f)
                    except:
                        with open(trial_folder + 'res_eval.pk', 'rb') as f:
                            res = pickle.load(f)
                    mean = res['F'].max(axis=0)  # mean of the points in Pareto front
                    if SWITCH:
                        mean[0], mean[1] = mean[1].copy(), mean[0].copy()
                    for i_c in range(2):
                        if mean[i_c] > max_costs[i_c]:
                            max_costs[i_c] = mean[i_c]
                        if mean[i_c] < min_costs[i_c]:
                            min_costs[i_c] = mean[i_c]
    return max_costs, min_costs

def extract_res(folder, algo, max_costs, min_costs):
    """
    This function go through all runs from an algorithm folder, gathers data about their Pareto front and compute the stair-case functions.
    Saves information into text file at the algorithm level.

    Parameters
    ----------
    folder: str
        Directory of the algorithm (full of directories, one for each run).
    algo: str
        Algorithm identifier

    """
    print('\n\tExtracting: ', folder.split('/')[-2])
    os.makedirs(folder + '/res', exist_ok=True)

    # First, we go through all runs and collect mean and std of the solutions in the Pareto front.
    centrals = []
    errors = []
    goals = []
    list_folds = sorted(os.listdir(folder))

    for fold in list_folds:
        if 'res' not in fold:
            try:
                with open(folder + fold + '/res_eval2.pk', 'rb') as f:
                    res = pickle.load(f)
            except:
                with open(folder + fold + '/res_eval.pk', 'rb') as f:
                    res = pickle.load(f)
            os.makedirs(folder + fold + '/plots', exist_ok=True)
            mean = res['F']  # mean of the points in Pareto front
            std = res['F_std']  # std of the points in Pareto front (over n evaluation episodes)

            # Swap costs depending on the desired x-axis.
            if SWITCH:
                mean[:, 0], mean[:, 1] = mean[:, 1].copy(), mean[:, 0].copy()
                std[:, 0], std[:, 1] = std[:, 1].copy(), std[:, 0].copy()

            artists = pareto_plot(mean, std)
            save_fig(folder + fold + '/plots/pareto_{}.pdf'.format(SWITCH), artists)

            if algo == 'DQN':
                goals.append(res['G_all'].flatten())  # collect the goal for DQN policies, so as to pull them into populations of policies
                mean = mean.flatten()
                std = std.flatten()

            centrals.append(mean)
            errors.append(std)

    # if DQN algorithm, we need to pull several policies to form a population (one for each value of the goal).
    if algo == 'DQN':
        unique_goals = np.unique(goals)
        nb_runs = np.argwhere(np.array(goals).flatten() == unique_goals[0]).size
        all_centrals = []
        all_errors = []
        centrals = np.array(centrals)
        errors = np.array(errors)
        for i_run in range(nb_runs):
            inds = np.array([np.argwhere(np.array(goals).flatten() == unique_goals[i])[i_run] for i in range(unique_goals.size)]).flatten()
            all_centrals.append(centrals[inds])
            all_errors.append(errors[inds])
        centrals = all_centrals.copy()
        errors = all_errors.copy()

    # compute areas under the curve
    areas = [compute_area_under_curve(mean, max_costs, min_costs) for mean in centrals]
    print(areas)

    # Here we form the staircase function from points in the Pareto front.
    n_lines = len(centrals)
    all_data = []
    all_data_std = []
    for i in range(n_lines):
        data = centrals[i]
        data_std = errors[i]
        sorted_inds = np.argsort(data[:, 0])
        data = data[sorted_inds]
        data_std = data_std[sorted_inds]

        # align data to compute mean
        inds = np.arange(0, XLIM[1], X_STEP)
        aligned_data = np.zeros([inds.size])
        aligned_std = np.zeros([inds.size])
        aligned_data.fill(np.nan)
        aligned_std.fill(np.nan)
        for i, cost in enumerate(inds):
            inds_inf = np.argwhere(data[:, 0] < cost).flatten()
            if inds_inf.size > 0:
                last_inf = inds_inf[-1]
                aligned_data[i] = data[last_inf, 1]
                aligned_std[i] = data_std[last_inf, 1]
        all_data.append(aligned_data)
        all_data_std.append(aligned_std)
    np.savetxt(folder + 'res/all_data_pareto_{}.txt'.format(SWITCH), np.array(all_data))
    np.savetxt(folder + 'res/all_data_pareto_std_{}.txt'.format(SWITCH), np.array(all_data_std))
    np.savetxt(folder + 'res/areas_under_curve.txt'.format(SWITCH), np.array(areas))


def plot_algo_fronts(folder):
    """
    Plot Pareto fronts of all runs for a given algorithm.

    Parameters
    ----------
    folder: str
        Directory of the algorithm (full of directories, one for each run).

    """
    print('\n\tPlotting: ', folder.split('/')[-2])
    all_data = np.loadtxt(folder + 'res/all_data_pareto_{}.txt'.format(SWITCH))
    all_data_std = np.loadtxt(folder + 'res/all_data_pareto_std_{}.txt'.format(SWITCH))

    n_lines = all_data.shape[0]
    artists, ax = setup_figure()
    inds = np.arange(0, XLIM[1], X_STEP)
    for i in range(n_lines):
        data = all_data[i]
        data_std = all_data_std[i]
        # plot all lines (one per run)
        color = matplotlib_colors[i]
        plt.plot(inds, data, c=color, linewidth=LINEWIDTH)
    central = LINE(all_data)
    errs = ERR_MIN(all_data), ERR_MAX(all_data)
    plt.plot(inds, central, linestyle='--', c='k', linewidth=3*LINEWIDTH)
    plt.fill_between(inds, errs[0], errs[1], color='k', alpha=ALPHA)
    plt.savefig(folder + 'res/all_pareto_{}.pdf'.format(SWITCH))
    plt.close('all')



def beta_plot(folder):
    """
    Custom plotting function to plot the evolution of the two costs as a function of the mixing parameter beta.

    Parameters
    ----------
    folder:  str
        Directory of the algorithm (full of directories, one for each run).

    """
    all_data = dict()
    goals = []
    list_folds = sorted(os.listdir(folder))
    for f in list_folds:
        if 'res' not in f:
            with open(folder + f + '/res_eval.pk', 'rb') as f:
                data = pickle.load(f)
            if 'G_all' in data.keys():
                for g, f, f_std in zip(data['G_all'], data['F_all'], data['F_std_all']):
                    g = np.atleast_1d(g)
                    if str(g) not in all_data.keys():
                        all_data[str(g)] = dict(F_all=[],
                                                F_std_all=[])
                        goals.append(g)
                    all_data[str(g)]['F_all'].append(f)
                    all_data[str(g)]['F_std_all'].append(f_std)

    if len(goals) > 0:
        ind = np.argsort(np.array(goals)[:, 0])
        goals = np.array(goals)[ind]
        beta_y = goals[:, 0]
        keys = [str(g) for g in goals]
        data = np.swapaxes(np.array([all_data[k]['F_all'] / np.array([XLIM[1], 110]) for k in keys]), axis1=0, axis2=1)
        central = LINE(data)
        errors = ERR_MIN(data), ERR_MAX(data)
        artists, ax = setup_figure(r"$\beta$", 'costs (% max)', xlim=[0, 1], ylim=[0, 1], figsize=(15, 10))
        nb_costs = central.shape[1]
        for i_c in range(nb_costs):
            c = colors[i_c]
            plt.plot(beta_y, central[:, i_c], c=c, linewidth=8)
            plt.fill_between(beta_y, errors[0][:, i_c], errors[1][:, i_c], color=c, alpha=ALPHA)
        leg = plt.legend(COST_LABELS2,
                         loc='upper center',
                         bbox_to_anchor=(0.5, 1.08),
                         ncol=2,
                         fancybox=True,
                         shadow=True,
                         prop={'size': 32, 'weight': 'bold'},
                         markerscale=1)
        artists += (leg,)
        save_fig(folder + 'res/beta_study.pdf', artists)

def plot_multi_algo(folder):
    """
    Plot comparison of Pareto fronts.

    Parameters
    ----------
    folder: str
        Directory that contains folders for each algorithm.

    """
    list_conds = os.listdir(folder)
    data = []
    data_std = []
    labels = []
    areas = []
    for cond in list_conds:
        if 'res' not in cond:
            labels.append(cond)
            data.append(np.loadtxt(RES_FOLDER + cond + '/res/all_data_pareto_{}.txt'.format(SWITCH)))
            data_std.append(np.loadtxt(RES_FOLDER + cond + '/res/all_data_pareto_std_{}.txt'.format(SWITCH)))
            areas.append(np.loadtxt(RES_FOLDER + cond + '/res/areas_under_curve.txt'))
    data = np.swapaxes(np.array(data), 0, 1)
    data_std = np.swapaxes(np.array(data_std), 0, 1)
    n_conds = data.shape[1]

    # Compute tests on areas under the curve
    areas = np.array(areas)
    print('\n\n\tAreas under Pareto front:\n')
    p_vals = np.ones([n_conds, n_conds])
    for i in range(n_conds):
        for j in range(n_conds):
            p_vals[i, j] = ttest_ind(areas[i, :], areas[j, :], equal_var=False)[1]
        plus_or_minus = np.array([areas[i].mean() > areas[j].mean() for j in range(n_conds)])
        sig = np.array([p_vals[i, j] < 0.05 for j in range(n_conds)])
        msg = '{}: {:.2f} +/- {:.2f}.'.format(labels[i], areas[i].mean(), areas[i].std())
        ind_pos = np.argwhere(np.logical_and(sig, plus_or_minus)).flatten()
        ind_neg = np.argwhere(np.logical_and(sig, ~plus_or_minus)).flatten()
        if ind_neg.size > 0:
            msg += '\n\tBetter than: '
            for ind in ind_neg:
                msg += '{} (p={:.2f}), '.format(labels[ind], p_vals[i, ind])
        if ind_pos.size > 0:
            msg += '\n\tWorse than: '
            for ind in ind_pos:
                msg += '{} (p={:.2f}), '.format(labels[ind], p_vals[i, ind])
        print(msg)


    if SWITCH:
        Y_SCALE = 1 / 1000
        X_SCALE = 1
        delta_sig = 3
        artists, ax = setup_figure(figsize=(15, 10), ylabel=r'# Deaths $(\times 10^3)$', ylim=(0, YLIM[1] * Y_SCALE * 1.4))
    else:
        X_SCALE = 1 / 1000
        Y_SCALE = 1
        delta_sig = 7
        artists, ax = setup_figure(figsize=(15, 10), xlabel=r'# Deaths $(\times 10^3)$', xlim=(0, XLIM[1] * X_SCALE), ylim=(0, YLIM[1] * 1.35))
    data *= Y_SCALE
    central = LINE(data)
    central_std = LINE(data_std)
    errs = ERR_MIN(data), ERR_MAX(data)
    inds = np.arange(0, XLIM[1], X_STEP) * X_SCALE
    if SWITCH:
        inds_freq = np.arange(0, inds.size, 5)
    else:
        inds_freq = np.arange(0, inds.size, 30)
    data = data[:, :, inds_freq]
    i_ref = 3
    p_vals = dict()
    for i in range(n_conds):
        if i != i_ref:
            p_val = ttest_ind(data[:, i], data[:, i_ref], equal_var=False)[1]
            p_val[np.argwhere(np.isnan(p_val))] = 1
            p_vals[i] = p_val

    counter = 0
    for i_c in range(central.shape[0]):
        plt.plot(inds, central[i_c, :], c=colors[i_c], linewidth=8)
        plt.fill_between(inds, errs[0][i_c, :], errs[1][i_c, :], color=colors[i_c], alpha=ALPHA)
        if i_c in p_vals.keys():
            counter += 1
            inds_inf = np.argwhere(np.logical_and(p_vals[i_c] < 0.05, central[i_c, inds_freq] < central[i_ref, inds_freq])).flatten()
            inds_sup = np.argwhere(np.logical_and(p_vals[i_c] < 0.05, central[i_c, inds_freq] > central[i_ref, inds_freq])).flatten()
            if inds_inf.size > 0:
                plt.scatter(inds[inds_freq][inds_inf], np.ones([inds_inf.size]) * YLIM[1] * 0.92 * Y_SCALE + delta_sig * counter, color=colors[i_c], s=180, marker='o')
            if inds_sup.size > 0:
                plt.scatter(inds[inds_freq][inds_sup], np.ones([inds_sup.size]) * YLIM[1] * 0.92 * Y_SCALE + delta_sig * counter, color=colors[i_c], s=250, marker='*')

    leg = plt.legend(labels,
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.08),
                     ncol=2,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 28, 'weight': 'bold'},
                     markerscale=1)
    artists += (leg, )
    save_fig(RES_FOLDER + 'res/pareto_{}.pdf'.format(SWITCH), artists)


if __name__ == '__main__':
    font = {'size'   : 45}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                      [0.494, 0.1844, 0.556], [0, 0.447, 0.7410], [0.3010, 0.745, 0.933], [0.85, 0.325, 0.098],
                      [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                      [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184]]

    matplotlib_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan']

    os.makedirs(RES_FOLDER + 'res/', exist_ok=True)
    max_costs, min_costs = compute_max_costs(RES_FOLDER)
    for algo in os.listdir(RES_FOLDER):
        if 'res' not in algo:
            algo_folder = RES_FOLDER + algo + '/'
            # extract_res(algo_folder, algo, max_costs, min_costs)
            # plot_algo_fronts(algo_folder)
            # beta_plot(algo_folder)

    print('\n\tComparison plots')
    plot_multi_algo(RES_FOLDER)




