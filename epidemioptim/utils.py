from abc import ABC, abstractmethod
import os
import random
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
plt.rcParams['figure.constrained_layout.use'] = True
font = {'size'   : 13}
import matplotlib
matplotlib.rc('font', **font)


# # # # # # # # # # # # # # # # # # # # # # # #
# Plots
# # # # # # # # # # # # # # # # # # # # # # # #


def plot_stats(t, states, labels, legends=None, title=None, lockdown=None, icu_capacity=None, axs=None, fig=None, time_jump=1, show=False):
    n_plots = len(states)
    if axs is None:
        print_a = True
        x = int(np.sqrt(n_plots))
        y = int(n_plots / x - 1e-4) + 1
        fig, axs = plt.subplots(x, y, figsize=(12, 7))
        axs = axs.ravel()
    else:
        print_a = False

    for i in range(n_plots):
        if isinstance(states[i], list):
            axs[i].plot(t, np.array(states[i]).transpose(), linewidth=2, color='#004c8f')
            # if legends is not None:
            #     if legends[i] is not None:
            #         axs[i].legend(legends[i], frameon=False, fontsize=15, prop={'weight': 'normal'})
        else:
            axs[i].plot(t, states[i], linewidth=5, color='#004c8f')

        axs[i].set_ylabel(labels[i], fontweight='bold')
        if i == 4:
            axs[i].set_xlabel('days', fontweight='bold')
        axs[i].set_xticks([0, 100, 200, 300])
        axs[i].spines['top'].set_linewidth(2)
        axs[i].spines['right'].set_linewidth(2)
        axs[i].spines['bottom'].set_linewidth(2)
        axs[i].spines['left'].set_linewidth(2)
        axs[i].tick_params(width=int(3), direction='in', length=5, labelsize='small')
        # axs[i].set_xticklabels([str(x) if isinstance(x, np.int64) else '{:.2f}'.format(x) for x in axs[i].get_xticks()], {'weight': 'bold'})
        # axs[i].set_yticklabels(axs[i].get_yticks(), {'weight': 'bold'})

        # if labels[i] == 'H' and icu_capacity is not None:
        #     axs[i].plot(t, states[i] * 0.25, linestyle="--", color='tab:blue')
        #     axs[i].axhline(xmin=t[0], xmax=t[-1], y=icu_capacity, linewidth=1, color='r', linestyle='--')
        #     axs[i].legend(['H', 'ICU', 'ICU capacity'], frameon=False)

    # plot lockdown days (for RL simulations)
    if lockdown is not None and print_a:
        inds_lockdown = np.argwhere(lockdown == 1).flatten() * time_jump
        for i in range(len(labels)):
            max_i = np.max(states[i])
            range_i = max_i - np.min(states[i])
            y_lockdown = np.ones([inds_lockdown.size]) * max_i + 0.05 * range_i
            axs[i].scatter(inds_lockdown, y_lockdown, s=10, c='red')

    if title:
        fig.suptitle(title)
    if show:
        plt.show()
    return axs, fig


def get_stat_func(line='mean', err='std'):
    """
    Wrapper around statistics measures: central tendencies (mean, median), and errors (std, sem, percentiles, etc)

    Parameters
    ----------
    line: str
        Central tendencies (mean or median).
    err: str
        Error (std, sem, range or interquartile)

    Returns
    -------
    line_f, err_min, err_max: functions
        Functions ready to apply to data (including data containing nans) for the central tendency, low error and high error.

    """
    if line == 'mean':
        def line_f(a):
            return np.nanmean(a, axis=0)
    elif line == 'median':
        def line_f(a):
            return np.nanmedian(a, axis=0)
    else:
        raise NotImplementedError

    if err == 'std':

        def err_plus(a):
            return line_f(a) + np.nanstd(a, axis=0)

        def err_minus(a):
            return line_f(a) - np.nanstd(a, axis=0)
    elif err == 'sem':

        def err_plus(a):
            return line_f(a) + np.nanstd(a, axis=0) / np.sqrt(a.shape[0])

        def err_minus(a):
            return line_f(a) - np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
    elif err == 'range':

        def err_plus(a):
            return np.nanmax(a, axis=0)

        def err_minus(a):
            return np.nanmin(a, axis=0)
    elif err == 'interquartile':

        def err_plus(a):
            return np.nanpercentile(a, q=75, axis=0)

        def err_minus(a):
            return np.nanpercentile(a, q=25, axis=0)
    else:
        raise NotImplementedError

    return line_f, err_minus, err_plus

# # # # # # # # # # # # # # # # # # # # # # # #
# Distributions
# # # # # # # # # # # # # # # # # # # # # # # #


class BaseDist(ABC):
    def __init__(self, params, stochastic):
        """
        Base distribution class.

        Parameters
        ----------
        params:
            Parameters of the distribution.
        stochastic: bool
            Whether the sampling is stochastic.
        """
        self.params = params
        self.stochastic = stochastic

    @abstractmethod
    def sample(self, n=1):
        """
        Sample from the distribution.

        Parameters
        ----------
        n: int
            Number of samples.

        Returns
        -------
        a:
           Sampled values (nd.array if n>1).
        """
        pass


class NormalDist(BaseDist):
    """
    Normal distribution.

    Parameters
    ----------
    params: list of size 2
        These are the mean and stdev of the normal distribution.
    stochastic: bool
        Whether the sampling is stochastic.
    """
    def __init__(self, params, stochastic):
        super(NormalDist, self).__init__(params, stochastic)
        assert len(self.params) == 2, 'params should be a list of length 2: [mean, std]'
        if self.params[1] == 0:
            self.params[1] += 1e-6
        assert self.params[1] > 0, 'params should be a list of length 2: [mean, std]'
        self.mean, self.std = self.params

    def sample(self, n=1):
        if self.stochastic:
            samples = np.random.normal(self.mean, self.std, size=n)
        else:
            samples = np.array([self.mean] * n)
        return float(samples) if n == 1 else samples


class LogNormalDist(BaseDist):
    """
    Log-normal distribution.

    Parameters
    ----------
    params: list of size 2
        These are the mean and stdev of the underlying normal distribution.
    stochastic: bool
        Whether the sampling is stochastic.
    """
    def __init__(self, params, stochastic):
        super(LogNormalDist, self).__init__(params, stochastic)
        assert len(self.params) == 2, 'params should be a list of length 2: [mean, std]'
        if self.params[1] == 0:
            self.params[1] += 1e-6
        assert self.params[1] > 0, 'params should be a list of length 2: [mean, std]'
        self.mean, self.std = self.params

    def sample(self, n=1):
        if self.stochastic:
            samples = np.random.normal(self.mean, self.std, size=n)
        else:
            samples = np.array([self.mean] * n)
        return np.exp(float(samples)) if n == 1 else np.exp(samples)


class ContUniformDist(BaseDist):
    """
    Continuous uniform distribution

    Parameters
    ----------
    params: list of size 2
        The first value is the minimum, the second is the maximum. Deterministic value is the rounded average.
    stochastic: bool
        Whether the sampling is stochastic.
    """
    def __init__(self, params, stochastic):
        super(ContUniformDist, self).__init__(params, stochastic)
        assert len(self.params) == 2, 'params should be a list of length 2: [min, max]'
        self.min, self.max = self.params

    def sample(self, n=1):
        if self.stochastic:
            samples = np.random.uniform(self.min, self.max, size=n)
        else:
            samples = np.array([(self.max - self.min) / 2] * n)
        return float(samples) if n == 1 else samples


class DiscreteUniformDist(BaseDist):
    """
    Uniform distribution of ints

    Parameters
    ----------
    params: list of size 2
        The first value is the minimum, the second is the maximum. Deterministic value is the rounded average.
    stochastic: bool
        Whether the sampling is stochastic.
    """
    def __init__(self, params, stochastic):
        super(DiscreteUniformDist, self).__init__(params, stochastic)
        assert len(self.params) == 2, 'params should be a list of length 2: [min, max]'
        self.min, self.max = self.params
        assert isinstance(self.min, int), 'params should be int'
        assert isinstance(self.max, int), 'params should be int'

    def sample(self, n=1):
        if self.stochastic:
            samples = np.random.randint(self.min, self.max, size=n)
        else:
            samples = np.array([int((self.max - self.min) / 2)] * n)
        return int(samples) if n == 1 else samples


class DiracDist(BaseDist):
    """
    Dirac distribution.

    Parameters
    ----------
    params: float
        Value of the Dirac.
    stochastic: bool
        Whether the sampling is stochastic.
    """
    def __init__(self, params, stochastic):
        super(DiracDist, self).__init__(params, stochastic)
        assert isinstance(float(params), float), 'params should be a single value'

    def sample(self, n=1):
        samples = np.array([self.params] * n)
        return float(samples) if n == 1 else samples


class DiscreteDist(BaseDist):
    def __init__(self, params, stochastic):
        """
        Discrete distribution.

        Parameters
        ----------
        params: list of size 3
            First element is the list of values, second the list of probabilities, third the value in the deterministic case.
        stochastic: bool
            Whether the sampling is stochastic.
        """
        super(DiscreteDist, self).__init__(params, stochastic)
        assert isinstance(params, list), "params should be a list of two lists: first the values, second the probabilities"
        assert len(params) == 3, "the third parameter must be the value in the deterministic case"
        assert len(params[0]) == len(params[1]), "two lists in params should be the same lengths (values and probas)"
        self.values = np.array(params[0])
        self.probabilities = np.array(params[1])
        self.deterministic_value = np.array([params[2]])

    def sample(self, n=1):
        if self.stochastic:
            samples = np.random.choice(self.values, p=self.probabilities, size=n)
        else:
            samples = np.array([self.deterministic_value] * n)
        return float(samples) if n == 1 else samples

# # # # # # # # # # # # # # # # # # # # # # # #
# Others
# # # # # # # # # # # # # # # # # # # # # # # #


def get_repo_path():
    dir_path = os.path.dirname(os.path.realpath(__file__)).split('/')
    if dir_path.count('epidemioptim') == 1:
        start_ind = dir_path.index('epidemioptim')
    else:
        start_ind = - (list(reversed(dir_path)).index('epidemioptim') + 1)

    repo_path = '/'.join(dir_path[:start_ind]) + '/'
    return repo_path


def get_logdir(params):
    """
    Create logging directory.

    Parameters
    ----------
    params: dict
        Params of the experiment required to create the logging directory.

    Returns
    -------

    """
    repo_path = get_repo_path()
    logdir = repo_path + 'data/results/' + params['env_id'] + '/' + params['algo_id'] + '/' + params['expe_name']
    if os.path.exists(logdir):
        directory = logdir + '_'
        trial_id = params['trial_id']
        i = 0
        while True:
            logdir = directory + str(trial_id + i * 100) + '/'
            if not os.path.exists(logdir):
                break
            i += 1
    else:
        logdir += '/'
    os.makedirs(logdir)
    print('Logging to: ', logdir)
    params['logdir'] = logdir
    with open(logdir + 'params.json', 'w') as f:
        json.dump(params, f)
    return params


def set_seeds(seed):
    """
    Set all seeds.

    Parameters
    ----------
    seed: int
        Random seed.
    env: Gym Env
        Gym environment that should be seeded.
    """
    if seed is None:
        seed = np.random.randint(1e6)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def mv2musig(m, v):
    """
    Convert mean and variance of log-normal distribution into mean and stdev of underlying normal distribution
    Parameters
    ----------
    m: float
       Mean of log-normal distribution.
    v: float
       Variance of log-normal distribution.

    Returns
    -------
    mu: float
        Mean of underlying normal distribution.
    sig: float
         Stdev of underlying normal distribution.
    """
    sig = np.sqrt(np.log(v / np.exp(2 * np.log(m)) + 1))
    mu = np.log(m) - sig ** 2 / 2
    return mu, sig


def musig2mv(mu, sig):
    """
    Converts mean and stdev of normal distribution into mean and variance of log-normal distribution.

    Parameters
    ----------
    mu: float
        Mean of normal distribution.
    sig: float
         Stdev of normal distribution.

    Returns
    -------
    m: float
       Mean of log-normal distribution.
    v: float
       Variance of log-normal distribution.
    """
    m = np.exp(mu + sig ** 2 / 2)
    v = (np.exp(sig ** 2) - 1) * np.exp(2 * mu + sig ** 2)
    return m, v


def compute_pareto_front(costs: list):
    """
    Find rows of entries in the Pareto front.
    Parameters
    ----------
    costs: list of arrays
        List of arrays of costs.

    Returns
    -------
    front_ids: list of ints
        List of row indices of elements from the pareto front.
    """
    front_ids = []
    n_points = len(costs)
    for ind1 in range(n_points):
        pareto = True
        for ind2 in range(n_points):
            r11, r12 = costs[ind1]
            r21, r22 = costs[ind2]

            if ind1 != ind2:
                if (r21 > r11 and r22 >= r12) or (r21 >= r11 and r22 > r12):
                    pareto = False
                    break
        if pareto:
            front_ids.append(ind1)
    return front_ids

class Logger:
    def __init__(self, keys, logdir):
        """
        Logging class

        Parameters
        ----------
        keys: list of str
            Keys that should be logged after every evaluation (in order of appearance in prints).
        logdir: str
            Path where the logs should be saved

        Attributes:
        ----------
        data: dict of list
            Tracks all metrics in keys in lists.
        """
        self.keys = keys
        self.data = dict(zip(keys, [[] for _ in range(len(keys))]))
        self.logdir = logdir

    def add(self, new_data):
        """
        Adds new entries to the logs.

        Parameters
        ----------
        new_data: dict
            New data should contain one metric for each key

        """
        assert sorted(list(new_data.keys())) == sorted(self.keys)
        for k in new_data.keys():
            self.data[k].append(new_data[k])

    def save(self):
        data = pd.DataFrame(self.data)
        data.to_csv(self.logdir + 'progress.csv')

    def print_last(self):
        msg = '---------------\n'
        goal_keys = []
        for k in self.keys:
            if 'g:' in k:
                goal_keys.append(k)
        if len(goal_keys) > 0:
            for k in self.keys:
                if 'g:' not in k:
                    msg += k + ': {:.2f}\n\t'.format(self.data[k][-1])
            goals = set([k.split('g:')[1][1:].split(':')[0] for k in goal_keys])
            for g in sorted(list(goals)):
                nb_costs = (sum([g in k for k in goal_keys]) - 2) // 2
                key_mean = 'Eval, g: {}: mean_agg'.format(g)
                key_std = 'Eval, g: {}: std_agg'.format(g)
                keys_costs_mean = ['Eval, g: {}: mean_C{}'.format(g, i) for i in range(nb_costs)]
                keys_costs_std = ['Eval, g: {}: std_C{}'.format(g, i) for i in range(nb_costs)]
                for i in range(nb_costs):
                    if i == 0:
                        msg += 'Eval, g: {}, '.format(g)
                    msg += 'C{}: {:.2f} +/- {:.2f}, '.format(i+1, self.data[keys_costs_mean[i]][-1], self.data[keys_costs_std[i]][-1])
                msg += 'Agg: {:.2f} +/- {:.2f}\n\t'.format(self.data[key_mean][-1], self.data[key_std][-1])

        else:
            for k in self.keys:
                msg += k + ': {:.2f}\n\t'.format(self.data[k][-1])

        print(msg)
        with open(self.logdir + 'log.txt', "a") as f:
            f.write(msg)


