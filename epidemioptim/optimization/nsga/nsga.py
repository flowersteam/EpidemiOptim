import os

import pickle
import numpy as np
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.model.sampling import Sampling
from sklearn.neighbors import NearestNeighbors
from pymoo.configuration import Configuration
Configuration.show_compile_hint = False

from epidemioptim.optimization.shared.rollout import run_rollout
from epidemioptim.optimization.base_algorithm import BaseAlgorithm
from epidemioptim.optimization.shared.networks import QNetFC
from epidemioptim.utils import compute_pareto_front


def create_problem(n_params, n_objs, runner):
    class NSGAProblem(Problem):
        """
        Defines a problem for the pymoo library
        """
        def __init__(self, nb_params, nb_objs):
            super().__init__(n_var=nb_params, n_obj=nb_objs, n_constr=0, xl=-1, xu=1)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"], out["F_std"] = runner(x)
            return out

    return NSGAProblem(n_params, n_objs)


class NSGAII(BaseAlgorithm):
    def __init__(self, env, params):
        """
        NSGA-II Algorithm based on the implementation in the Pymoo library.

        Parameters
        ----------
        env: BaseEnv
            The learning environment.
        params: dict
            All experiment params

         Attributes
        ----------
        layers: tuple of ints
            Describes sizes of hidden layers of the critics.
        logdir: str
            Logging directory
        eval_and_log_every: int
            Frequency to pring logs (in episodes).
        n_evals_if_stochastic: int
            Number of evaluation episodes if the environment is stochastic.
        stochastic: bool
            Whether the environment is stochastic.
        dims: dict
            Dimensions of states and actions.
        cost_function: BaseMultiCostFunction
            Multi-cost function.
        nb_costs: int
            Number of cost functions

        """

        super(NSGAII, self).__init__(env, params)

        # Save parameters
        self.is_multi_obj = True  # NSGA-II is a multi-obj algorithm
        self.logdir = params['logdir']
        self.log_every = self.algo_params['eval_and_log_every']
        self.seed = params['seed']
        self.popsize = self.algo_params['popsize']  # size of NSGA population
        self.layers = tuple(self.algo_params['layers'])
        self.nb_gens = self.algo_params['nb_gens']  # number of generations to run NSGA.
        self.stochastic = params['model_params']['stochastic']
        self.n_evals_if_stochastic = self.algo_params['n_evals_if_stochastic']
        self.dims = dict(s=env.observation_space.shape[0],
                         a=env.action_space.n)
        self.nb_costs = self.env.unwrapped.cost_function.nb_costs
        self.cost_function = self.env.unwrapped.cost_function

        if self.logdir:
            os.makedirs(self.logdir + 'models/', exist_ok=True)

        # Create policy
        self.policy = QNetFC(dim_state=self.dims['s'],
                             dim_goal=0,
                             dim_actions=self.dims['a'],
                             layers=self.layers,
                             goal_ids=())
        self.dim_params = self.policy.nb_params

        # Sample initial weights just like pytorch would do.
        dims = self.dims
        layers = self.layers
        class NNSampling(Sampling):
            def __init__(self, var_type=np.float) -> None:
                super().__init__()
                self.var_type = var_type

            def _do(self, problem, n_samples, **kwargs):
                # val = np.random.random((n_samples, problem.n_var))
                val = []
                for _ in range(n_samples):
                    policy = QNetFC(dim_state=dims['s'],
                                    dim_goal=0,
                                    dim_actions=dims['a'],
                                    layers=layers,
                                    goal_ids=[])
                    params = policy.get_params()
                    val.append(params)
                return np.array(val)

        # Initialize counters
        self.learn_step_counter = 0
        self.env_step_counter = 0
        self.episode = 0
        self.history = []
        self.res = None

        # define rollout function for nsga
        def runner(x):
            costs_mean = []
            costs_std = []
            for i in range(x.shape[0]):
                self.policy.set_params(x[i])
                episodes = run_rollout(policy=self,
                                       env=self.env,
                                       n=self.n_evals_if_stochastic if self.stochastic else 1,
                                       eval=False,
                                       additional_keys=['costs', 'n_icu'],
                                       )
                costs_eps = np.array([np.sum(episodes[i_ep]['costs'], axis=0) for i_ep in range(self.n_evals_if_stochastic if self.stochastic else 1)])
                costs_mean.append(costs_eps.mean(axis=0))
                costs_std.append(costs_eps.std(axis=0))

            return np.array(costs_mean), np.array(costs_std)

        # Create problem for pymoo
        self.nsga_problem = create_problem(n_params=self.dim_params,
                                           n_objs = self.nb_costs,
                                           runner=runner)

        if self.algo_params['policy'] == 'nn':
            self.algorithm = NSGA2(pop_size=self.popsize,
                                   sampling=NNSampling())
        else:
            self.algorithm = NSGA2(pop_size=self.popsize)

    def act(self, state, deterministic=False):
        """
        Policy that uses the learned critics.

        Parameters
        ----------
        state: 1D nd.array
            Current state.
        deterministic: bool
            Whether the policy should be deterministic (e.g. in evaluation mode).

        Returns
        -------
        action: nd.array
            Action vector.
        """
        return self.policy.act(state), None


    def learn(self, num_train_steps):
        """
        Main training loop.

        Parameters
        ----------
        num_train_steps: int
            Number of training steps (environment steps)

        Returns
        -------

        """
        self.res_run = minimize(problem=self.nsga_problem,
                            algorithm=self.algorithm,
                            termination=('n_gen', self.nb_gens),
                            verbose=True,
                            seed=self.seed,
                            save_history=True)
        self.res_eval = self.evaluate(n=self.n_evals_if_stochastic if self.stochastic else 1, all=True)
        F_std = self.res_run.algorithm.opt.get("F_std")
        self.res_run.F_std = F_std

        self.history = self.res_run.history
        self.log(self.res_eval)


    def evaluate(self, n=None, all=False, best=False, goal=None, reset_same_model=False):
        res = dict()

        if all:
            costs_mean = []
            costs_std = []
            for w in self.res_run.X:
                self.policy.set_params(w)
                episodes = run_rollout(policy=self,
                                       env=self.env,
                                       n=n,
                                       eval=True,
                                       reset_same_model=reset_same_model,
                                       additional_keys=['costs'],
                                       )

                costs = np.array([np.array(e['costs']).sum(axis=0) for e in episodes])
                costs_mean.append(costs.mean(axis=0))
                costs_std.append(costs.std(axis=0))

            front_ids = compute_pareto_front(costs_mean)
            costs_mean = np.array(costs_mean)
            costs_std = np.array(costs_std)
            costs_std = costs_std[front_ids]
            costs_mean = costs_mean[front_ids]
            weights = self.res_run.X[front_ids]
            res['F'] = costs_mean
            res['F_std'] = costs_std
            res['X'] = weights
            costs = costs_mean
        elif best:
            weights = self.res_eval['X']
            costs = self.res_eval['F']
            normalized_costs = np.array([c_f.scale(c) for c_f, c in zip(self.cost_function.costs, costs.transpose())]).transpose()
            agg_cost = normalized_costs.sum(axis=1)
            ind_min = np.argmin(agg_cost)
            self.policy.set_params(weights[ind_min])
            episodes = run_rollout(policy=self,
                                   env=self.env,
                                   n=n,
                                   eval=True,
                                   additional_keys=['costs'],
                                   )
            costs = np.array([np.array(e['costs']).sum(axis=0) for e in episodes])
            # res['X'] = weights[ind_min]
            for i, c_m, c_std in zip(range(costs.shape[1]), costs.mean(axis=0), costs.std(axis=0)):
                res['C{} mean'.format(i)] = c_m
                res['C{} std'.format(i)] = c_std

        elif goal is not None:
            nn_model = NearestNeighbors(n_neighbors=1)

            weights = self.res_eval['X']
            costs = self.res_eval['F']
            normalized_costs = np.array([c_f.scale(c) for c_f, c in zip(self.cost_function.costs, costs.transpose())]).transpose()
            nn_model.fit(normalized_costs)
            normalized_goal = np.atleast_2d(np.array([c_f.scale(g) for c_f, g in zip(self.cost_function.costs, goal)]))
            ind_nn = nn_model.kneighbors(normalized_goal, return_distance=False).flatten()
            self.policy.set_params(weights[ind_nn].flatten())
            episodes = run_rollout(policy=self,
                                   env=self.env,
                                   n=n,
                                   eval=True,
                                   additional_keys=['costs'],
                                   )
            costs = np.array([np.array(e['costs']).sum(axis=0) for e in episodes])
            res['X'] = weights[ind_nn]
            res['F'] = costs.mean(axis=0)
            res['F_std'] = costs.std(axis=0)
        else:
            episodes = run_rollout(policy=self,
                                   env=self.env,
                                   n=n,
                                   eval=True,
                                   additional_keys=['costs'],
                                   )
            costs = np.array([np.array(e['costs']).sum(axis=0) for e in episodes])
            for i, c_m, c_std in zip(range(costs.shape[1]), costs.mean(axis=0), costs.std(axis=0)):
                res['C{} mean'.format(i)] = c_m
                res['C{} std'.format(i)] = c_std

        return res, costs



    def load_model(self, path):
        with open(path, 'rb') as f:
            self.res_eval = pickle.load(f)

        weights = self.res_eval['X']
        costs = self.res_eval['F']
        normalized_costs = np.array([c_f.scale(c) for c_f, c in zip(self.cost_function.costs, costs.transpose())]).transpose()
        agg_cost = normalized_costs.sum(axis=1)
        ind_min = np.argmin(agg_cost)
        self.policy.set_params(weights[ind_min])

    def log(self, res_eval):

        self.res_run.problem = None
        for h in self.res_run.history:
            h.problem = None
            h.initialization = None
        self.res_run.algorithm.problem = None
        self.res_run.algorithm.initialization.sampling = None
        with open(self.logdir + 'res_train.pk', 'wb') as f:
            pickle.dump(self.res_run, f)

        with open(self.logdir + 'res_eval.pk', 'wb') as f:
            pickle.dump(res_eval, f)
        print('Run has terminated successfully')

        plot = Scatter()
        plot.add(res_eval[0]['F'], color="red")
        plot.show()
