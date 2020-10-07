from abc import ABC, abstractmethod

import numpy as np


class BaseMultiCostFunction(ABC):
    def __init__(self,
                 use_constraints=False):
        """
        Base class for unique cost functions.

        Parameters
        ----------
        use_constraints: bool
            Whether to use constraints on maximum cumulative costs.
        """
        self.use_constraints = use_constraints

    @abstractmethod
    def sample_goal_params(self):
        """
        Sample goal parameters.

        Returns
        -------
        goal: 1D nd.array
            Made of three params in [0, 1]: beta is the mixing parameter,
            the following are normalized constraints on the maximal values of cumulative costs.

        """
        pass

    def get_eval_goals(self, n):
        if self.use_constraints:
            # eval_goals = np.array([[0, 1, 1]] * n + [[0.1, 1, 1]] * n + [[0.2, 1, 1]] * n +  [[0.3, 1, 1]] * n + \
            #                       [[0.4, 1, 1]] * n + [[0.45, 1, 1]] * n + [[0.55, 1, 1]] * n + [[0.6, 1, 1]] * n + \
            #                       [[0.7, 1, 1]] * n + [[0.8, 1, 1]] * n + [[0.9, 1, 1]] * n + [[1, 1, 1]] * n)
            goals = []
            for beta in np.arange(0, 1.001, 0.05):
                for c in [0.25, 0.5, 1]:
                    goals += [[beta, c, 1]] * n
                    goals += [[beta, 1, c]] * n
            eval_goals = np.array(goals)
            # eval_goals = np.array([[0, 1, 1]] * n + [[0.25, 1, 1]] * n + [[0.5, 1, 1]] * n + [[0.75, 1, 1]] * n + [[1, 1, 1]] * n + \
            #                       [[0.5, 1, 0.5]] * n + [[0.5, 1, 0.25]] * n + [[0.5, 0.5, 1]] * n + [[0.5, 0.25, 1]] * n +\
            #                       [[0.3, 1, 0.5]] * n + [[0.3, 1, 0.25]] * n + [[0.3, 0.5, 1]] * n + [[0.3, 0.25, 1]] * n + \
            #                       [[0.7, 1, 0.5]] * n + [[0.7, 1, 0.25]] * n + [[0.7, 0.5, 1]] * n + [[0.7, 0.25, 1]] * n)
        else:
            goals = []
            values = np.arange(0, 1.001, 0.025)
            for v in values:
                goals += [v] * n
            eval_goals = np.atleast_2d(np.array(goals)).transpose()
            # eval_goals = np.atleast_2d(np.array([0] * n + [0.25] * n + [0.5] * n + [0.75] * n + [1] * n)).transpose()
        return eval_goals

    def get_main_goal(self):
        if self.use_constraints:
            eval_goals = np.array([0.5, 1, 1])
        else:
            eval_goals = np.array([0.5])
        return eval_goals

    @abstractmethod
    def set_goal_params(self, goal):
        """
        Set a goal.

        Parameters
        ----------
        goal: 1D nd.array


        """
        pass

    @abstractmethod
    def compute_cost(self, previous_state, state, label_to_id, action, others={}):
        """
        Compute each cost and an aggregated measure of costs as well as constraints.

        Parameters
        ----------
        previous_state: 2D nd.array
            Previous model states (either 1D or 2D with first dimension # of states).
        state: 2D nd.array
            Current model states (either 1D or 2D with first dimension # of states).
        label_to_id: dict
            Mapping between state labels and indices in the state vector.
        action: int or nd.array
            Int for discrete envs and nd.array in continuous envs.

        Returns
        -------
        cost_aggregated: float
            Aggregated cost using the beta mixing parameter.
        costs: list of floats
            All costs.
        over_constraints: list of bools
            Whether the constraints are violated.
        """
        pass

    @abstractmethod
    def compute_deaths(self, previous_state, state, label_to_id, action, others={}):
        """
        Compute death toll

       Parameters
        ----------
        previous_state: 2D nd.array
            Previous model states (either 1D or 2D with first dimension # of states).
        state: 2D nd.array
            Current model states (either 1D or 2D with first dimension # of states).
        label_to_id: dict
            Mapping between state labels and indices in the state vector.
        action: int or nd.array
            Int for discrete envs and nd.array in continuous envs.

        Returns
        -------
        int
            Number of deaths
        """
        pass

    @abstractmethod
    def compute_aggregated_cost(self, costs, constraints=None):
        """
        Compute aggregated measure of cost with mixing beta and optional constraints.
        Parameters
        ----------
        costs: 2D nd.array
            Array of costs (n_points, n_costs).
        constraints: nd.array of bools, optional
            Whether constraints are violated (n_point, n_costs).

        Returns
        -------
        float
            Aggregated cost.
        """
        pass
