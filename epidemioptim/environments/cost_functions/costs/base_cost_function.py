from abc import ABC, abstractmethod

import numpy as np


class BaseCostFunction(ABC):
    def __init__(self,
                 scale_factor=1,
                 range_constraints=()):
        """
        Base class for unique cost functions.

        Parameters
        ----------
        scale_factor: float
            The factor by which the cost should be scaled (for cost aggregation).
        range_constraints: tuple
            Minimum and maximum values for the constraint on the cumulative cost.
        """
        self.scale_factor = scale_factor
        self.range_constraints = range_constraints
        self.normalized_constraint = 1.  # set to max value

    def sample_constraint(self):
        self.normalized_constraint = np.random.rand()
        return self.normalized_constraint

    def set_constraint(self, value):
        self.normalized_constraint = value

    def scale(self, cost):
        """
        Scales the cost

        Parameters
        ----------
        cost: float
            Unscaled cost.

        """
        return cost / self.scale_factor

    def compute_normalized_constraint(self, constraint):
        """
         normalized constraints as a function of the true constraint.

        Parameters
        ----------
        normalized_constraint: float
            Value in [0, 1].

        Returns
        -------
        float
            Value in the unit of the cost.

        """
        return (constraint - self.range_constraints[0]) / (self.range_constraints[1] - self.range_constraints[0])

    def compute_constraint(self, normalized_constraint):
        """
        Return true constraints as a function of normalized constraint.

        Parameters
        ----------
        normalized_constraint: float
            Value in [0, 1].

        Returns
        -------
        float
            Value in the unit of the cost.

        """
        return normalized_constraint * (self.range_constraints[1] - self.range_constraints[0]) + self.range_constraints[0]

    def check_constraint(self, value, normalized_constraint):
        """
        Checks whether constraints are respected.

        Parameters
        ----------
        values: list
            Values to be compared to constraints.
        normalized_constraints: list
            Normalized values of the constraints.

        Returns
        -------
        over_constraints: nd.array of bools
            Each element says whether the constraint as been violated.

        """
        constraint = self.compute_constraint(normalized_constraint)
        over_constraint = int(value > constraint)
        return over_constraint

    @abstractmethod
    def compute_cost(self, previous_state, state, label_to_id, action, others={}):
        """
        Computes cost since the last state.

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

        """
        pass

    @abstractmethod
    def compute_cumulative_cost(self, previous_state, state, label_to_id, action, others={}):
        """
        Computes cumulative costs

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

        """
        pass
