from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    def __init__(self, env, params):
        """
        Base class for optimization algorithms.

        Parameters
        ----------
        env: BaseEnv
            Learning environment.
        params: dict
            Dictionary of parameters.
        """
        self.env = env
        self.global_params = params
        self.algo_params = params['algo_params']

    @abstractmethod
    def learn(self, num_env_steps):
        """
        Main learning loop

        Parameters
        ----------
        num_env_steps: int
            Budget of environment steps.

        """
        pass

    @abstractmethod
    def evaluate(self, n, goal=None):
        """
        Evaluation of the current policy on a set of evaluation goals.

        Parameters
        ----------
        n: int
            Number of evaluation episodes.
        goal: nd.array, optional
            Goal to be targeted during evaluation. If not provided, evaluation goals are used.

        """
    @abstractmethod
    def log(self, **kwargs):
        """
        Logging function

        """
        pass

    @abstractmethod
    def act(self, state, deterministic):
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
