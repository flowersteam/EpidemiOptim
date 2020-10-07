from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    def __init__(self,
                 internal_states_labels: list,
                 internal_params_labels: list,
                 range_delay=None,
                 stochastic=False):
        """This is an abstract class for epidemiological models.

        Parameters
        ----------
        internal_states_labels: list of str
                                Labels for each internal state of the model.
        internal_params_labels: list of str
                                Labels for each parameter of the model.
        range_delay: tuple
            Min and max delay between epidemic start and episode start.
        stochastic: bool
                    Whether to use stochastic models or not.

        Attributes
        ----------
        internal_states_labels
        internal_params_labels
        initial_state: 1D nd.array
                       Initial internal state of the model.
        initial_internal_params: 1D nd.array
                                 Initial internal params of the model.
        current_state: 1D nd.array
                       Current state of the model.

        """

        self.internal_states_labels = internal_states_labels
        self.internal_params_labels = sorted(internal_params_labels)
        self.stochastic = stochastic

        self.initial_state = None
        self.initial_internal_params = None
        self.current_state = None
        self.range_delay = range_delay
        self.reset()  # reset model to initial states and parameters

    def reset(self, delay=None) -> None:
        """Resets the model parameters, and state, add delay.

        Parameters
        ----------
        delay: int, optional
               Number of days the model should be run for before the start of the episode.
               Default is 0.

        """
        self._sample_model_params()
        self._sample_initial_state()
        self._reset_state()
        if self.stochastic:
            if delay is not None:
                self.delay(random=False, delay=delay)
            else:
                self.delay()

    def reset_same_model(self) -> None:
        """
        Resets the model to its former initial conditions with its current parameters.

        """
        self._reset_model_params()
        self._reset_state()

    def delay(self, random=True, delay=None):
        """
        Run the model for 'delay' steps before the beginning of the episode.

        Parameters
        ----------
        delay: int
               Delay before the episode starts in days.

        Returns
        -------
        1D nd.array
            Latest model state
        """
        if not random:
            assert delay is not None
        else:
            if self.range_delay is not None:
                delay = np.random.randint(*self.range_delay)
            else:
                delay = 0
        if delay > 0:
            return self.run_n_steps(n=delay)[-1]


    @abstractmethod
    def _sample_initial_state(self):
        """
        Samples the initial state of the model (abstract)

        """
        pass

    @abstractmethod
    def _sample_model_params(self):
        """
        Samples parameters of the model (abstract)

        """
        pass

    @abstractmethod
    def run_n_steps(self,
                    current_state=None,
                    n=1):
        """
        Runs the model for n steps.

        Parameters
        ----------
        current_state: 1D nd.array
                       Current state of the model.
        n: int
           Number of steps the model should be run for.

        Returns
        -------
        new_state: 2D nd.array
                   Returns the list of n states from the initial step to the latest step.

        """
        pass

    def _reset_model_params(self) -> None:
        """
        Resets model parameters to their initial values.

        """
        self.current_internal_params = self.initial_internal_params.copy()

    def _get_model_params(self) -> tuple:
        """
        Get current parameters of the model

        Returns
        -------
        tuple
            tuple of the model parameters in the order of the list of labels
        """
        return tuple([self.current_internal_params[k] for k in self.internal_params_labels])

    def _reset_state(self):
        """
        Resets model state to initial state.
        """
        self.current_state = dict(zip(self.internal_states_labels, np.array([self.initial_state['{}0'.format(s)] for s in self.internal_states_labels])))

    def _get_current_state(self):
        """
        Get current state in the order of state labels.


        """
        return np.array([self.current_state['{}'.format(s)] for s in self.internal_states_labels])

    def _set_current_state(self, current_state):
        """
        Set current state to given values.

        Parameters
        ----------
        current_state: 1D nd.array
                       State the current state should be set to.

        """
        self.current_state = dict(zip(self.internal_states_labels, current_state))

    def _convert_to_labelled_states(self, states):
        """
        Converts the state into a dict where keys are state labels
        Parameters
        ----------
        states: 1D nd.array
                Model state.

        Returns
        -------
        dict
            Dict where keys are state labels and values are states.

        """
        return  zip(self.internal_states_labels, states.transpose())





