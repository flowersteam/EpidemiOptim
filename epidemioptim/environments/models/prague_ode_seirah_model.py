# This model is an implementation of:
# Population modeling of early COVID-19 epidemic dynamics in French regions and estimation of the lockdown impact on infection rate
# Prague et al., 2020

from scipy.integrate import odeint
import pandas as pd

from epidemioptim.environments.models.base_model import BaseModel
from epidemioptim.utils import *

PATH_TO_FITTED_PARAMS = get_repo_path() + '/data/model_data/estimatedIndividualParameters.csv'
PATH_TO_FITTED_COV = get_repo_path() + '/data/model_data/data_cov.csv'

# ODE model
def seirah_model(y: tuple,
                 t: int,
                 De: float,
                 Dh: float,
                 Di: float,
                 Dq: float,
                 N: int,
                 alpha: float,
                 b: float,
                 r: float):
    """
    SEIRAH epidemiological model from Population modeling of early COVID-19 epidemic dynamics in French
    regions and estimation of the lockdown impact on infection rate, Prague et al., 2020.
    Parameters
    ----------
    y: tuple
       Current states SEIRAH.
       y = [S, E, I, R, A, H]
       S: # susceptible individuals
       E: # individuals in latent state
       I: # symptomatic infected individuals
       R: # recovered & dead individuals (deaths represent 0.5 % of R).
       A: # asymptomatic infected individuals
       H: # hospitalized individuals
    t: int
       Timestep.
    De: float
        Latent period (days).
    Dh: float
        Average duration of hospitalizations (days).
    Di: float
        Infectious period (days).
    Dq: float
        Duration from illness onset to hospital (days).
    N: int
        Population size.
    alpha: float
        Ratio between transmission rates of reported vs not-reported, in [0, 1].
    b: float
       Transmission rate.
    r: float
       Verification rate

    Returns
    -------
    tuple
        Next states.
    """
    S, E, I, R, A, H = y
    dSdt = - b * S * (I + alpha * A) / N

    dEdt = b * S * (I + alpha * A) / N - E / De

    dIdt = r * E / De - I / Dq - I / Di

    dRdt = (I + A) / Di + H / Dh

    dAdt = (1 - r) * E / De - A / Di

    dHdt = I / Dq - H / Dh

    dydt = [dSdt, dEdt, dIdt, dRdt, dAdt, dHdt]

    return dydt


class PragueOdeSeirahModel(BaseModel):
    def __init__(self,
                 region='IDF',
                 stochastic=False,
                 noise_params=0.1,
                 range_delay=(0, 21)
                 ):
        """
        Implementation of the SEIRAH model from Prague et al., 2020:
        Population modeling of early COVID-19 epidemic dynamics in French regions and estimation of the lockdown impact on infection rate.

        Parameters
        ----------
        region: str
                Region identifier.
        stochastic: bool
                    Whether to use stochastic models or not.
        noise_params: float
                      Normally distributed parameters have an stdev of 'noise_params' x their mean.

        Attributes
        ---------
        region
        stochastic
        noise_params
        """
        self.fitted_params = pd.read_csv(PATH_TO_FITTED_PARAMS)
        self.fitted_cov = pd.read_csv(PATH_TO_FITTED_COV)
        self._regions = list(self.fitted_params['id'])
        self.pop_sizes = dict(zip(self.fitted_params['id'], (self.fitted_params['popsize'])))
        assert region in self._regions, 'region code should be one of ' + str(self._regions)

        self.region = region
        self.stochastic = stochastic
        self.noise = noise_params
        self._all_internal_params_distribs = dict()
        self._all_initial_state_distribs = dict()

        # Initialize distributions of parameters and initial conditions for all regions
        self.define_params_and_initial_state_distributions()

        # Sample initial conditions and initial model parameters
        internal_params_labels = list(self._all_internal_params_distribs['IDF'].keys())
        internal_params_labels.remove('icu')  # remove irrelevant keys
        internal_params_labels.remove('N_av')
        internal_params_labels.remove('beta1')
        internal_params_labels.remove('beta2')
        internal_params_labels.remove('beta3')
        internal_params_labels.remove('beta4')

        # Define ODE SEIRAH model
        self.internal_model = seirah_model

        super().__init__(internal_states_labels=['S', 'E', 'I', 'R', 'A', 'H'],
                         internal_params_labels=internal_params_labels,
                         stochastic=stochastic,
                         range_delay=range_delay)



    def define_params_and_initial_state_distributions(self):
        """
        Extract and define distributions of parameters for all French regions
        """

        label2ind = dict(zip(list(self.fitted_cov.columns), np.arange(len(self.fitted_cov.columns))))
        for i in self.fitted_params.index:
            r = self.fitted_params['id'][i]  # region
            self._all_internal_params_distribs[r] = dict(b_fit=LogNormalDist(params=mv2musig(self.fitted_params['b1_mean'][i],
                                                                                             self.fitted_cov['b1_pop'][label2ind['b1_pop']]),
                                                                             stochastic=self.stochastic),
                                                         r_fit=NormalDist(params=[0.034, 0.034 * self.noise], stochastic=self.stochastic),
                                                         N=DiracDist(params=self.fitted_params['popsize'][i], stochastic=self.stochastic),
                                                         N_av=DiracDist(params=float(np.mean(self.fitted_params['popsize'])), stochastic=self.stochastic),
                                                         Dq_fit=LogNormalDist(params=mv2musig(self.fitted_params['Dq_mean'][i],
                                                                                              self.fitted_cov['Dq_pop'][label2ind['Dq_pop']]),
                                                                              stochastic=self.stochastic),
                                                         De=NormalDist(params=[5.1, 5.1 * self.noise], stochastic=self.stochastic),
                                                         Dh=NormalDist(params=[30, 30 * self.noise], stochastic=self.stochastic),
                                                         Di=NormalDist(params=[2.3, 2.3 * self.noise], stochastic=self.stochastic),
                                                         alpha=NormalDist(params=[0.55, 0.55 * self.noise], stochastic=self.stochastic),
                                                         icu=DiracDist(params=self.fitted_params['ICUcapacity'][i], stochastic=self.stochastic),
                                                         beta1=NormalDist(params=[self.fitted_params['betaw1_mean'][i],
                                                                                  np.sqrt(self.fitted_cov['betaw1_pop'][label2ind['betaw1_pop']])],
                                                                          stochastic=self.stochastic),
                                                         beta2=NormalDist(params=[self.fitted_params['betaw2_mean'][i],
                                                                                  np.sqrt(self.fitted_cov['betaw2_pop'][label2ind['betaw2_pop']])],
                                                                          stochastic=self.stochastic),
                                                         beta3=NormalDist(params=[self.fitted_params['betaw3_mean'][i],
                                                                                  np.sqrt(self.fitted_cov['betaw3_pop'][label2ind['betaw3_pop']])],
                                                                          stochastic=self.stochastic),
                                                         beta4=NormalDist(params=[self.fitted_params['betaw4_mean'][i],
                                                                                  np.sqrt(self.fitted_cov['betaw4_pop'][label2ind['betaw4_pop']])],
                                                                          stochastic=self.stochastic),
                                                         )
            self._all_initial_state_distribs[r] = dict(E0=LogNormalDist(params=mv2musig(self.fitted_params['initE_mean'][i], self.fitted_cov['initE_pop'][label2ind['initE_pop']]),
                                                                        stochastic=self.stochastic),
                                                       I0=DiracDist(params=self.fitted_params['I0_kalman_mean'][i], stochastic=self.stochastic),
                                                       R0=DiracDist(params=0, stochastic=self.stochastic),
                                                       A0=DiracDist(params=1, stochastic=self.stochastic),  # is updated below
                                                       H0=DiracDist(params=self.fitted_params['H0_kalman_mean'][i], stochastic=self.stochastic)
                                                       )

    def _sample_initial_state(self):
        """
        Samples an initial model state from its distribution (Dirac distributions if self.stochastic is False).


        """
        self.initial_state = dict()
        for k in self._all_initial_state_distribs[self.region].keys():
            self.initial_state[k] = self._all_initial_state_distribs[self.region][k].sample()

        # A0 is computed as a function of I0 and r_fit (see Prague et al., 2020)
        self.initial_state['A0'] = self.initial_state['I0'] * (1 - self.current_internal_params['r_fit']) / self.current_internal_params['r_fit']
        for k in self._all_initial_state_distribs[self.region].keys():
            self.initial_state[k] = int(self.initial_state[k])

        # S0 is computed from other states, as the sum of all states equals the population size N
        self.initial_state['S0'] = self.current_internal_params['N'] - np.sum([self.initial_state['{}0'.format(s)] for s in self.internal_states_labels[1:]])

    def _sample_model_params(self):
        """
        Samples parameters of the model from their distribution (Dirac distributions if self.stochastic is False).

        """
        self.initial_internal_params = dict()
        for k in self._all_internal_params_distribs[self.region].keys():
            self.initial_internal_params[k] = self._all_internal_params_distribs[self.region][k].sample()
        self._reset_model_params()

    def run_n_steps(self, current_state=None, n=1, labelled_states=False):
        """
        Runs the model for n steps

        Parameters
        ----------
        current_state: 1D nd.array
                       Current model state.
        n: int
           Number of steps the model should be run for.

        labelled_states: bool
                         Whether the result should be a dict with state labels or a nd array.

        Returns
        -------
        dict or 2D nd.array
            Returns a dict if labelled_states is True, where keys are state labels.
            Returns an array of size (n, n_states) of the last n model states.

        """
        if current_state is None:
            current_state = self._get_current_state()

        # Use the odeint library to run the ODE model.
        z = odeint(self.internal_model, current_state, np.linspace(0, n, n + 1), args=self._get_model_params())
        self._set_current_state(current_state=z[-1].copy())  # save new current state

        # format results
        if labelled_states:
            return self._convert_to_labelled_states(np.atleast_2d(z[1:]))
        else:
            return np.atleast_2d(z[1:])


if __name__ == '__main__':
    # Get model
    model = PragueOdeSeirahModel(region='IDF',
                                 stochastic=False)

    # Run simulation
    simulation_horizon = 364
    model_states = model.run_n_steps(n=simulation_horizon)

    # Plot
    time = np.arange(simulation_horizon)
    labels = model.internal_states_labels

    plot_stats(t=time,
               states=model_states.transpose(),
               labels=labels,
               show=True)
