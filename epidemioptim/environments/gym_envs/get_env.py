import gym


def get_env(env_id, **kwargs):
    """
    Get environment

    Parameters
    ----------
    env_id: str
        Environment identifier.
    kwargs: dict
        Parameters of the environment

    """
    env = gym.make(env_id, **kwargs)

    return env


if __name__ == '__main__':
    from epidemioptim.utils import plot_stats
    from epidemioptim.environments.cost_functions import get_cost_function
    from epidemioptim.environments.models import get_model
    from epidemioptim.environments.gym_envs import get_env

    import numpy as np

    simulation_horizon = 364
    stochastic = False
    region = 'IDF'

    model = get_model(model_id='prague_seirah', params=dict(region=region,
                                                      stochastic=stochastic))

    N_region = model.pop_sizes[region]
    N_country = np.sum(list(model.pop_sizes.values()))
    ratio_death_to_R = 0.005

    cost_function = get_cost_function(cost_function_id='multi_cost_death_gdp_controllable', params=dict(N_region=N_region,
                                                                                          N_country=N_country,
                                                                                          ratio_death_to_R=ratio_death_to_R)
                                      )

    env = get_env(env_id='EpidemicDiscrete-v0', cost_function=cost_function, model=model, simulation_horizon=simulation_horizon)
    env.reset()

    actions = np.random.choice([0, 1], size=53)
    t = 0
    r = 0
    done = False
    while not done:
        out = env.step(actions[t])
        t += 1
        r += out[1]
        done = out[2]
    stats = env.unwrapped.get_data()

    # plot model states
    plot_stats(t=stats['history']['env_timesteps'],
               states=np.array(stats['history']['model_states']).transpose(),
               labels=stats['model_states_labels'],
               lockdown=np.array(stats['history']['lockdown']),
               icu_capacity=stats['icu_capacity'],
               time_jump=stats['time_jump'])
    plot_stats(t=stats['history']['env_timesteps'][1:],
               states=stats['stats_run']['to_plot'],
               labels=stats['stats_run']['labels'],
               legends=stats['stats_run']['legends'],
               title=stats['title'],
               lockdown=np.array(stats['history']['lockdown']),
               time_jump=stats['time_jump'],
               show=True
               )
