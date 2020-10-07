import matplotlib.pyplot as plt
import numpy as np
import gym
from copy import deepcopy


from epidemioptim.utils import plot_stats
from epidemioptim.environments.cost_functions import get_cost_function
from epidemioptim.environments.models import get_model
from epidemioptim.environments.gym_envs import get_env
if __name__ == '__main__':

    simulation_horizon = 364
    stochastic = True
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

    env = get_env(env_id='EpidemicDiscrete-v0',
                  cost_function=cost_function,
                  model=model,
                  simulation_horizon=simulation_horizon)


    all_stats = []
    for i_loop in range(10):
        env.reset()

        # STEP 4: Initialize the scripted sequence of actions
        # This is useful only here for simulation, learning algorithm will replace this.
        # * use a action_type corresponding to your env (on/off, sticky_one_week, switch or commit)
        # * descriptor can be 'never', 'always' or of the form 'open_N1_{day,week,month}_every_N2_{day,week,month}'
        # with N1 and N2 integers. It can also be a list of days alternatively on and off [10, 20, 30]
        # corresponds to 10 days with lockdown on, then 20 off, then 30 on, then on until the end.
        actions = np.zeros(53)

        # STEP 5: Run the simulation
        t = 0
        r = 0
        done = False
        while not done:
            out = env.step(int(actions[t]))
            t += 1
            r += out[1]
            done = out[2]
        stats = env.unwrapped.get_data()
        all_stats.append(deepcopy(stats))
        print(r)

    ax1 = None
    ax2 = None
    fig2 = None

    # STEP 6: Plot results
    for i in range(10):
        stats = all_stats[i]
        # plot model states
        labs = [l + r' $(\times 10^3)$' for l in stats['model_states_labels']]
        labs[2] = 'I'
        ax1, fig1 = plot_stats(t=stats['history']['env_timesteps'],
                               states=np.array(stats['history']['model_states']).transpose(),
                               labels= stats['model_states_labels'],
                               lockdown=np.array(stats['history']['lockdown']),
                               time_jump=stats['time_jump'],
                               icu_capacity=stats['icu_capacity'],
                               axs=ax1)
        plt.savefig('/home/flowers/Desktop/distrib.pdf')

        ax2, fig2 = plot_stats(t=stats['history']['env_timesteps'][1:],
                               states=stats['stats_run']['to_plot'],
                               labels=stats['stats_run']['labels'],
                               title=stats['title'],
                               lockdown=np.array(stats['history']['lockdown']),
                               time_jump=stats['time_jump'],
                               axs=ax2,
                               fig=fig2
                               )
    plt.show()
