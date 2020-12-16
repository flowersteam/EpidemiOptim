import json
import numpy as np
import matplotlib.pyplot as plt
from epidemioptim.utils import set_seeds, get_repo_path
from ipywidgets import *

# # # # # # # # # # # # # # # # # # # # # # # #
# Notebook utils
# # # # # # # # # # # # # # # # # # # # # # # #

def setup_diy(seed, run_eval, n_evals, deterministic_model):

    if run_eval:
        deterministic_model = False

    algorithm, cost_function, env, params = setup_for_replay(get_repo_path() + "/data/data_for_visualization/NSGA/1/", seed, deterministic_model)

    def run_env_with_actions(actions, reset_same_model):

        additional_keys = ('costs', 'constraints')
        # Setup saved values
        episode = dict(zip(additional_keys, [[] for _ in range(len(additional_keys))] ))
        env_states = []
        aggregated_costs = []
        dones = []
        if reset_same_model:
            env.reset_same_model()
        state = env.reset()
        env_states.append(state)

        done = False
        t = 0
        counter = 0
        while not done:
            # Interact
            next_state, agg_cost, done, info = env.step(actions[counter])

            # Save stuff
            state = next_state
            t = env.unwrapped.t
            counter += 1
            aggregated_costs.append(agg_cost)
            env_states.append(state)
            dones.append(done)

            for k in additional_keys:
                episode[k].append(info[k])

        # Form episode dict
        episode.update(env_states=np.array(env_states),
                       aggregated_costs=np.array(aggregated_costs),
                       actions=np.array(actions),
                       dones=np.array(dones))

        aggregated_costs = np.sum(episode['aggregated_costs'])
        costs = np.sum(episode['costs'], axis=0)
        stats = env.unwrapped.get_data()

        return stats, costs
    global actions
    actions = get_action_base('never')
    stats, costs = run_env_with_actions(actions, reset_same_model=False)
    fig, lines, plots_i, high, axs = setup_fig_notebook(stats)

    #NSGA:

    # Plot pareto front
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sign = 1
    a = sign * algorithm.res_eval['F'][:, 0]
    b = sign * algorithm.res_eval['F'][:, 1]
    sc = ax.scatter(a, b, picker=5)
    data = sc.get_offsets().data
    off_sets = sc.get_offsets()
    nb_points = data.shape[0]

    size = 30
    color = "#004ab3"
    color_highlight = "#b30000"
    old_colors = [color] * nb_points
    sc.set_color(old_colors)
    sizes = np.ones(nb_points) * size
    sc.set_sizes(sizes)
    text = ax.text(0, 0, "", va="bottom", ha="left")



    checkboxes = [widgets.Checkbox(value=False,
                                   description='Week {}'.format(i + 1),
                                   disabled=False,
                                   indent=False) for i in range(53)]
    button_widgets_ = dict(zip(['Week {}'.format(i) for i in range(53)],
                              checkboxes))

    start = widgets.Dropdown(options=[str(i) for i in range(1, 54)],
                             value='1',
                             description="# weeks before pattern starts",
                             layout=Layout(width='50%', height='80px'),
                             style={'description_width': 'initial', 'widget_width': '50%'})

    stop = widgets.Dropdown(options=[str(i) for i in range(1, 55)],
                             value='54',
                             description="# weeks before pattern stops",
                             layout=Layout(width='50%', height='80px'),
                             style={'description_width': 'initial', 'widget_width': '50%'})

    nb_weeks = widgets.Dropdown(options=[str(i) for i in range(0, 54)],
                            value='0',
                            description="Duration of lockdown phase (weeks)",
                            layout=Layout(width='50%', height='80px'),
                            style={'description_width': 'initial', 'widget_width': '50%'})

    every = widgets.Dropdown(options=[str(i) for i in range(1, 54)],
                                value='1',
                                description="Duration of the cycle or period (weeks)",
                                layout=Layout(width='50%', height='80px'),
                                style={'description_width': 'initial', 'widget_width': '50%'})

    set_button = widgets.ToggleButton(value=True,
                                      description='Set to pattern',
                                      disabled=False,
                                      button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                      layout=Layout(width='50%', height='80px'),
                                      style={'description_width': 'initial'},
                                      tooltip='Description',
                                      icon='check'  # (FontAwesome names without the `fa-` prefix)
                                      )

    @interact(start=start, stop=stop, nb_weeks=nb_weeks, every=every, set_button=set_button, **button_widgets_)
    def update(start, stop, nb_weeks, every, set_button, **button_widgets):

        start = int(start) - 1
        stop = int(stop) - 1
        nb_weeks = int(nb_weeks)
        every = int(every)
        action_str = str(nb_weeks) + '_' + str(every)
        if set_button:
            print('Set to pattern. Closing {} weeks every {} weeks.'.format(nb_weeks, every))
        else:
            print('Custom strategy.')

            if every < nb_weeks:
                print('When "every" is superior or equal to "nb_weeks", lockdown is always on.')
        actions = get_action_base(action_str, start, stop)

        if set_button:
            for i in range(53):
                button_widgets_['Week {}'.format(i)].value = bool(actions[i])
        else:
            for i in range(53):
                actions[i] = int(button_widgets['Week {}'.format(i)])

        stats, costs = run_env_with_actions(actions, reset_same_model=deterministic_model)

        if run_eval:
            all_costs = [run_env_with_actions(actions, reset_same_model=False)[1] for _ in range(n_evals)]
            all_costs = np.array(all_costs)
            print(all_costs)
            means = all_costs.mean(axis=0)
            x, y = means
            stds = all_costs.std(axis=0)
            msg = '\nEvaluation (over {} seeds):'.format(n_evals)
            msg += '\n\t Death toll: {} +/- {}'.format(int(means[0]), int(stds[0]))
            msg += '\n\t Economic cost: {:.2f} +/- {:.2f} B.'.format(int(means[1]), int(stds[1]))
            print(msg)
        else:
            x, y = costs
        print('\nDeath toll: {}, Economic cost: {:.2f} B.'.format(int(costs[0]), costs[1]))
        replot_stats(lines, stats, plots_i, cost_function, high)

        # update PAreto:
        new_offsets = np.concatenate([off_sets, np.array([[x, y]])], axis=0)
        sc.set_offsets(new_offsets)
        new_colors = [color] * nb_points + [color_highlight]
        sc.set_color(new_colors)
        new_sizes = [size] * nb_points + [size * 2]
        sc.set_sizes(new_sizes)

        fig.canvas.draw_idle()

    return actions




def setup_visualization(folder, algorithm_str, seed, deterministic_model):
    if seed is None:
        seed = np.random.randint(1e6)
    if algorithm_str == 'DQN':
        to_add = '0.5/'
    else:
        to_add = ''
    algorithm, cost_function, env, params = setup_for_replay(folder + to_add, seed, deterministic_model)
    if algorithm_str == 'NSGA':
        stats, msg = run_env(algorithm, env)
        fig, lines, plots_i, high, axs = setup_fig_notebook(stats)

        # Plot pareto front
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sign = 1
        a = sign * algorithm.res_eval['F'][:, 0]
        b = sign * algorithm.res_eval['F'][:, 1]
        sc = ax.scatter(a, b, picker=5)
        data = sc.get_offsets().data
        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)
        nb_points = data.shape[0]

        def normalize(x):
            return (x - data_min) / (data_max - data_min)

        normalized_data = normalize(data)
        size = 30
        color = "#004ab3"
        color_highlight = "#b30000"
        colors = [color] * nb_points
        sc.set_color(colors)
        sizes = np.ones(nb_points) * size
        sc.set_sizes(sizes)
        text = ax.text(0, 0, "", va="bottom", ha="left")

        def onclick(event):
            x = event.xdata
            y = event.ydata

            # find closest in dataset
            point = np.array([x, y])
            normalized_point = normalize(point)
            dists = np.sqrt(np.sum((normalized_point - normalized_data) ** 2, axis=1))
            closest_ind = np.argmin(dists)

            # highlight it
            sizes = np.ones(nb_points) * size
            sizes[closest_ind] = size * 3
            colors = [color] * nb_points
            colors[closest_ind] = color_highlight
            sc.set_sizes(sizes)  # you can set you markers to different sizes
            sc.set_color(colors)

            # rerun env
            weights = algorithm.res_eval['X'][closest_ind]
            algorithm.policy.set_params(weights)
            stats, msg = run_env(algorithm, env)
            print(msg)
            replot_stats(lines, stats, plots_i, cost_function, high)

            # refresh figure
            fig1.canvas.draw_idle()
            tx = 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata)
            text.set_text(tx)

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    elif 'GOAL_DQN' in algorithm_str:
        if cost_function.use_constraints:
            goal = np.array([0.5, 1, 1])
        else:
            goal = np.array([0.5])

        stats, msg = run_env(algorithm, env, goal, first=True)
        fig, lines, plots_i, high, axs = setup_fig_notebook(stats)

        if cost_function.use_constraints:
            # Plot constraints as dotted line.
            M_sanitary = cost_function.costs[0].compute_constraint(1)
            line, = axs[1].plot([0, params['simulation_horizon']],
                                [M_sanitary, M_sanitary],
                                c='k',
                                linestyle='--')
            lines.append(line)
            M_economic = cost_function.costs[1].compute_constraint(1)
            line, = axs[3].plot([0, params['simulation_horizon']],
                                [M_economic, M_economic],
                                c='k',
                                linestyle='--')
            lines.append(line)

            # Define the update function for model 2
            def update(beta=widgets.FloatSlider(min=0, max=1, step=0.05, value=0.5),
                       M_sanitary=widgets.FloatSlider(min=1000, max=62000, step=5000, value=62000),
                       M_economic=widgets.FloatSlider(min=20, max=160, step=20, value=160)):
                # normalize constraints
                c_sanitary = cost_function.costs[0].compute_normalized_constraint(M_sanitary)
                c_economic = cost_function.costs[1].compute_normalized_constraint(M_economic)
                print(c_sanitary, c_economic)
                stats, msg = run_env(algorithm, env, goal=np.array([beta, c_sanitary, c_economic]))
                replot_stats(lines, stats, plots_i, cost_function, high, constraints=[c_sanitary, c_economic])
                fig.canvas.draw_idle()
                print(msg)
        else:
            # Define the update function for model 1
            def update(beta=widgets.FloatSlider(min=0, max=1, step=0.05, value=0.5)):
                stats, msg = run_env(algorithm, env, goal=np.array([beta]))
                print(msg)
                msg = ''
                replot_stats(lines, stats, plots_i, cost_function, high)

                fig.canvas.draw_idle()
        interact(update);
    elif algorithm_str == 'DQN':
        stats, msg = run_env(algorithm, env, first=True)
        fig, lines, plots_i, high, axs = setup_fig_notebook(stats)

        # Define the update function
        def update(beta=widgets.FloatSlider(min=0, max=1, step=0.05, value=0.5)):
            print('Load a new DQN model for beta = {}'.format(beta))
            # Load a new DQN model for each new beta
            algorithm, cost_function, env, params = setup_for_replay(folder + str(beta) + '/', seed, deterministic_model)
            stats, msg = run_env(algorithm, env, goal=np.array([beta]))
            print(msg)
            msg = ''
            replot_stats(lines, stats, plots_i, cost_function, high)
            fig.canvas.draw_idle()
        interact(update);

    else:
        raise NotImplementedError



def setup_for_replay(folder, seed=np.random.randint(1e6), deterministic_model=False):
    from epidemioptim.environments.models import get_model
    from epidemioptim.environments.cost_functions import get_cost_function
    from epidemioptim.environments.gym_envs import get_env
    from epidemioptim.optimization import get_algorithm

    # print('Replaying: ', folder)
    with open(folder + 'params.json', 'r') as f:
        params = json.load(f)

    if deterministic_model:
        params['model_params']['stochastic'] = False
    params['logdir'] = None#get_repo_path() + 'data/results/experiments' + params['logdir'].split('EpidemicDiscrete-v0')[1]
    model = get_model(model_id=params['model_id'],
                        params=params['model_params'])

    # update reward params
    params['cost_params']['N_region'] = int(model.pop_sizes[params['model_params']['region']])
    params['cost_params']['N_country'] = int(np.sum(list(model.pop_sizes.values())))

    # set_seeds(seed)

    cost_function = get_cost_function(cost_function_id=params['cost_id'],
                                      params=params['cost_params'])

    # Form the Gym-like environment
    env = get_env(env_id=params['env_id'],
                  cost_function=cost_function,
                  model=model,
                  simulation_horizon=params['simulation_horizon'],
                  seed=seed)

    # Get DQN algorithm parameterized by beta
    algorithm = get_algorithm(algo_id=params['algo_id'],
                              env=env,
                              params=params)


    if params['algo_id'] == 'NSGAII':
        algorithm.load_model(folder + 'res_eval.pk')
    else:
        algorithm.load_model(folder + 'models/best_model.cp')

    return algorithm, cost_function, env, params

def replot_stats(lines, stats, plots_i, cost_function, high, constraints=None):
    states = stats['stats_run']['to_plot']
    lockdown = np.array(stats['history']['lockdown'])
    inds_lockdown = np.argwhere(lockdown == 1).flatten()
    for i in range(len(plots_i)):
        ind = plots_i[i]
        lines[i].set_ydata(states[ind])
        lines[i + 4].set_offsets(np.array([inds_lockdown, np.ones([inds_lockdown.size]) * (high[i] * 0.98)]).transpose())
    if constraints:
        c_death = cost_function.costs[0].compute_constraint(constraints[0])
        lines[-2].set_ydata([c_death, c_death])
        c_eco = cost_function.costs[1].compute_constraint(constraints[1])
        lines[-1].set_ydata([c_eco, c_eco])
        #print(c_death, c_eco)

def run_env(algorithm, env, goal=None, first=False):
    res, costs = algorithm.evaluate(n=1, goal=goal, reset_same_model= not first)
    msg = '----------------\n'
    for k in res.keys():
        msg += str(k) + ': {:.2f}\n'.format(res[k])
    stats = env.unwrapped.get_data()
    return stats, msg

def setup_fig_notebook(stats):
    labels = stats['stats_run']['labels']
    t = stats['history']['env_timesteps'][1:]
    states = stats['stats_run']['to_plot']
    lockdown = np.array(stats['history']['lockdown'])
    legends = stats['stats_run']['legends']
    time_step_size = 1
    inds_lockdown = np.argwhere(lockdown == 1).flatten() * time_step_size
    high = [3100, 67000, 1, 180]

    fig1, axs = plt.subplots(2, 2, figsize=(9, 7))
    axs = axs.ravel()
    lines = []
    plots_i = [0, 1, 3, 4]
    for i in range(len(plots_i)):
        ind = plots_i[i]
        axs[i].set_ylim([0, high[i]])
        axs[i].set_xlim([0, 365])
        axs[i].set_xlabel('days')
        axs[i].set_ylabel(labels[ind])
        if isinstance(states[ind], list):
            line, axs[i].plot(t, np.array(states[ind]).transpose())
            if legends is not None:
                if legends[ind] is not None:
                    axs[i].legend(legends[ind])
        else:
            line, = axs[i].plot(t, states[ind])
        # line, = axs[i].plot(t, states[:, i])
        lines.append(line)
    for i in range(len(plots_i)):
        line = axs[i].scatter(inds_lockdown, np.ones([inds_lockdown.size]) * (high[i] * 0.9),
                              s=10,
                              c='r')
        lines.append(line)
    return fig1, lines, plots_i, high, axs

def get_action_base(action_str, start=None, stop=None, duration=53):
    actions = np.zeros([duration])
    if action_str == 'always':
        actions = np.ones([duration])
        if isinstance(start, int):
            actions[:start] = 0
        if isinstance(stop, int):
            actions[stop:] = 0
    elif action_str == 'never':
        actions = np.zeros([duration])
    else:
        splitted_act_str = action_str.split('_')
        nb_weeks = int(splitted_act_str[0])
        every = int(splitted_act_str[-1])
        start_week = 0
        stop_week = 53
        if isinstance(start, int):
            start_week = start
        if isinstance(stop, int):
            stop_week = stop
        counter = start_week
        on = True
        while counter < stop_week:
            if on:
                actions[counter: min(counter + nb_weeks, stop_week)] = 1
                counter += nb_weeks
                on = False
            else:
                on = True
                counter += max(every - nb_weeks, 0)
    return actions