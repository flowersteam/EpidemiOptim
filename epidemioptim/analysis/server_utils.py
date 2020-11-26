#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:02:27 2020

@author: ddutartr
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from epidemioptim.utils import *
from epidemioptim.analysis.notebook_utils import setup_for_replay,replot_stats,setup_fig_notebook,run_env,get_action_base
from ipywidgets import HTML,Layout,VBox,FloatSlider,IntSlider,HBox,Label,ToggleButton,Dropdown,Checkbox,interactive_output,Box
# About


# -apple-system,.SFNSText-Regular,San Francisco,Segoe UI,Helvetica Neue,Lucida Grande,
p_style = 'style="line-height:150%;font-weight:300;font-size:22px;font-family:Hind,sans-serif;"'
h3_style = 'style="color:#004c8f;line-height:150%;font-weight:700;font-size:24px;font-family:Montserrat,sans-serif;">'
h2_style = 'style="color:#004c8f;line-height:150%;font-weight:700;font-size:40px;font-family:Montserrat,sans-serif;">'
h2_style_2 = 'style="color:#004c8f;line-height:150%;font-weight:700;font-size:35px;font-family:Montserrat,sans-serif;">'
def introduction():

    intro_html=HTML(layout=Layout(width='800px',
                                  height='100%',
                                  margin='auto',
                                  ),
                    value=(' <link href="https://fonts.googleapis.com/css2?family=Hind:wght@300;400;500;600;700&family=Montserrat:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;1,100;1,200;1,300;1,400;1,500;1,600;1,700&display=swap" rel="stylesheet"> ' +
                            "<font color='black'><font face = 'Comic sans MS'>" +
                           '<center><h2 ' + h2_style + 'EpidemiOptim: A Toolbox for the Optimization of Control Policies in Epidemiological '
                           'Models</h2></center>'
                           +'<h3 ' + h3_style + 'Context</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'Epidemiologists  model  the  dynamics  of  epidemics  in  order  to  propose  control strategies based on pharmaceutical and non-pharmaceutical '
                            'interventions (contact limitation,  lock down,  vaccination,  etc.). '
                           +'Hand-designing such strategies is not trivial because of the number of possible interventions and the difficulty to predict long-term effects.  This task can be cast as an optimization problem where state-of-the-art  machine  learning  algorithms  might  bring  significant  value. '
                           + '</p>'
                           + '<h3 ' + h3_style + 'What we propose</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           + 'The  specificity  of  each  domain — epidemic modelling or solving optimization problem — requires strong collaborations  between  researchers  from  different  fields  of  expertise.'
                           + 'This  is  why  we introduce <var>EpidemiOptim</var>, a Python toolbox that facilitates collaborations between researchers in epidemiology and optimization. '
                           +'EpidemiOptim turns epidemiological models and cost functions into optimization problems via a standard interface commonly used by optimization '
                            'practitioners. This library is presented in details in the <a href="https://arxiv.org/pdf/2010.04452.pdf" style="color:#004c8f;" '
                            'target="_blank">EpidemiOptim paper</a>. '
                           +'</p>'
                           +'<h3 ' + h3_style + 'Interact with trained models, design your own intervention strategy!</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'To demonstrate the use of EpidemiOptim, we run experiments to optimize the design of an on/off lock-down policy in the context of the COVID-19 epidemic in the French region of Ile-de-France. '
                           +'We have two objectives here: minimizing the death toll and minimizing the economic recess. '
                           +'In the tabs below, you will be able to <span style="font-weight:500;">explore strategies optimized by various state-of-the-art optimization algorithms</span>. '
                           +'In the last tab, you will be able to design your own strategy, apply it over a year of epidemic and observe its health and economic consequences.'
                           +'</p>'
                           +'</font>'))
    return intro_html

def footer():
    footer = HTML(layout=Layout(width='800px',
                                    height='100%',
                                    margin='auto',
                                    ),
                      value=(
                                  '<link href="https://fonts.googleapis.com/css2?family=Hind:wght@300;400;500;600;700&family=Montserrat:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;1,100;1,200;1,300;1,400;1,500;1,600;1,700&display=swap" rel="stylesheet"> ' +
                                  "<font color='black'><font face = 'Comic sans MS'>"
                                  + '<h3 ' + h3_style + 'Reference</h3>'
                                  + '<p align="left" ' + p_style + '>'
                                  + 'The bibtex reference to the EpidemiOptim paper can be found here:'
                                  + '<code> @article{colas2020epidemioptim, <br>'
                                  + '&nbsp;&nbsp;&nbsp;&nbsp;title={EpidemiOptim: A Toolbox for the Optimization of Control Policies in Epidemiological<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Models}, <br>'
                                  + '&nbsp;&nbsp;&nbsp;&nbsp;author={Colas, C{\'e}dric and Hejblum, Boris and Rouillon, S{\'e}bastien and Thi{\'e}baut, '
                                    'Rodolphe<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Oudeyer, '
                                    'and Pierre-Yves and Moulin-Frier, Cl{\'e}ment and Prague, M{\'e}lanie} <br>'
                                  + '&nbsp;&nbsp;&nbsp;&nbsp;journal={arXiv preprint arXiv:2010.04452}, <br>'
                                  + '&nbsp;&nbsp;&nbsp;&nbsp;year={2020}} </code>'
                                  + '</p>'
                                  + '</font>'))
    return footer

def algorithm_description(algorithm):
    if algorithm=='DQN':
        str_html=HTML(layout=Layout(width='800px',
                                  height='100%',
                                  margin='auto',
                                  ),
                    value=("<font color='black'><font face = 'Verdana'>" +
                           '<center><h2 ' + h2_style_2 + 'Algo 1: Deep Q-Networks (DQN)</h2></center>'
                           +'<h3 ' + h3_style + 'Objective</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'We want to minimize two metrics: the death toll <var>C<sub>health</sub></var> and the economic recess <var>C<sub>economic</sub></var>, computed over a one-year period.'
                           + '</p>'
                           + '<h3 ' + h3_style + 'The algorithm</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           + 'The first algorithm belongs to the family of <span style="font-weight:500;">reinforcement learning</span> algorithms: <span style="font-weight:500;">Deep Q-Networks ('
                           + 'DQN)</span>. DQN is traditionally used to minimize a unique cost function. To circumvent this problem, we train several control policies, '
                           + 'where each policy minimizes a certain combination of the two costs:</p>'
                           +'<center><var> C = (1- &#946)&#215C<SUB>h</SUB> +  &#946&#215C<SUB>e</SUB></var>,</center>'
                           + '<p align="justify" ' + p_style + '>'
                           +'where <var>C</var> is the aggregated cost and <var>&#946</var> is the mixing parameter.</p>'
                           +'<h3 ' + h3_style + 'What is plotted</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'The four plots below show the evolution of the daily economic and health costs over a one-year period. Red dots indicate lock-down enforcement for the corresponding week. '
                           +'<h3 ' + h3_style + 'Try it yourself!</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'The slider<var> &#946 </var>allows to control the mixing of the two costs. <var>&#946=1</var> results in the pure minimization of the economic cost. &#946<var>=0</var> results in the pure minimization of the death toll.  '
                           + '</p>'
                           +'</font>'))
    elif algorithm=='GOAL_DQN':
        str_html=HTML(layout=Layout(width='800px',
                                  height='100%',
                                  margin='auto',
                                  ),
                    value=("<font color='black'><font size = 5><font face = 'Verdana'>" +
                           '<center><h2 ' + h2_style_2 + 'Algo 2: Goal-Conditioned Deep Q-Networks (Goal-DQN)</h2></center>'
                           +'<h3 ' + h3_style + 'Objective</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'We want to minimize two metrics: the death toll <var>C<sub>health</sub></var> and the economic recess <var>C<sub>economic</sub></var>, computed over a one-year period.'
                           + '</p>'
                           + '<h3 ' + h3_style + 'The algorithm</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'This algorithm is a variant of the traditional <span style="font-weight:500;">Deep Q-Network</span>. In the <span style="font-weight:500;">Goal-Conditioned Q-Networks (Goal-DQN)</span>, we train one policy to minmize all the combinations of the health and economic costs:'
                           +'<center><var> C = (1- &#946)&#215C<SUB>h</SUB> +  &#946&#215C<SUB>e</SUB></var>,</center>'
                           + '<p align="justify" ' + p_style + '>'
                           +'for all values of <var>&#946</var> in <var>[0, 1]</var>.</p>'
                           + '<p align="justify" ' + p_style + '>'
                           +'To do so, the policy receives the value of<var> &#946 </var>corresponding to the mixture of costs it needs to minimize. This dramatically reduces training time compared to a simple DQN, as only one policy is trained (see Algo 1 tab).'
                           + '</p>'
                           +'<h3 ' + h3_style + 'What is plotted</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'The four plots below show the evolution of the daily economic and health costs over a one-year period. Red dots indicate lock-down enforcement for the corresponding week. '
                           +'<h3 ' + h3_style + 'Try it yourself!</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'The slider<var> &#946 </var>allows to control the mixing of the two costs. &#946<var>=1</var> results in the pure minimization of the economic cost. &#946<var>=0</var> results in the pure minimization of the death toll.  '
                           + '</p>'
                           +'</font>'))
    elif algorithm=='GOAL_DQN_CONST':
        str_html=HTML(layout=Layout(width='800px',
                                  height='100%',
                                  margin='auto',
                                  ),
                    value=("<font color='black'><font size = 5><font face = 'Verdana'>" +
                           '<center><h2 ' + h2_style_2 + 'Algo 3: Goal-Conditioned Deep Q-Networks with Constraints (Goal-DQN-C)</h2></center>'
                           +'<h3 ' + h3_style + 'Objective</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'We want to minimize two metrics: the death toll <var>C<sub>health</sub></var> and the economic recess <var>C<sub>economic</sub></var>, computed over a one-year period.'
                           + '</p>'
                           + '<h3 ' + h3_style + 'The algorithm</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'This algorithm is a variant of the traditional <span style="font-weight:500;">Deep Q-Network</span>. In the <span style="font-weight:500;">Goal-Conditioned Q-Networks with Constraints (Goal-DQN-C)</span>, we train one policy to minmize all the combinations of the health and economic costs:'
                           +'<center><var> C = (1- &#946)&#215C<SUB>h</SUB> +  &#946&#215C<SUB>e</SUB></var>,</center>'
                           + '<p align="justify" ' + p_style + '>'
                           +'for all values of <var>&#946</var> in <var>[0, 1]</var>.</p>'
                           + '<p align="justify" ' + p_style + '>'
                           +' In addition, we can set constraints on maximum values for each of the cumulated cost over the one-year period. To do so, the policy receives the value of<var> &#946 </var>corresponding to the mixture of costs it needs to minimize, as well as the value of the maximum cumulative cost that forms its constraints .'
                           + '</p>'
                           +'<h3 ' + h3_style + 'What is plotted</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'The four plots below show the evolution of the daily economic and health costs over a one-year period. Red dots indicate lock-down enforcement for the corresponding week. The black dashed-line represents the constraints on the maximum value of the cost. '
                           +'<h3 ' + h3_style + 'Try it yourself!</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'The slider<var> &#946 </var>allows to control the mixing of the two costs. &#946<var>=1</var> results in the pure minimization of the economic cost. &#946<var>=0</var> results in the pure minimization of the death toll.'
                           +' The other two sliders control the maximum values the cumulative costs can take. Explore the effect of these parameters. Note how the policy adapts to the constraints. If you push further, and set strong constraints on the two costs, a good policy might not exist (e.g. 0 death and 0 euros of economic recess.)'
                           + '</p>'
                           +'</font>'))
    elif algorithm=='NSGA':
        str_html=HTML(layout=Layout(width='800px',
                                  height='100%',
                                  margin='auto',
                                  ),
                    value=("<font color='black'><font size = 5><font face = 'Verdana'>" +
                           '<center><h2 ' + h2_style_2 + 'Algo 4: Non-dominated Sorting Genetic Algorithm II (NSGA-II)</h2></center>'
                           +'<h3 ' + h3_style + 'Objective</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'We want to minimize two metrics: the death toll <var>C<sub>health</sub></var> and the economic recess <var>C<sub>economic</sub></var>, computed over a one-year period.'
                           + '</p>'
                           + '<h3 ' + h3_style + 'The algorithm</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'<span style="font-weight:500;">Non-dominated Sorting Genetic Algorithm II (NSGA-II)</span> is a state-of-the-art multi-objective optimization algorithm from the family of evolutionary algorithms. In contrast to previous algorithms, this one is explicitely built to optimize several costs at a time instead of linear combinations of them. In practice, this algorithm aims to find a '
                           +'<span style="font-weight:500;">Pareto Front</span>, the set of <span style="font-weight:500;">non-dominated solutions</span>: solutions for which one cannot find any other solution that performs better on both dimensions (better health cost <span style="font-weight:500;">and</span> better economic cost). The result of this algorithm is thus a set of control policies, each having their particular trade-off with respect to the two costs.'
                           +'<h3 ' + h3_style + 'What is plotted</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'The first plot represents the Pareto front found by one run of the NSGA-II algorithm. Note that no solution performs better than any other on both dimensions, or worse on both dimensions. Each point represent the average performance of a given policy on the two costs, after it is run on 30 different simulations of the epidemic. The four plots below show the evolution of the daily economic and health costs over a one-year period. Red dots indicate lock-down enforcement for the corresponding week. '
                           +'<h3 ' + h3_style + 'Try it yourself!</h3>'
                           +'<p align="justify" ' + p_style + '>'
                           +'You can click on points in the first plot to select the corresponding policy and see its consequences in terms of the two costs in the graphs below. Note that the model is stochastic. Each time you click on the policy, a new simulation is run, and the way the policy reacts to the epidemic might vary.'
                           + '</p>'
                           +'</font>'))
    elif algorithm=='yourself':
        str_html=HTML(layout=Layout(width='800px',
                                  height='100%',
                                  margin='auto',
                                  ),
                          value=("<font color='black'><font face = 'Verdana'>" +
                                 '<center><h2 ' + h2_style_2 + 'Try It Yourself!</h2></center>'
                                 +'<h3 ' + h3_style + 'Objective</h3>'
                                 +'<p align="justify" ' + p_style + '>'
                                 +'We want to minimize two metrics: the death toll <var>C<sub>health</sub></var> and the economic recess <var>C<sub>economic</sub></var>, computed over a one-year period.'
                                 + '</p>'
                                 + '<h3 ' + h3_style + 'The algorithm</h3>'
                                 +'<p align="justify" ' + p_style + '>'
                                 + 'Here, you are the algorithm!'
                                 +'<h3 ' + h3_style + 'What is plotted</h3>'
                                 +'<p align="justify" ' + p_style + '>'
                                 +'The first plot represents the Pareto front found by one run of the NSGA-II algorithm. The red dot is the average performance of the strategy you design (computed over 30 simulations). The four plots below show the evolution of the daily economic and health costs over a one-year period. Red dots indicate lock-down enforcement for the corresponding week. '
                                 +'<h3 ' + h3_style + 'Try it yourself!</h3>'
                                 +'<p align="justify" ' + p_style + '>'
                                 +'To perform better than NSGA-II, you need to get closer to the origin of the plot <var> (0,0) </var>. Note that algorithms train policy that are reactive to the epidemic and can adapt to its state as it progresses. You are designing, on the other hand, a <span style="font-weight:500;">fixed-strategy</span> that is evaluated on 30 different simulated epidemics.'
                                 + '<br>You can design your strategy with two tools:'
                                 + '<ol ' + p_style +'><li>The four first sliders enable you to define a pattern of the form <span style="font-weight:500;"> </span> implement lock-down N1 weeks every N2 weeks. The first two sliders control the start and end of the pattern (in weeks), the two following sliders control the duration of the lock-down and the period of the pattern respectively.</li>'
                                 +'<li>The checkbox control the enforcement of the lockdown on a weekly basis. Pressing the <span style="font-weight:500;">reset</span> button synchronizes the checkboxes with the pattern defined by the sliders. Checkboxes can then be checked/unchecked to finetune the control strategy.</li></ol>'
                                 + '</p>'
                                 +'</font>'))
            
    else:
        NotImplementedError
    
    return str_html

def slider_setup(slider):
    slider.layout.max_width = '100%'
    #desc=slider.description
    #slider.description=''
    #sliderHbox=HBox([Label(desc,style={'description_width' : 'initial'}),slider])
    return slider
def modify_description(slider):
    desc=slider.description
    slider.description=''
    sliderHbox=HBox([Label(desc,style={'description_width' : 'initial'}),slider])
    return sliderHbox
def update_fig(fig):
    fig.canvas.draw_idle()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig
def canvas_setup(fig):
    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = False
    fig.canvas.layout.min_height = '400px'
    return fig

def deter_checkbox():
    is_deter=Checkbox(
        value=False,
        description='Deterministic model',
        disabled=False,
        indent=False,
        layout={'max_width': '100%'})
    return is_deter

def plot_pareto(algorithm,size,color):
    # Plot pareto front
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sign = 1
    a = sign * algorithm.res_eval['F'][:, 0]
    b = sign * algorithm.res_eval['F'][:, 1]
    nb_points=a.shape[0]
    sc = ax.scatter(a, b, picker=5)
    ax.tick_params(axis='x', labelrotation=30,labelsize=12)
    ax.tick_params(axis='y',labelsize=12)
    colors = [color] * nb_points
    sc.set_color(colors)
    sizes = np.ones(nb_points) * size
    sc.set_sizes(sizes)
    ax.set_xlabel('Total Deaths',fontsize=14)
    ax.set_ylabel('Total GDP Loss (B)',fontsize=14)

    return fig,ax,sc
def normalize(x,data_min,data_max):
    return (x - data_min) / (data_max - data_min)

def center_vbox(children):
    box_layout = Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='100%')
    centered_layout = VBox(children=children, layout = box_layout)
    return centered_layout



def run_env_with_actions(actions,env, reset_same_model):

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

def try_it_ui(checkbox_objects,box_layout):
    number_of_week=52
    number_of_week_per_row=4
    offset_button=5

    weekBox=Box(children=[])
    for i in range(int(number_of_week/number_of_week_per_row)):
        weekBox=VBox([weekBox,Box(checkbox_objects[number_of_week_per_row*(i)+offset_button:
                                                   number_of_week_per_row*(i+1)+offset_button])])
    setBox=Box([checkbox_objects[4]],layout=Layout(display='flex',
                                                   flex_flow='column',
                                                   align_items='center',
                                                   width='100%'))
    ui=Box(children=[HBox(checkbox_objects[0:2]),
                      HBox(checkbox_objects[2:4]),
                      setBox,
                      weekBox]
            ,layout=box_layout)
    return ui
def test_layout(algorithm_str,seed,deterministic_model):
    def update_algo_deter(change):
        deterministic_model=change.new
        algorithm, cost_function, env, params = setup_for_replay(folder+to_add , seed, deterministic_model)
        return algorithm, cost_function, env, params
    if seed is None:
        seed = np.random.randint(1e6)
    if algorithm_str == 'DQN':
        to_add = '0.5/'
        folder = get_repo_path() + "/data/data_for_visualization/DQN/"
    elif algorithm_str=='yourself':
        folder = get_repo_path() + "/data/data_for_visualization/NSGA/1/"
        to_add = ''
    else:
        to_add = ''
        folder = get_repo_path() + "/data/data_for_visualization/"+ algorithm_str+ "/1/"
    algorithm, cost_function, env, params = setup_for_replay(folder+to_add , seed, deterministic_model)

    if algorithm_str == 'DQN':
        is_deter=deter_checkbox()
        str_html=algorithm_description(algorithm_str)
        stats, msg = run_env(algorithm, env, first=True)
        fig, lines, plots_i, high, axs = setup_fig_notebook(stats)
        slider = FloatSlider(orientation='horizontal',description='beta:',value=0.5,
                             min=0,
                             max=1,
                             step=0.05,layout={'width': '450px'}
                             )

        slider=slider_setup(slider)
        fig=canvas_setup(fig)
        def update_lines(change):
            beta=slider.value
            deterministic_model=is_deter.value
            algorithm, cost_function, env, params = setup_for_replay(folder + str(beta) + '/', seed, deterministic_model)
            stats, msg = run_env(algorithm, env, goal=np.array([beta]))
            replot_stats(lines, stats, plots_i, cost_function, high)
            update_fig(fig)
        slider.observe(update_lines, names='value')
        is_deter.observe(update_lines,names='value')
        final_layout = center_vbox([str_html,is_deter,slider,fig.canvas])
        return final_layout
    elif algorithm_str == 'NSGA':
        is_deter=deter_checkbox()
        str_html=algorithm_description(algorithm_str)
        stats, msg = run_env(algorithm, env)
        fig1, lines, plots_i, high, axs = setup_fig_notebook(stats)
        size = 15
        color = "#004ab3"
        color_highlight = "#b30000"

        fig,ax,sc=plot_pareto(algorithm,size,color)
        data = sc.get_offsets().data
        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)
        nb_points = data.shape[0]
        
        def normalize(x):
            return (x - data_min) / (data_max - data_min)

        normalized_data = normalize(data)
        def onclick2(event):
            x = event.xdata
            y = event.ydata

            # find closest in dataset
            point = np.array([x, y])
            normalized_point = normalize(point)
            dists = np.sqrt(np.sum((normalized_point - normalized_data) ** 2, axis=1))
            closest_ind = np.argmin(dists)

            # highlight it
            order = np.concatenate([np.arange(closest_ind), np.arange(closest_ind + 1, nb_points), np.array([closest_ind])])
            sc.set_offsets(data[order])
            sizes = np.ones(nb_points) * size
            sizes[-1] = size * 5
            colors = [color] * nb_points
            colors[-1] = color_highlight
            sc.set_sizes(sizes)  # you can set you markers to different sizes
            sc.set_color(colors)
            # rerun env
            weights = algorithm.res_eval['X'][closest_ind]
            algorithm.policy.set_params(weights)
            stats, msg = run_env(algorithm, env)
            replot_stats(lines, stats, plots_i, cost_function, high)
            print(env.model.stochastic)
            # refresh figure
            update_fig(fig1)
            update_fig(fig)
        def update_deter(change):
            deterministic_model=change.new
            env.model.stochastic = not deterministic_model
            env.model.define_params_and_initial_state_distributions()
        is_deter.observe(update_deter,names='value')
        cid = fig.canvas.mpl_connect('button_press_event', onclick2)
        fig=canvas_setup(fig)
        fig1=canvas_setup(fig1)
        final_layout = center_vbox([str_html,is_deter,fig.canvas, fig1.canvas])
        return(final_layout)
    elif 'GOAL_DQN' in algorithm_str:
        if cost_function.use_constraints:
            goal = np.array([0.5, 1, 1])
        else:
            goal = np.array([0.5])
        str_html=algorithm_description(algorithm_str)
        
        stats, msg = run_env(algorithm, env, goal, first=True)
        fig, lines, plots_i, high, axs = setup_fig_notebook(stats)
        if cost_function.use_constraints:
            # Plot constraints as dotted line.
            style={'description_width': '150px'}
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
            #Define slider
            slider_beta = FloatSlider(orientation='horizontal',
                                            description='beta',
                                            style = style,
                                            value=0.5,
                                            min=0,
                                            max=1,
                                            step=0.05,
                                            layout={'width': '450px'}
                                            )   
            slider_M_sanitary = IntSlider(orientation='horizontal',
                                            description='Sanitary constraint',
                                            style = style,
                                            value=62000,
                                            min=1000,
                                            max=62000,
                                            step=5000,
                                            layout={'width': '450px'}
                                            )   
            slider_M_economic = IntSlider(orientation='horizontal',
                                            description='Economic constraint',
                                            style = style,
                                            value=160,
                                            min=20,
                                            max=160,
                                            step=20,
                                            layout={'width': '450px'}
                                            )
            slider_beta=slider_setup(slider_beta)
            slider_M_sanitary=slider_setup(slider_M_sanitary)
            slider_M_economic=slider_setup(slider_M_economic)
            fig=canvas_setup(fig)
            is_deter=deter_checkbox()
            is_deter.style=style
            is_deter.layout.width='200px'
            def update_const(change):
                # normalize constraints
                M_sanitary=slider_M_sanitary.value
                M_economic=slider_M_economic.value
                beta=slider_beta.value
                deterministic_model=is_deter.value
                algorithm, cost_function, env, params = setup_for_replay(folder + to_add, seed, deterministic_model)
                c_sanitary = cost_function.costs[0].compute_normalized_constraint(M_sanitary)
                c_economic = cost_function.costs[1].compute_normalized_constraint(M_economic)
                stats, msg = run_env(algorithm, env, goal=np.array([beta, c_sanitary, c_economic]))
                replot_stats(lines, stats, plots_i, cost_function, high, constraints=[c_sanitary, c_economic])
                update_fig(fig)
            slider_beta.observe(update_const, 'value')
            slider_M_sanitary.observe(update_const, 'value')
            slider_M_economic.observe(update_const, 'value')
            is_deter.observe(update_const,names='value')
            final_layout = center_vbox([str_html,
                                        center_vbox([is_deter,slider_beta,slider_M_sanitary,slider_M_economic]),
                                        fig.canvas])
            return final_layout
        else :
            is_deter=deter_checkbox()
            slider_goal = FloatSlider(orientation='horizontal',
                                      description='beta:',
                                      value=0.5,
                                      min=0,
                                      max=1,
                                      step=0.05,
                                      layout={'width': '450px'}
                                      )
            slider_goal=slider_setup(slider_goal)
            fig=canvas_setup(fig)
            def update_goal(change):
                beta=slider_goal.value
                deterministic_model=is_deter.value
                algorithm, cost_function, env, params = setup_for_replay(folder + to_add, seed, deterministic_model)
                stats, msg = run_env(algorithm, env, goal=np.array([beta]))
                replot_stats(lines, stats, plots_i, cost_function, high)
                update_fig(fig)
            slider_goal.observe(update_goal, names='value')
            is_deter.observe(update_goal,names='value')
            final_layout = center_vbox([str_html,is_deter,slider_goal,fig.canvas])
            return final_layout
    elif algorithm_str == 'yourself':
        style={'description_width': '250px', 'widget_width': '50%'}
        run_eval = False 
        n_evals = 10  # number of evaluation rolloutsseed = None  # None picks a random seed
        str_html=algorithm_description(algorithm_str)
        global actions
        actions = get_action_base('never')
        stats, costs = run_env_with_actions(actions,env, reset_same_model=False)
        fig1, lines, plots_i, high, axs = setup_fig_notebook(stats)
        size = 15
        color = "#004ab3"
        color_highlight = "#b30000"
        
        fig,ax,sc=plot_pareto(algorithm,size,color)
        data = sc.get_offsets().data
        off_sets = sc.get_offsets()
        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)
        nb_points = data.shape[0]
        set_button = ToggleButton(value=True,
                                      description='Set to pattern',
                                      disabled=False,
                                      button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                      layout=Layout(width='50%', height='80px'),
                                      style=style,
                                      tooltip='Description',
                                      icon='check'  # (FontAwesome names without the `fa-` prefix)
                                      )
        start = Dropdown(options=[str(i) for i in range(1, 54)],
                                 value='1',
                                 description="# weeks before pattern starts",
                                 layout=Layout(width='50%', height='80px'),
                                 style=style)

        stop = Dropdown(options=[str(i) for i in range(1, 55)],
                                 value='54',
                                 description="# weeks before pattern stops",
                                 layout=Layout(width='50%', height='80px'),
                                 style=style)

        nb_weeks = Dropdown(options=[str(i) for i in range(0, 54)],
                                    value='0',
                                    description="Duration of lockdown phase (weeks)",
                                    layout=Layout(width='50%', height='80px'),
                                    style=style)

        every = Dropdown(options=[str(i) for i in range(1, 54)],
                                 value='1',
                                 description="Duration of the cycle or period (weeks)",
                                 layout=Layout(width='50%', height='80px'),
                                 style=style)
        names = ['start','stop','nb_weeks','every','set_button']
        checkbox_objects = [start,stop,nb_weeks,every,set_button]
        for i in range(52):
            desc='Week {}'.format(i + 1)
            checkbox_objects.append(Checkbox(value=False, description=desc))
            names.append(desc)
        arg_dict = {names[i]: checkbox for i, checkbox in enumerate(checkbox_objects)}

        box_layout = Layout(overflow_y='auto',
                    border='3px solid black',
                    height='450px',
                    display='block',width='800px')
        ui = Box(children=checkbox_objects, layout=box_layout)
        ui=try_it_ui(checkbox_objects,box_layout)
        def update_try(**kwargs):
            start=int(kwargs['start'])-1
            stop=int(kwargs['stop'])-1
            nb_weeks=int(kwargs['nb_weeks'])
            every=int(kwargs['every'])
            action_str = str(nb_weeks) + '_' + str(every)
            print(action_str)
            set_button=kwargs['set_button']
            if set_button:
                print('Set to pattern. Closing {} weeks every {} weeks.'.format(nb_weeks, every))
            else:
                print('Custom strategy.')

                if every < nb_weeks:
                    print('When "every" is superior or equal to "nb_weeks", lockdown is always on.')
            actions = get_action_base(action_str, start, stop)
            if set_button:
                for i in range(52):
                    checkbox_objects[5+i].value = bool(actions[i])
        
            else:
                for i in range(52):
                    actions[i] = int(kwargs['Week {}'.format(i+1)])
            stats, costs = run_env_with_actions(actions,env, reset_same_model=deterministic_model)
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

            update_fig(fig)
            update_fig(fig1)
            return actions
        out = interactive_output(update_try, arg_dict)
        fig=canvas_setup(fig)
        fig1=canvas_setup(fig1)
        final_layout = center_vbox([str_html,ui,fig.canvas, fig1.canvas])
        return final_layout
    else:
        raise NotImplementedError
