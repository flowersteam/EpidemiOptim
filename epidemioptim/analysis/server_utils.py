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
from epidemioptim.analysis.notebook_utils import setup_for_replay,replot_stats,setup_fig_notebook,run_env
from ipywidgets import HTML,Layout,VBox,FloatSlider,HBox,Label
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

def test_layout(algorithm_str,seed,deterministic_model):
    if seed is None:
        seed = np.random.randint(1e6)
    if algorithm_str == 'DQN':
        to_add = '0.5/'
        folder = get_repo_path() + "/data/data_for_visualization/DQN/"
    else:
        to_add = ''
        folder = get_repo_path() + "/data/data_for_visualization/"+ algorithm_str+ "/1/"
    algorithm, cost_function, env, params = setup_for_replay(folder+to_add , seed, deterministic_model)

    if algorithm_str == 'DQN':
        str_html=algorithm_description(algorithm_str)
        stats, msg = run_env(algorithm, env, first=True)
        fig, lines, plots_i, high, axs = setup_fig_notebook(stats)
        slider = FloatSlider(orientation='horizontal',description='beta:',value=0.5,
                             min=0,
                             max=1,
                             step=0.05
                             )

        slider=slider_setup(slider)
        fig=canvas_setup(fig)
        def update_lines(change):
            beta=change.new
            algorithm, cost_function, env, params = setup_for_replay(folder + str(beta) + '/', seed, deterministic_model)
            stats, msg = run_env(algorithm, env, goal=np.array([beta]))
            replot_stats(lines, stats, plots_i, cost_function, high)
            update_fig(fig)
        slider.observe(update_lines, names='value')

        final_layout = center_vbox([str_html,fig.canvas, slider])
        #final_layout = center_vbox([fig.canvas, slider])
        #final_layout = VBox([fig.canvas, slider])
        return final_layout
    elif algorithm_str == 'NSGA':
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

            #
            # sizes = np.ones(nb_points) * size
            # sizes[closest_ind] = size * 3
            # colors = [color] * nb_points
            # colors[closest_ind] = color_highlight
            # sc.set_sizes(sizes)  # you can set you markers to different sizes
            # sc.set_color(colors)

            # rerun env
            weights = algorithm.res_eval['X'][closest_ind]
            algorithm.policy.set_params(weights)
            stats, msg = run_env(algorithm, env)
            replot_stats(lines, stats, plots_i, cost_function, high)

            # refresh figure
            update_fig(fig1)
            update_fig(fig)
        cid = fig.canvas.mpl_connect('button_press_event', onclick2)
        fig=canvas_setup(fig)
        fig1=canvas_setup(fig1)
        final_layout = center_vbox([str_html,fig.canvas, fig1.canvas])
        #final_layout = center_vbox([fig.canvas, fig1.canvas])
        #final_layout = VBox([fig.canvas, fig1.canvas])
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
                                            description='beta:',
                                            value=0.5,
                                            min=0,
                                            max=1,
                                            step=0.05
                                            )   
            slider_M_sanitary = FloatSlider(orientation='horizontal',
                                            description='Sanitary constraint:',
                                            value=62000,
                                            min=1000,
                                            max=62000,
                                            step=5000
                                            )   
            slider_M_economic = FloatSlider(orientation='horizontal',
                                            description='Economic constraint:',
                                            value=160,
                                            min=20,
                                            max=160,
                                            step=20
                                            )
            slider_beta=slider_setup(slider_beta)
            slider_M_sanitary=slider_setup(slider_M_sanitary)
            slider_M_economic=slider_setup(slider_M_economic)
            fig=canvas_setup(fig)
            def update_const(change):
                # normalize constraints
                M_sanitary=slider_M_sanitary.value
                M_economic=slider_M_economic.value
                beta=slider_beta.value
                c_sanitary = cost_function.costs[0].compute_normalized_constraint(M_sanitary)
                c_economic = cost_function.costs[1].compute_normalized_constraint(M_economic)
                stats, msg = run_env(algorithm, env, goal=np.array([beta, c_sanitary, c_economic]))
                replot_stats(lines, stats, plots_i, cost_function, high, constraints=[c_sanitary, c_economic])
                update_fig(fig)
            slider_beta.observe(update_const, 'value')
            slider_M_sanitary.observe(update_const, 'value')
            slider_M_economic.observe(update_const, 'value')
            slider_M_economic_desc=modify_description(slider_M_economic)
            slider_M_sanitary_desc=modify_description(slider_M_sanitary)

            final_layout = center_vbox([str_html, 
                                        fig.canvas,
                                        center_vbox([slider_beta,
                                              HBox([slider_M_sanitary_desc,slider_M_economic_desc])])
                                        ])
            #final_layout = center_vbox([fig.canvas, slider_beta,slider_M_sanitary,slider_M_economic])
            #final_layout = VBox([fig.canvas, slider_beta,slider_M_sanitary,slider_M_economic])
            return final_layout
        else :
            slider_goal = FloatSlider(orientation='horizontal',
                                      description='beta:',
                                      value=0.5,
                                      min=0,
                                      max=1,
                                      step=0.05
                                      )
            slider_goal=slider_setup(slider_goal)
            fig=canvas_setup(fig)
            def update_goal(change):
                beta=change.new
                stats, msg = run_env(algorithm, env, goal=np.array([beta]))
                replot_stats(lines, stats, plots_i, cost_function, high)
                update_fig(fig)
            slider_goal.observe(update_goal, names='value')
            final_layout = center_vbox([str_html,fig.canvas, slider_goal])
            #final_layout = center_vbox([fig.canvas, slider_goal])
            #final_layout = VBox([fig.canvas, slider_goal])
            return final_layout
                
    else:
        raise NotImplementedError
