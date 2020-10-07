import numpy as np


def run_rollout(policy, env, eval, n=1, additional_keys=(), goal=None, reset_same_model=False):
    """
    Rollout function. Executes 'n' trajectories in 'env' using 'policy' with given 'goal'.

    Parameters
    ----------
    policy: BaseAlgorithm
        The algorithm whose policy is used as the agent.
    env: BaseEnv
        The learning environment.
    eval: bool
        Whether this is an evaluation episode (no exploration noise)
    n: int
        Number of episodes of interaction, default 1.
    additional_keys: tuple of strings, optional
        Additional elements from the info dict that should be collected and save in the episode dict, default ().
    goal: 2D nd.array, optional
        Goal vectors of dim (n, dim_goal) that the agent should target for each interaction.

    Returns
    -------
    list of dict
        List of 'n' episode summaries represented by dictionnaries containing informative metrics (states, actions, goals, etc).
    """

    if goal is not None:
        goal = np.atleast_2d(goal)
        assert goal.shape[0] == n
    else:
        goal = [None] * n

    episodes = []
    for i in range(n):
        # Setup saved values
        episode = dict(zip(additional_keys, [[] for _ in range(len(additional_keys))] ))
        env_states = []
        aggregated_costs = []
        actions = []
        dones = []
        if reset_same_model:
            env.reset_same_model()
        state = env.reset()
        env_states.append(state)

        # Parameterize the cost function by the goal
        if goal[i] is not None:
            env.unwrapped._set_rew_params(goal[i])

        done = False
        t = 0
        q_constraints_list = []
        while not done:

            # Augment the state by the goal if there is one.
            if goal[i] is not None:
                augmented_state = np.concatenate([state, goal[i]]).copy()
            else:
                augmented_state = state.copy()

            # Interact
            action, q_constraints = policy.act(augmented_state, deterministic=eval)
            next_state, agg_cost, done, info = env.step(action)

            # Save stuff
            state = next_state
            t = env.unwrapped.t
            q_constraints_list.append(q_constraints)
            actions.append(action.flatten())
            aggregated_costs.append(agg_cost)
            env_states.append(state)
            dones.append(done)

            for k in additional_keys:
                episode[k].append(info[k])

        # Form episode dict
        episode.update(env_states=np.array(env_states),
                       aggregated_costs=np.array(aggregated_costs),
                       actions=np.array(actions),
                       goal=goal[i],
                       eval=eval,
                       dones=np.array(dones))
        episodes.append(episode)
    return episodes