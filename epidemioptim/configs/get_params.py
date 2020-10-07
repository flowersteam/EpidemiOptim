
def get_params(config_id, expe_name=None):
    """
    Get experiment parameters.

    Parameters
    ----------
    config_id: str
        Name of the config.
    expe_name: str
        Name of the experiment, optional.

    Returns
    -------
    params: dict
        Dictionary of experiment parameters.

    """
    if config_id == 'dqn':
        from epidemioptim.configs.dqn import params
    elif config_id == 'goal_dqn':
        from epidemioptim.configs.goal_dqn import params
    elif config_id == 'goal_dqn_constraints':
        from epidemioptim.configs.goal_dqn_constraints import params
    elif config_id == 'nsga_ii':
        from epidemioptim.configs.nsga_ii import params
    else:
        raise NotImplementedError
    if expe_name:
        params.update(expe_name=expe_name)
    return params