from epidemioptim.environments.cost_functions.multi_cost_death_gdp_controllable import MultiCostDeathGdpControllable

def get_cost_function(cost_function_id, params={}):
    if cost_function_id == 'multi_cost_death_gdp_controllable':
        return MultiCostDeathGdpControllable(**params)
    else:
        raise NotImplementedError
