from epidemioptim.optimization.dqn.dqn import DQN
from epidemioptim.optimization.nsga.nsga import NSGAII


def get_algorithm(algo_id, env, params={}):
    if algo_id == 'DQN':
        alg = DQN(env, params)
    elif algo_id == 'NSGAII':
        alg = NSGAII(env, params)
    else:
        return NotImplementedError

    return alg
