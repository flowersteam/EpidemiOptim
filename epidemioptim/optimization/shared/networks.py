import torch
import torch.nn as nn
import numpy as np

import torch.autograd as autograd
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Critic(nn.Module):
    def __init__(self,
                 n_critics,
                 dim_state,
                 dim_actions,
                 dim_goal,
                 layers,
                 goal_ids):
        """
        Critic class that can use several critic networks (one for each cost).
        Modeling each cost by a critic and using the mixing weights to mix q-values
        instead of rewards is an idea from Agent57, Badia et al., 2019.

        Parameters
        ----------
        n_critics: int
            Number of critics (e.g. number of costs).
        dim_state: int
            Dimension of states.
        dim_actions: int
            Dimension of actions.
        dim_goal: int
            Dimension of goals, that is parameters of the cost function.
        layers: list of ints
            Description of the inner layer sizes.
        goal_ids: tuple of n_critics tuples of ints
            Each element is a tuple that describes the indexes of the goal to be added as input of the network (UVFA, Schaul et al., 2015).
        """
        super().__init__()

        # Initialize Q networks.
        self.qs = [QNetFC(dim_state, dim_goal, dim_actions, layers, goal_ids[i]) for i in range(n_critics)]
        self.n_critics = n_critics

    def get_model(self):
        """
        Extract critic state dicts
        """
        return [q.state_dict() for q in self.qs]

    def set_model(self, model):
        """
        Set critic load dicts
        """
        for q, o in zip(self.qs, model):
            q.load_state_dict(o)

    def set_goal_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for i_q in range(self.n_critics):
            for param in self.qs[i_q].parameters():
                tmp = np.product(param.size())
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
                cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor as numpy array
        """
        return np.concatenate([np.hstack([v.data.cpu().numpy().flatten() for v in q.parameters()]).copy() for q in self.qs])

    def forward(self, obs):
        """
        Make a forward pass for each q network.
        Parameters
        ----------
        obs: tensor of size (n_batch, dim_state + dim_goal)
            Vector of observation, that is the concatenation of the state and goal vectors.

        Returns
        -------
        tuple size n_critics
            Q-value for each q-network

        """
        return tuple(q.forward(obs) for q in self.qs)



class QNetFC(nn.Module):
    def __init__(self,
                 dim_state,
                 dim_goal,
                 dim_actions,
                 layers,
                 goal_ids):
        """
        Fully connected Q-Network.

        Parameters
        ----------
        dim_state: int
            Dimension of the state.
        dim_goal: int
            Dimension of the goal.
        dim_actions: int
            Dimension of the actions.
        layers: list of ints
            Description of the hidden layers sizes.
        goal_ids: tuple of ints
            Describes the indexes of the goal to be added as input of the network (UVFA, Schaul et al., 2015).
        """
        super(QNetFC, self).__init__()
        self.dim_state = dim_state
        self.dim_goal = dim_goal
        self.dim_actions = dim_actions
        self.goal_ids = np.array(goal_ids)

        self.input_ids =  np.concatenate([np.arange(self.dim_state), self.goal_ids + self.dim_state])
        self.layers = (dim_state + len(goal_ids),) + layers + (dim_actions,)
        self.network = mlp(sizes=self.layers, activation=nn.ReLU, output_activation=nn.Identity)

    @property
    def nb_params(self):
        return count_vars(self.network)

    def forward(self, obs):
        return self.network(obs[:, self.input_ids])

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())
            param.data.copy_(torch.from_numpy(
                params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp


    def get_params(self):
        """
        Returns parameters of the actor as numpy array
        """
        return np.hstack([v.data.cpu().numpy().flatten() for v in self.parameters()]).copy()

    def act(self, state):
        with torch.no_grad():
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state).detach().numpy().flatten()
        action = np.argmax(q_value)
        return np.atleast_1d(action)
