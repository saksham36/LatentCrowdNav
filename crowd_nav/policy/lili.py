import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from copy import deepcopy
import numpy as np


# class LiliSAC(nn.Module): 
#     """
#     Line by line adaptation of LILI's SAC algorithm - Annie, et. al 2020
#     """

#     def __init__(
#         self,
#         policy,
#         Qs,
#         pool,
#         cell_num,  #State size, should be 4
#         plotter=None,
#         action_space=81, # Pass via env?
#         lr=3e-4,
#         reward_scale=1.0,
#         target_entropy="auto",
#         discount=0.99,
#         tau=5e-3,
#         target_update_interval=1,
#         action_prior="uniform",
#         reparameterize=False,


#         latent_dim=8, # may need to change
#         encoder_size=(128,128),
#         decoder_size=(128,128),

#         encode_obs_act=True,
#         encode_rew=True,
#         encode_next_obs=False,

#         recon_rew=True,
#         recon_next_obs=False,
#         continuous=True,

#         pretrain_iters=0,
#         per_task_batch_size=8,
#         episode_length=50,

#         save_full_state=False,
#         **kwargs,

#         ):
#         """
#         Args:
#             policy: A policy function approximator.
#             initial_exploration_policy: ('Policy'): A policy that we use
#                 for initial exploration which is not trained by the algorithm.
#             Qs: Q-function approximators. The min of these
#                 approximators will be used. Usage of at least two Q-functions
#                 improves performance by reducing overestimation bias.
#             pool (`PoolBase`): Replay pool to add gathered samples to.
#             plotter (`QFPolicyPlotter`): Plotter instance to be used for
#                 visualizing Q-function during training.
#             lr (`float`): Learning rate used for the function approximators.
#             discount (`float`): Discount factor for Q-function updates.
#             tau (`float`): Soft value function target update weight.
#             target_update_interval ('int'): Frequency at which target network
#                 updates occur in iterations.
#             reparameterize ('bool'): If True, we use a gradient estimator for
#                 the policy derived using the reparameterization trick. We use
#                 a likelihood ratio based estimator otherwise.
#         """

#         # super(SAC, self).__init__(**kwargs) # TODO: Implement train and eval!

#         self._policy = policy

#         self._Qs = Qs
#         self._Q_targets = tuple(deepcopy(Q) for Q in Qs)

#         self._pool = pool
#         self._plotter = plotter

#         self._policy_lr = lr
#         self._Q_lr = lr

#         self._reward_scale = reward_scale
#         self._target_entropy = (
#             -action_space  # TODO: Possible source of error? Multiply by n_humans
#             if target_entropy == 'auto'
#             else target_entropy)

#         self._state_dim = cell_num
#         self._action_dim = action_space
#         self._latent_dim = latent_dim
#         self._pretrain_iters = pretrain_iters
#         self._encoder_size = encoder_size
#         self._decoder_size = decoder_size

#         self._encode_obs_act = encode_obs_act
#         self._encode_rew = encode_rew
#         self._encode_next_obs = encode_next_obs
#         self._recon_rew = recon_rew
#         self._recon_next_obs = recon_next_obs
#         self._continuous = continuous

#         self._episode_length = episode_length
#         self._per_task_batch_size = per_task_batch_size

#         self._encoder_optimizer = torch.optim.Adam(lr=self._Q_lr) # Q_encoder_optimizer'

#         self._decoder_optimizer = torch.optim.Adam(lr=1e-3)  # decoder_optimizer

#         self._discount = discount
#         self._tau = tau
#         self._target_update_interval = target_update_interval
#         self._action_prior = action_prior

#         self._reparameterize = reparameterize

#         self._save_full_state = save_full_state

#         self._build()
        
#     def _build(self):
#         super(SAC, self)._build()

#         self._init_encoder_update()
#         self._init_actor_update()
#         self._init_critic_update()
#         self._init_diagnostics_ops()

class QNetwork(nn.Module):
    def __init__(self, input_dim, self_action_size, self_state_dim, phi_e_dims, psi_h_dims, attention_dims, with_global_state,
                 cell_size, cell_num,
                 buffer_output_dim, encoder_dims, decoder_dims, Q_dims):
        super().__init__()
        self.action_dim = self_action_size
        self.self_state_dim = self_state_dim
        self.global_state_dim = phi_e_dims[-1]
        self.phi_e = mlp(input_dim, phi_e_dims, last_relu=True) # mlp1 = phi_e
        self.psi_h = mlp(phi_e_dims[-1], psi_h_dims)  # mlp2 = psi_h
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(phi_e_dims[-1] * 2, attention_dims)  #  psi_alpha
        else:
            self.attention = mlp(phi_e_dims[-1], attention_dims) # psi_alpha
        self.cell_size = cell_size
        self.cell_num = cell_num

        self.encoder = mlp(buffer_output_dim, encoder_dims)
        latent_embedding_dim = encoder_dims[-1]
        self.decoder = mlp(latent_embedding_dim, decoder_dims)
        assert decoder_dims[-1] == buffer_output_dim
        # psi_h_dims[-1]: c
        # self.self_state_dim: s
        # latent_embedding_dim: z
        # self.action_dim: a
        Q_input_dims = psi_h_dims[-1] + self.self_state_dim + latent_embedding_dim
        self.Q = mlp(Q_input_dims, Q_dims) # mlp3 = Q
        assert Q_dims[-1] == self.action_dim
        self.attention_weights = None



    def forward(self, state, traj):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :param traj: tensor of shape (batch_size, 5*H) =>(state, action, reward, next state, terminal) * max history length
        :return: # TODO: Verify H is in config file
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        phi_e_output = self.phi_e(state.view((-1, size[2])))
        psi_h_output = self.psi_h(phi_e_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(phi_e_output.view(size[0], size[1], -1), 1, keepdim=True)  # e_m
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([phi_e_output, global_state], dim=1)  # alpha (attention score)
        else:
            attention_input = phi_e_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = psi_h_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)  # crowd representation

        
        latent_rep = self.encoder(traj)
        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature, latent_rep], dim=1)
        
        Q = self.Q(joint_state)
        # value = torch.max(Q,1)
        return Q


class SARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights
