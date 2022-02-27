import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL2
# TODO: May need to make new CADRL.py to accomodate new VNetwork params
# TODO: May need to make this VNetwork to accomodate SARL and then modify LILI Q to V network
class VNetwork(nn.Module):
    def __init__(self, input_dim, self_action_space, self_state_dim, phi_e_dims, psi_h_dims, attention_dims, with_global_state,
                 cell_size, cell_num,
                 buffer_output_dim, encoder_dims, decoder_dims, Q_dims):
        super().__init__()
        self.action_dim = len(self_action_space)
        print(f'in lili.py: self_action_space: {self_action_space.shape}')
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


class LiliSARL(MultiHumanRL2):
    def __init__(self):
        super().__init__()
        self.name = 'LiliSARL'

    def configure(self, config):
        self.set_common_parameters(config)
        phi_e_dims = [int(x) for x in config.get('lili-sarl', 'phi_e_dims').split(', ')]
        psi_h_dims = [int(x) for x in config.get('lili-sarl', 'psi_h_dims').split(', ')]
        Q_dims = [int(x) for x in config.get('lili-sarl', 'Q_dims').split(', ')]
        buffer_output_dim = [int(x) for x in config.get('lili-sarl', 'latent_rep_dims').split(', ')]
        encoder_dims = [int(x) for x in config.get('lili-sarl', 'encoder_dims').split(', ')]
        decoder_dims = [int(x) for x in config.get('lili-sarl', 'decoder_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('lili-sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('lili-sarl', 'with_om')
        with_global_state = config.getboolean('lili-sarl', 'with_global_state')
        self.model = VNetwork(self.input_dim(), self.action_space, self.self_state_dim, phi_e_dims, psi_h_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num, 
                                  buffer_output_dim, encoder_dims, decoder_dims, Q_dims)
        self.multiagent_training = config.getboolean('lili-sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-LILI-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights
