import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.multi_human_rl import MultiHumanRL
import numpy as np
# TODO: May need to make new CADRL.py to accomodate new VNetwork params
# TODO: May need to make this VNetwork to accomodate SARL and then modify LILI Q to V network
class VNetwork(nn.Module):
    def __init__(self, hist, num_humans, input_dim, self_action_space, self_state_dim, phi_e_dims, psi_h_dims, attention_dims, with_global_state,
                 cell_size, cell_num,
                 buffer_output_dim, encoder_dims, decoder_dims, Q_dims,lili_flag):
        super().__init__()
        self.hist = hist
        self.input_dim = input_dim
        self.num_humans = num_humans
        self.num_actions = self_action_space
        self.self_state_dim = self_state_dim
        self.lili_flag = lili_flag

        if not lili_flag:
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
        latent_embedding_dim = buffer_output_dim
        self.encoder = mlp(self.hist*(self.num_humans*input_dim+2), encoder_dims + [latent_embedding_dim])
        self.decoder = mlp(latent_embedding_dim, decoder_dims+[self.hist*(self.num_humans*input_dim+1)])
        # psi_h_dims[-1]: c
        # self.self_state_dim: s
        # latent_embedding_dim: z
        # self.num_actions: a
        if not lili_flag:
            Q_input_dims = psi_h_dims[-1] + self.self_state_dim + latent_embedding_dim
        else:
            Q_input_dims = self.self_state_dim + latent_embedding_dim
        self.Q = mlp(Q_input_dims, Q_dims) # mlp3 = Q
        assert Q_dims[-1] == self.num_actions
        self.attention_weights = None



    def forward(self, state, prev_traj):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation
        :param state: tensor of shape (batch_size, # of humans, length of a rotated state) -> current state
        :param traj: tensor of shape (batch_size, H, (|S|+|A|+2) =>(state, action, reward, done)
        :return: # TODO: Verify H is in config file
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        if not self.lili_flag:
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
            weighted_feature = torch.sum(torch.mul(weights, features), dim=1)  # crowd representation (c)

        prev_traj_input = prev_traj[:,:, :size[1]*size[2]+2]
        
        latent_rep = self.encoder(torch.reshape(prev_traj_input, (prev_traj_input.shape[0],-1)))
        # concatenate agent's state with global weighted humans' state
        # import pdb; pdb.set_trace()
        if not self.lili_flag:
            joint_state = torch.cat([self_state, weighted_feature, latent_rep], dim=1)
        else:
            joint_state = torch.cat([self_state, latent_rep], dim=1)
        decoder_output = self.decoder(latent_rep) # s_hat, r_hat
        Q = self.Q(joint_state)
        # value = torch.max(Q,1)
        return Q, decoder_output


class LiliSARL(MultiHumanRL):
    def __init__(self, hist, num_human, action_space):
        super().__init__()
        self.name = 'LiliSARL'
        self.action_dim = action_space
        self.num_human = num_human
        self.hist = hist
    
    def configure(self, config):
        self.set_common_parameters(config)
        phi_e_dims = [int(x) for x in config.get('lili-sarl', 'phi_e_dims').split(', ')]
        psi_h_dims = [int(x) for x in config.get('lili-sarl', 'psi_h_dims').split(', ')]
        Q_dims = [int(x) for x in config.get('lili-sarl', 'Q_dims').split(', ')]
        buffer_output_dim = config.getint('lili-sarl', 'latent_rep_dims')
        encoder_dims = [int(x) for x in config.get('lili-sarl', 'encoder_dims').split(', ')]
        decoder_dims = [int(x) for x in config.get('lili-sarl', 'decoder_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('lili-sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('lili-sarl', 'with_om')
        with_global_state = config.getboolean('lili-sarl', 'with_global_state')

        lili_flag = config.getboolean('lili-sarl', 'lili_flag')
        self.model = VNetwork(self.hist, self.num_human, self.input_dim(), self.action_dim, self.self_state_dim, phi_e_dims, psi_h_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num, 
                                  buffer_output_dim, encoder_dims, decoder_dims, Q_dims, lili_flag)
        print(f'{self.name}: Flag: {lili_flag}')
        self.multiagent_training = config.getboolean('lili-sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-LILI-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights
    
    def predict(self, state, prev_traj):
        """
        A modified predict class that takes pairwise joint state as input to Q network.
        The input to the Q network is always of shape
        state: (batch_size, # humans, rotated joint state length)
        prev_traj: (batch_size, H, (|S|+|A|+2) =>(state, action, reward, done)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            joint_state = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                  for human_state in state.human_states], dim=0)
            
            rotated_batch_input = self.rotate(joint_state).unsqueeze(0).to(self.device)
            
            if prev_traj is None:
                prev_traj = torch.zeros(rotated_batch_input.shape[0], self.hist, rotated_batch_input.shape[1] * rotated_batch_input.shape[2] +2 + 2, device=self.device)
            
            Q, pred_traj = self.model(rotated_batch_input, prev_traj)
            max_action = self.action_space[torch.argmax(Q, dim=1)]

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    def compute_reward(self, nav, humans):
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        del_phi = np.arctan2(nav.py + self.vy*self.time_step, nav.px+nav.vx*self.time_step) - np.arctan2(nav.py, nav.px)
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        else:
            reward = -0.005 * np.abs(del_phi) + ((dmin - 0.2) * 0.5 * self.time_step if dmin<0.2 else 0)

        return reward


class Lili(LiliSARL):
    def __init__(self, hist, num_human, action_space):
        super().__init__(hist, num_human, action_space)
        self.name = 'Lili'
    
    def configure(self, config):
        self.set_common_parameters(config)
        phi_e_dims = [int(x) for x in config.get('lili', 'phi_e_dims').split(', ')]
        psi_h_dims = [int(x) for x in config.get('lili', 'psi_h_dims').split(', ')]
        Q_dims = [int(x) for x in config.get('lili', 'Q_dims').split(', ')]
        buffer_output_dim = config.getint('lili', 'latent_rep_dims')
        encoder_dims = [int(x) for x in config.get('lili', 'encoder_dims').split(', ')]
        decoder_dims = [int(x) for x in config.get('lili', 'decoder_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('lili', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('lili', 'with_om')
        with_global_state = config.getboolean('lili', 'with_global_state')

        lili_flag = config.getboolean('lili', 'lili_flag')
        
        self.model = VNetwork(self.hist, self.num_human, self.input_dim(), self.action_dim, self.self_state_dim, phi_e_dims, psi_h_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num, 
                                  buffer_output_dim, encoder_dims, decoder_dims, Q_dims, lili_flag)
        print(f'{self.name}: Flag: {lili_flag}')
        self.multiagent_training = config.getboolean('lili', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-LILI'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))