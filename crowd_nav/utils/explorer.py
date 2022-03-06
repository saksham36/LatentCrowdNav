import logging
import copy
import torch
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from tqdm import tqdm
import time
import random

class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        for i in tqdm(range(k)):
            ob = self.env.reset(phase=phase)
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                action = self.robot.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                
                actions.append(action)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)

                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            
            self.memory.push((state, value))


class LiliExplorer(object):
    def __init__(self, env, robot, device, memory=None, traj_memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.traj_memory = traj_memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None
        self.expanded_state_size = int(self.env.time_limit//self.env.time_step)
        self.term_action = ActionXY(0, 0) if self.robot.kinematics == 'holonomic' else ActionRot(0, 0)
        self.prev_traj = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        for i in tqdm(range(k)):
            ob = self.env.reset(phase=phase)
            if len(self.traj_memory) > 0:
                self.prev_traj, _ = random.choice(self.traj_memory)
            done = False
            states = []
            actions = []
            rewards = []
            dones  = []
            while not done:
                if self.robot.policy.name in ('LiliSARL', 'Lili', 'LiliSARL2'):
                    # try:
                    action = self.robot.act(ob, self.prev_traj.unsqueeze(0))
                    # except Exception as e:
                    #     print(e)
                    #     import pdb; pdb.set_trace()
                else:
                    action = self.robot.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(self.prev_traj, (states, actions, rewards, dones), imitation_learning)
                    
            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, prev_traj, traj, imitation_learning=False):
        # tic = time.time()
        
        if prev_traj is None:
            states, actions, rewards, dones = traj
            if imitation_learning: 
                expanded_states = states + [states[-1] for _ in range(self.expanded_state_size - len(states))]

                expanded_rewards = rewards + [0 for _ in range(self.expanded_state_size - len(states))]

                expanded_actions = actions + [self.term_action for _ in range(self.expanded_state_size - len(states))]

                expanded_dones = dones+ [True for _ in range(self.expanded_state_size - len(states))]
  
                traj = torch.stack([torch.cat([torch.flatten(self.target_policy.transform(expanded_states[idx]).to('cpu'),start_dim=0), 
                                    torch.Tensor(expanded_actions[idx], device='cpu'), 
                                    torch.Tensor([expanded_rewards[idx]], device='cpu'),
                                    torch.Tensor([1 if expanded_dones[idx] == True else 0], device='cpu')], dim=-1) \
                                 for idx in range(self.expanded_state_size)]) 
            self.prev_traj = traj
            return # the left most traj is not pushed into memory

        states, actions, rewards, dones = traj
        if self.traj_memory is None or self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        
        if imitation_learning: 
            expanded_states = states + [states[-1] for _ in range(self.expanded_state_size - len(states))]

            expanded_rewards = rewards + [0 for _ in range(self.expanded_state_size - len(states))]

            expanded_actions = actions + [self.term_action for _ in range(self.expanded_state_size - len(states))]

            expanded_dones = dones+ [True for _ in range(self.expanded_state_size - len(states))]
  
            traj = torch.stack([torch.cat([torch.flatten(self.target_policy.transform(expanded_states[idx]).to('cpu'),start_dim=0), 
                                    torch.Tensor(expanded_actions[idx], device='cpu'), 
                                    torch.Tensor([expanded_rewards[idx]], device='cpu'),
                                    torch.Tensor([1 if expanded_dones[idx] == True else 0], device='cpu')], dim=-1) \
                                 for idx in range(self.expanded_state_size)])
                           
        for i, state in enumerate(states):
            reward = rewards[i]
            done = dones[i]
            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if done or i == len(states)-1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    self.target_model.eval()
                    q_hat, pred_traj = self.target_model(next_state.unsqueeze(0).to(self.device), prev_traj.unsqueeze(0).to(self.device))
                    value = reward + gamma_bar * torch.amax(q_hat.data, -1)
            
            value = torch.Tensor([value]).to('cpu')

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            # print(f'Time to upate memory: {time.time()-tic}')
            self.memory.push((state, value))  # value of prev_traj. NOT traj
        self.traj_memory.push((prev_traj, traj))
    
def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
