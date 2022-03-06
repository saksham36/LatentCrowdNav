from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob, traj=None):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        if self.policy.name in ('LiliSARL', 'Lili', 'LiliSARL2'):
            action = self.policy.predict(state, traj)
        else:
            action = self.policy.predict(state)
        return action
