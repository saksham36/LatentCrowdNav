from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.lili import LiliSARL, Lili
from crowd_nav.policy.lili2 import LiliSARL2

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['lili-sarl'] = LiliSARL
policy_factory['lili'] = Lili
policy_factory['lili-sarl2'] = LiliSARL2