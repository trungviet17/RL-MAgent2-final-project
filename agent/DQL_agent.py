from base_agent import Agent
from model.DQL_model import DRQNets

class DQLAgent(Agent):

    def __init__(self, n_actions, n_observation, hidden_dim: int = 120):
        super().__init__(n_actions)

        




    def get_action(self, observation):
        pass

    def learn(self):
        pass