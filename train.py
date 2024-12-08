"""
File này dùng để training model 

"""
from .agent.base_agent import Agent, RandomAgent, PretrainedAgent
from .agent.DQL_agent import DQLAgent
from magent2.environments import battle_v4
import os 
import torch 




class Trainer: 

    def __init__(self, render_mode):

        self.env = battle_v4.env(map_size=45, render_mode=render_mode)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    


    def train_dpn(self, episodes: int, target_update_freq: int, red_agent: DQLAgent, blue_agent: Agent) : 

        total_rewards = []

        for episode in range(episodes):
            self.env.reset()
            ep_reward = 0

            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()

                if termination or truncation:
                    action = None
                else:
                    agent_handle = agent.split("_")[0]
                    if agent_handle == "red":
                        action = red_agent.get_action(observation)
                    else:
                        action = blue_agent.get_action(observation)

                self.env.step(action)
                ep_reward += reward

                if agent == 'red_0':
                    red_agent.buffer.push(observation, action, reward, 
                                          self.env.last()[0], termination or truncation)
                    
                    ep_reward += reward

                red_agent.train()

                if episode % target_update_freq == 0:
                    red_agent.update_target_network()

            total_rewards.append(ep_reward)
            print(f"Episode {episode}, Total Reward: {ep_reward}, Epsilon: {red_agent.epsilon:.2f}")

        self.save(red_agent.qnetwork, "/home/trung/workspace/final-project-RL/model/state_dict/dqn.pt")
        self.env.close()
                    


    def save_model(model, file_path):
        torch.save(model.state_dict(), file_path)
        print(f"Model saved to {file_path}")
        

    def plot_reward(self):
        pass


if __name__ == '__main__': 

    trainer = Trainer(render_mode='rgb_array', save_dir='video')

    observation_shape = trainer.env.observation_space("red_0").shape
    action_shape = trainer.env.action_space("red_0").n

    red_agent = DQLAgent(observation_shape, action_shape, device=trainer.device)
    blue_agent = PretrainedAgent(observation_shape, action_shape, device=trainer.device)

    trainer.train_dpn(1000, 10, red_agent, blue_agent)