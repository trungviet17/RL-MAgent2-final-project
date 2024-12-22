"""
File này dùng để training model 

"""
from .agent.base_agent import Agent, RandomAgent, PretrainedAgent
from utils.memory import StateMemory, ReplayBuffer
from agent.DQL_agent import DQNAgent
from agent.Qmix_agent import QMIXAgent

from magent2.environments import battle_v4
import os 
import torch 
from torch.utils.data import DataLoader
from time import time
import argparse


class Trainer : 
    """
    Sử dụng blue để huấn luyện 
    
    """
    def __init__(self, env, red_agent: Agent, blue_agent:Agent, buffer, batch_size = 64, is_self_play = False): 
        self.red_agent = red_agent
        self.blue_agent = blue_agent
        self.buffer = buffer 
        self.batch_size = batch_size 
        self.env = env 
        self.is_self_play = is_self_play 

    def agent_give_action(self, name: str, observation):
        if self.is_self_play : 
            return  self.blue_agent.get_action(observation)
        if name == "blue": 
            return  self.blue_agent.get_action(observation)
        return self.red_agent.get_action(observation)


    
    def update_memory(self, is_longterm: bool = False): 
        """
        Tạo ra một vòng lặp lưu trữ và cập nhật dữ liệu cho từng agent 
        """
        self.env.reset()
        prev_obs = {}
        prev_actions = {}
        red_reward = 0 
        blue_reward = 0 

        prev_team = "red"

        n_kills = {"red": 0, "blue": 0}
        # vong lap 1 
        for idx, agent in enumerate(self.env.agent_iter()): 
            prev_ob, reward, termination, truncation, _ = self.env.last()
            team = agent.split("_")[0]
            n_kills[team] += (reward > 4.5)

            if truncation or termination: 
                prev_action = None
            else: 
                if agent.split("_")[0] == "red": 
                    prev_action =  self.agent_give_action("red", prev_ob)
                    red_reward += reward
                else: 
                    prev_action = self.agent_give_action("blue", prev_ob)
                    blue_reward += reward 
    

        
            prev_obs[agent] = prev_ob 
            prev_actions[agent] = prev_action 
            self.env.step(prev_action)

            if (idx + 1) % self.env.num_agents == 0: break 

        # vong lap 2 
        for agent in self.env.agent_iter(): 

            obs, reward, termination, truncation, _ = self.env.last()
            team = agent.split("_")[0]
            n_kills[team] += (reward > 4.5)
            
            if truncation or termination: 
                action = None 
            else: 
                if agent.split("_")[0] == "red" : 
                    action = self.agent_give_action("red", obs)
                    red_reward += reward 
                
                else: 
                    action = self.agent_give_action("blue", obs)
                    blue_reward += reward
                

            self.env.step(action)
            if isinstance(self.buffer, StateMemory):
                if team != prev_team : 
                    self.buffer.ensemble()  
                    prev_team = team
                idx = int(agent.split("_")[1]) % self.buffer.grouped_agents
                self.buffer.push(
                    idx,
                    prev_obs[agent], 
                    prev_actions[agent], 
                    reward, 
                    obs, 
                    termination 
                )
            else: 
                 self.buffer.push(
                    prev_obs[agent], 
                    prev_actions[agent], 
                    reward, 
                    obs, 
                    termination 
                )

            prev_obs[agent] = obs 
            prev_actions[agent] = action

        return  blue_reward - red_reward,  n_kills, blue_reward # red thắng  


    def save_model (self, file_path):
        
        torch.save(self.blue_agent.q_net.state_dict(), file_path)
        print(f"Model saved to {file_path}")
    
    def train(self, episodes=500, target_update_freq=2, is_type = "dqn"):
        gap_rewards = []


        for eps in range(episodes): 
            start = time()
            gap_reward, n_kills, blue_reward = self.update_memory()

            
            if is_type == "qmix": 
                self.buffer.ensemble()
            
            dataloader = DataLoader(self.buffer, batch_size = self.batch_size, shuffle = True)
            # print(f"Out of dataloader {len(self.buffer)}")
            self.blue_agent.train(dataloader)
            
            self.blue_agent.decay_epsilon()
            if eps % target_update_freq == 0:
                self.blue_agent.update_target_network()
    
            end = time() - start 
            
            # wandb.log({
            #     "episode": eps,
            #     "gap_rewards": gap_reward,
            #     "epsilon": self.blue_agent.epsilon,
            #     "time": end,
            #     "red_kill": n_kills["red"], 
            #     "blue_kill": n_kills["blue"]
            # })
    
            
            gap_rewards.append(gap_reward)
            print(f"Episode {eps}, Gap Reward: {gap_reward}, Total Reward: {blue_reward}, Epsilon: {self.blue_agent.epsilon:.2f}, Time: {end}, Kill: {n_kills}")
    
        self.env.close()


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Train a Double Deep Q for MAgent")
    parser.add_argument("-mode", type=str, required=True, help="self-play if you want to train with self-play, otherwise random")
    parser.add_argument("-save_dir", type=str, required=True, help="Path to save model")
    args = parser.parse_args()

    is_self_play = args.is_self_play == "self-play"
    save_dir = args.save_dir

    env = battle_v4.env(map_size=45, render_mode="rgb_array", attack_opponent_reward=0.5)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    observation_shape = env.observation_space("red_0").shape
    action_shape = env.action_space("red_0").n
    num_agents = 27

    blue_agent = DQNAgent(observation_shape,action_shape, device=device)

    red_agent = RandomAgent(action_shape)
    buffer = ReplayBuffer(capacity=10000)

    trainer = Trainer(env, red_agent, blue_agent, buffer, batch_size = 64, is_self_play=is_self_play)
    trainer.train(episodes = 70)

    trainer.save_model(save_dir)
