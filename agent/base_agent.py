import torch 
from model.networks import Pretrained_QNets, Final_QNets
import numpy as np 


"""
File này dùng để khởi tạo agent trong huấn luyện và infer  
"""


class Agent: 
    """
    Class abstract cho tất cả agent 
    
    """
    def __init__(self, n_observation , n_actions): 
        self.n_action = n_actions
        self.n_observation = n_observation
        

    def get_action(self, observation): 
        """
        Hàm dùng để thực hiện action theo agent 
        Input: 
            observation : np.array - mảng ảnh của môi trường
        Output:
            action : int - hành động được chọn
        """
        pass 


    def train(self): 
        """
        hàm dùng để huấn luyện agent 
        """
        pass 


class RandomAgent(Agent): 
    def __init__(self, n_observation,  n_actions):
        super().__init__(n_observation , n_actions)


    def get_action(self, observation):
        return torch.randint(0, self.n_action, (1,)).item()  

        



class PretrainedAgent(Agent): 

    def __init__(self, n_observation, n_actions, model_path: str): 
        super().__init__(n_observation, n_actions)
        self.qnetwork = Pretrained_QNets(n_observation, n_actions)
        self.n_action = n_actions
        self.qnetwork.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu")
        ) 

    def get_action(self, observation):

        if np.random.rand() < 0.5:
            return np.random.randint(self.n_action)
        else:
            observation = (
                        torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                    )
            with torch.no_grad():
                q_values = self.qnetwork(observation)
            action = torch.argmax(q_values, dim=1).numpy()[0]

        return action


class Final_Agent(Agent): 

    def __init__(self, n_observation, n_actions, model_path: str): 
        super().__init__(n_observation, n_actions)
        self.qnetwork = Final_QNets(n_observation, n_actions)
        self.n_action = n_actions
        self.qnetwork.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu")
        ) 

    def get_action(self, observation):
        observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
        with torch.no_grad():
            q_values = self.qnetwork(observation)
        action = torch.argmax(q_values, dim=1).numpy()[0]

        return action



