import torch 
from model.pretrained_model import QNetwork 



"""
File này dùng để khởi tạo agent trong huấn luyện và infer  
"""


class Agent: 
    """
    Class abstract cho tất cả agent 
    
    """


    def __init__(self, n_actions): 
        self.n_action = n_actions
        pass 

    def get_action(self, observation): 
        """
        Hàm dùng để thực hiện action theo agent 
        Input: 
            observation : np.array - mảng ảnh của môi trường
        Output:
            action : int - hành động được chọn
        """

        pass 


    def learn(self): 
        """
        hàm dùng để huấn luyện agent 
        """
        pass 


class RandomAgent(Agent): 
    def __init__(self, n_actions):
        super().__init__(n_actions)


    def get_action(self, observation):
        return torch.randint(0, self.n_action, (1,)).item()  

        



class PretrainedAgent(Agent): 

    def __init__(self, n_observation, n_actions): 
        super().__init__(n_actions)
        self.qnetwork = QNetwork(n_observation, n_actions)

        self.qnetwork.load_state_dict(
            torch.load("model/state_dict/red.pt", weights_only=True, map_location="cpu")
        ) 

    def get_action(self, observation):
        observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
        with torch.no_grad():
            q_values = self.qnetwork(observation)
        action = torch.argmax(q_values, dim=1).numpy()[0]

        return action




