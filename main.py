from magent2.environments import battle_v4
import os
import cv2
from agent.base_agent import Agent, RandomAgent, PretrainedAgent, Final_Agent, MyPretrainedAgent
from agent.DQL_agent import MyQAgent
import time

"""
File code này dùng để chạy thử nghiệm các model đã được train sẵn trong ván chơi 

#NOTE : vòng loop chi duyển qua những agent nào còn sống 
"""

class Inference: 

    """
    Input : 
        env_name : str - tên của môi trường 
        agent1 : khởi tạo agent 
        agent2 : khởi tạo agent2

    Ouput: 
        Run môi trường + save thành video 
    """

    def __init__(self, env_name: str,  save_dir: str):
        self.env = battle_v4.env(map_size=45, render_mode="rgb_array")
        self.save_dir = 'video'
        os.makedirs(self.save_dir, exist_ok=True)
        self.fps = 35


    def play(self, red_agent: Agent, blue_agent: Agent):
        """
        Hàm nhận thực hiện một game với 2 agent cho trước 

        Input: 
            red_agent : Agent - agent đỏ 
            blue_agent : Agent - agent xanh

        Output:
            Lưu vào frame 
        
        """
        self.env.reset()
        self.frames = []
        str_time = time.time()
        # state = self.env.state()
        name = ['blue_' + str(i) for i in range(81)]
        break_bool = True 

       


        temp = 0 
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
            # if agent.split("_")[0] == 'blue':
            #     print(f"Red name:{agent}")


            if "red" in agent: break_bool = True

            if agent in name and break_bool: 
                self.frames.append(self.env.render())
                break_bool = False
            
        print(f"Time: {time.time() - str_time}") 
           
        self.env.close()

    def draw_video(self, names: str):
        """
        Vẽ video từ frame 
        """

        height, width, _ = self.frames[0].shape
        out = cv2.VideoWriter(
            os.path.join(self.save_dir, f"{names}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )

        for frame in self.frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print("Xong !")



if __name__ == "__main__":

    infer = Inference('battle_v4', 'video')
    n_actions = infer.env.action_space("red_0").n
    n_observation = infer.env.observation_space("red_0").shape

    agent1 = MyPretrainedAgent(n_observation,  n_actions, model_path= 'model/state_dict/my_random5.pt')
    # agent1 = MyQAgent(n_observation,  n_actions, model_path= 'model/state_dict/my_model5.pt')
    agent2 = RandomAgent(n_observation, n_actions)
    # agent2 = PretrainedAgent(n_observation,  n_actions, model_path= 'model/state_dict/red.pt')
    # agent2 = Final_Agent(n_observation,  n_actions, model_path= 'model/state_dict/red_final.pt')
    # agent1 = PretrainedAgent(n_observation,  n_actions, model_path= 'model/state_dict/model2.pt')

    infer.play(agent2, agent1)
    infer.draw_video('myrandom_vs_random')



    pass 



    # env = battle_v4.env(map_size=45, render_mode="rgb_array")
    # vid_dir = "video"
    # os.makedirs(vid_dir, exist_ok=True)
    # fps = 35
    # frames = []

    # # random policies
    # env.reset()
    # for agent in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()

    #     if termination or truncation:
    #         action = None  # this agent has died
    #     else:
    #         action = env.action_space(agent).sample()
    #         print(action)
    #         break

        # env.step(action)

        # if agent == "red_0":
        #     frames.append(env.render())

    # height, width, _ = frames[0].shape
    # out = cv2.VideoWriter(
    #     os.path.join(vid_dir, f"random.mp4"),
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     fps,
    #     (width, height),
    # )
    # for frame in frames:
    #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     out.write(frame_bgr)
    # out.release()
    # print("Done recording random agents")

    # pretrained policies
    # frames = []
    # env.reset()
    # from model.pretrained_model import QNetwork
    # import torch

    # print(env.observation_space("red_0").shape)
    # print(env.action_space("red_0").n)


    # q_network = QNetwork(
    #     env.observation_space("red_0").shape, env.action_space("red_0").n
    # )
    # q_network.load_state_dict(
    #     torch.load("red.pt", weights_only=True, map_location="cpu")
    # )
    # for agent in env.agent_iter():

    #     observation, reward, termination, truncation, info = env.last()

    #     if termination or truncation:
    #         action = None  # this agent has died
    #     else:
    #         agent_handle = agent.split("_")[0]
    #         if agent_handle == "red":
    #             observation = (
    #                 torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
    #             )
    #             with torch.no_grad():
    #                 q_values = q_network(observation)
    #             action = torch.argmax(q_values, dim=1).numpy()[0]
    #         else:
    #             action = env.action_space(agent).sample()
                

    #     env.step(action)

    #     if agent == "red_0":
    #         frames.append(env.render())

    # height, width, _ = frames[0].shape
    # out = cv2.VideoWriter(
    #     os.path.join(vid_dir, f"pretrained.mp4"),
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     fps,
    #     (width, height),
    # )
    # for frame in frames:
    #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     out.write(frame_bgr)
    # out.release()
    # print("Done recording pretrained agents")

    # env.close()
