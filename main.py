from magent2.environments import battle_v4
import os
import cv2
from agent.base_agent import Agent, RandomAgent, PretrainedAgent

"""
File code này dùng để chạy thử nghiệm các model đã được train sẵn trong ván chơi 
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
        self.frames = []

    

    def play(self, red_agent: Agent, blue_agent: Agent):
        self.env.reset()

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

            if agent == 'red_0':
                self.frames.append(self.env.render())
                
        self.env.close()

    def draw_video(self, names: str):

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
        print("Done recording random agents")











if __name__ == "__main__":

    infer = Inference('battle_v4', 'video')
    n_actions = infer.env.action_space("red_0").n
    n_observation = infer.env.observation_space("red_0").shape

    agent1 = PretrainedAgent(n_observation,  n_actions)
    agent2 = RandomAgent(n_actions)

    infer.play(agent1, agent2)
    infer.draw_video('pretrained_vs_random')



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
