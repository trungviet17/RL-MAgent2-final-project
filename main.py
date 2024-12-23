from magent2.environments import battle_v4
import os
import cv2
from agent.base_agent import Agent, RandomAgent, PretrainedAgent, Final_Agent, MyPretrainedAgent
import time
import argparse
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
        self.save_dir = save_dir
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


def init_agent(agent_name, n_observation, n_actions):
    if agent_name == "random":
        return RandomAgent(n_observation, n_actions)
    elif agent_name == "pretrained":
        return PretrainedAgent(n_observation, n_actions)
    elif agent_name == "final":
        return Final_Agent(n_observation, n_actions)
    elif agent_name == "self_play":
        return MyPretrainedAgent(n_observation, n_actions, model_path= 'model/state_dict/self_play.pt', nets_name = "pretrained")
    elif agent_name == "my_random":
        return MyPretrainedAgent(n_observation, n_actions, model_path= 'model/state_dict/my_model.pt', nets_name = "pretrained")
    else:
        raise ValueError("Invalid agent name")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo games for 2 Agents")
    parser.add_argument("-blue_agent", type=str, required=True, help="Name of blue agent")
    parser.add_argument("-red_agent", type=str, required=True, help="Name of red agent")
    parser.add_argument("-save_path", type=str, required=True, help="Path to save model")
    args = parser.parse_args()




    infer = Inference('battle_v4', args.save_path)
    n_actions = infer.env.action_space("red_0").n
    n_observation = infer.env.observation_space("red_0").shape

    agent1 = init_agent(args.red_agent, n_observation, n_actions)
    agent2 = init_agent(args.blue_agent, n_observation, n_actions)


    # agent1 = MyPretrainedAgent(n_observation,  n_actions, model_path= 'model/state_dict/self_play.pt', nets_name = "pretrained")
    # agent1 = MyQAgent(n_observation,  n_actions, model_path= 'model/state_dict/my_model5.pt')
    # agent2 = RandomAgent(n_observation, n_actions)
    # agent2 = PretrainedAgent(n_observation,  n_actions)
    # agent2 = Final_Agent(n_observation,  n_actions)
    # agent1 = PretrainedAgent(n_observation,  n_actions, model_path= 'model/state_dict/model2.pt')

    infer.play(agent1, agent2)
    infer.draw_video('game')

