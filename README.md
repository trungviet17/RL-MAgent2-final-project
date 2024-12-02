# MAgent2 RL Final Project

## Project Structure 
Các folder được sắp xếp theo thứ tự như sau: 

```
MAgent2 RL Final Project
├── agent               # cấu trúc lại agent 
│   └── base_agent.py
├── assets              # lưu ảnh 
│   ├── random.gif
│   └── pretrained.gif
├── configs             # lưu các file .yaml là config của model (nếu có) 
├── main.py            # file này dùng để chạy infer 
├── model                      # cài đặt các torch network 
│   ├── pretrained_model.py 
│   └── state_dict           # save checkpoint 
│       └── red.pt
├── README.md 
├── requirements.txt
├── train.py               # file này dùng để huấn luyện model 
└── video                  # lưu video game play 
```





## Overview
In this final project, you will develop and train a reinforcement learning (RL) agent using the MAgent2 platform. The task is to solve a specified MAgent2 environment `battle`, and your trained agent will be evaluated on all following three types of opponents:

1. Random Agents: Agents that take random actions in the environment.
2. A Pretrained Agent: A pretrained agent provided in the repository.
3. A Final Agent: A stronger pretrained agent, which will be released in the final week of the course before the deadline.

Your agent's performance should be evaluated based on reward and win rate against each of these models. You should control *blue* agents when evaluating.


<p align="center">
  <img src="assets/random.gif" width="300" alt="random agent" />
  <img src="assets/pretrained.gif" width="300" alt="pretrained agent" />
</p>

See `video` folder for a demo of how each type of opponent behaves.

## Installation
clone this repo and install with
```
pip install -r requirements.txt
```

## Demos
See `main.py` for a starter code.

## References

1. [MAgent2 GitHub Repository](https://github.com/Farama-Foundation/MAgent2)
2. [MAgent2 API Documentation](https://magent2.farama.org/introduction/basic_usage/)

For further details on environment setup and agent interactions, please refer to the MAgent2 documentation.
