import torch
import random
import gym
from dqn import greedy
    

env_name = 'CartPole-v1'
model = torch.jit.load('torch_prj/DQN/model/dqn_net(works).pt').to('cpu')
model.eval()
env = gym.make(env_name, render_mode='human')
observation, info = env.reset(seed=random.randint(1, 30))
strategy = greedy()
act = lambda state: strategy.select_action(model, torch.tensor(state))

for _ in range(2000):
   action = act(observation)  # User-defined policy function
   print(action)
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      print('again')
      observation, info = env.reset(seed=random.randint(1, 30))
env.close()