from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import torch
import random
import gym
from dueling_dqn import greedy
    

env_name = 'CartPole-v1'


def save_frames_as_gif(frames, path='./', filename='dueling_net.gif'):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


env = gym.make('CartPole-v1')
model = torch.jit.load('torch_prj/DuelingDQN/models/dueling_dqn_net.pt').to('cpu')
model.eval()
env = gym.make(env_name, render_mode='rgb_array')
observation, info = env.reset(seed=random.randint(1, 30))
strategy = greedy()
act = lambda state: strategy.select_action(model, torch.tensor(state))


frames = []
for t in range(2000):
   
    frames.append(env.render())
    action = act(observation)

    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        break
env.close()
save_frames_as_gif(frames)