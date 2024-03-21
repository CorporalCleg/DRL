import gymnasium
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from itertools import count
import tqdm
from icecream import ic
import matplotlib.pyplot as plt
from nets import FCN, ConvNet
from matplotlib import animation

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env_name = "FlappyBird-v0"
ic(device)

make_env = lambda : gymnasium.make(env_name)
make_model = lambda nS, nA, device='cpu': FCN(nS, nA, n_hidden=(128, 64), device=device)
make_optimizer = lambda model : optim.Adam(params=model.parameters(), lr=0.0005)

def save_model(model, model_name='torch_prj/PG/models/pg_net.pth'):
    torch.save(model.state_dict(), model_name)

def optimize_model(logpas, rewards, policy_optimizer, gamma=1.0):
        T = len(rewards)
        discounts = np.logspace(0, T, num=T, base=gamma, endpoint=False)
        returns = np.array([np.sum(rewards[t:] * discounts[:T-t]) for t in range(T)])

        discounts = torch.FloatTensor(discounts).unsqueeze(1).to(device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
        
        logpas = torch.cat(logpas).to(device)
        
        policy_loss = -(discounts * returns * logpas).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        

def train(make_env, make_model, make_optimizer, max_episodes: int = 200):
    env = make_env()
    nA, nS = env.action_space.n, env.observation_space.shape[0]
    policy_model = make_model(nS, nA, device).to(device)
    policy_optimizer = make_optimizer(policy_model)
    ep_reward = []
    
    policy_model.train()
    for episode in tqdm.tqdm(range(max_episodes)):
        state, _ = env.reset()
        rewards = np.array([])
        logpas = []
        
        for _ in count():
            action, logpa, is_exploratory, entropy = policy_model.full_pass(state)
            state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            
            rewards = np.append(rewards, reward)
            logpas.append(logpa)
            
            if done:
                break


        
        ep_reward.append(np.sum(rewards))

        optimize_model(logpas, rewards, policy_optimizer)
        if episode % 20 == 0:
            print(f'max 20 ep. - {np.max(ep_reward[len(ep_reward) - 20:])}')
    
    save_model(model=policy_model)
    env.close()
    plt.plot(ep_reward)
    plt.savefig('mean_reward.png')

def demo_result(make_env, policy_model=make_model, max_episodes: int = 12):
    env = make_env()
    nA, nS = env.action_space.n, env.observation_space.shape[0]
    policy_model = make_model(nS, nA, device='cpu')
    policy_model.load_state_dict(torch.load('torch_prj/PG/models/pg_net.pth'))
    env = make_env()
    ep_reward = []
    
    policy_model.eval()
    for _ in range(max_episodes):
        state, _ = env.reset()
        rewards = np.array([])
        
        frames = []

        for _ in range(2000):
            frames.append(env.render())
            action = policy_model.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            
            if done:
                break


            rewards = np.append(rewards, reward)
        
        ep_reward.append(np.sum(rewards))
    
    
    save_frames_as_gif(frames)
    env.close()
    plt.plot(ep_reward)
    plt.savefig('mean_reward.png')

def save_frames_as_gif(frames, path='./', filename=f'{env_name}+pg.gif'):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=30)




#train
# train(make_env=make_env, make_model=make_model, make_optimizer=make_optimizer, max_episodes=2000)

#watch result
make_env_demo = lambda : gymnasium.make(env_name, render_mode='rgb_array')
demo_result(make_env=make_env_demo)