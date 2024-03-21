import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
import tqdm
from itertools import count
from FCDuelingNet import FCNDueling


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'using {device}')
print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))




class ReplayMemory(): # this one will contain sample for learning
    def __init__(self, capacity=1000000):
        self.state_buffer = np.empty(shape=(capacity), dtype=np.ndarray)
        self.action_buffer = np.empty(shape=(capacity), dtype=np.ndarray)
        self.reward_buffer = np.empty(shape=(capacity), dtype=np.ndarray)
        self.next_state_buffer = np.empty(shape=(capacity), dtype=np.ndarray)
        self.failure_buffer = np.empty(shape=(capacity), dtype=np.ndarray)
        self.idx = 0

    def sample(self, batch_size):
        if(len(self) > batch_size):
            idxs = np.random.choice(self.idx, batch_size, replace=False)
        else:
            idxs = np.random.choice(self.idx, len(self), replace=False)

        return (np.vstack(self.state_buffer[idxs]), np.vstack(self.action_buffer[idxs]), \
                np.vstack(self.reward_buffer[idxs]), np.vstack(self.next_state_buffer[idxs]), np.vstack(self.failure_buffer[idxs]))
    
    def push(self, experience):
        self.state_buffer[self.idx], self.action_buffer[self.idx], self.reward_buffer[self.idx], self.next_state_buffer[self.idx], \
            self.failure_buffer[self.idx] = experience
        self.idx += 1

    def __len__(self):
        return self.idx

class e_lin_greedy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.3, decay_steps=10000):
        self.t = 0
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps
        self.exploratory_action_taken = None
        
    def _epsilon_update(self):
        epsilon = 1 - self.t / self.decay_steps
        epsilon = (self.init_epsilon - self.min_epsilon) * epsilon + self.min_epsilon
        epsilon = np.clip(epsilon, self.min_epsilon, self.init_epsilon)
        self.t += 1
        return epsilon

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else: 
            action = np.random.randint(len(q_values))

        self.epsilon = self._epsilon_update()
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action
    
#strategy for val.
class greedy():
    def __init__(self):
        pass

    def select_action(self, model, state):
        q_values = model(state).detach().to('cpu').numpy()
        action = np.argmax(q_values)
        return action

#dqn agent all-in-one class
class DQN():
    def __init__(self, mk_model_fn, env, training_strategy, mk_optimizer_fn, \
                replay_buffer, batch_size=64, gamma=0.95, num_episodes=2000, criterion=nn.SmoothL1Loss(), common_startegy= greedy(), device='cuda:0'):
        
        self.online_model = mk_model_fn().to(device)
        self.target_model = mk_model_fn().to(device)
        self.env = env
        self.training_strategy = training_strategy
        self.optimizer = mk_optimizer_fn(self.online_model)
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.criterion = criterion
        self.common_strategy = common_startegy
        self.torchify = lambda x: torch.Tensor(x).to(device)
        self.device = device
        self.rewards = []

    def train(self, update_model_every_x_step=10, min_buff_size=10):
        step_elapsed = 0
        self.online_model.train()# <-- need to use dropout
        for e in tqdm.tqdm(range(self.num_episodes)):
            state, _ = self.env.reset()
            sum_reward = 0.0
            for _ in count():
                action = self.training_strategy.select_action(self.online_model, torch.tensor(state).to(self.device))
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                sum_reward += reward
                #done = terminated or info
                done = terminated or truncated

                self.replay_buffer.push((state, np.array(int(action)), np.array(reward), next_state, np.array(done)))

                state = next_state

                if(len(self.replay_buffer) > min_buff_size):
                    self.optimize_model()

                if(step_elapsed % update_model_every_x_step == 0):
                    self.update_model()

                step_elapsed += 1

                if done:
                    self.rewards.append(sum_reward)
                    break
                
    def update_model(self, tau=0.5):
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(tau * target.data + (1 - tau) * online.data)
            #target.data.copy_(online.data)

    def optimize_model(self):    
        batch_s, batch_a, batch_r, batch_ns, batch_tr = self.replay_buffer.sample(self.batch_size)
    
        
        batch_s = torch.from_numpy(batch_s).float().to(self.device)
        batch_a = torch.from_numpy(batch_a).long().to(self.device)
        batch_r = torch.from_numpy(batch_r).float().to(self.device)
        batch_ns = torch.from_numpy(batch_ns).float().to(self.device)
        batch_tr = torch.from_numpy(batch_tr).float().to(self.device)

        max_a_q_sp = self.target_model(batch_ns).detach().max(1)[0].unsqueeze(1)
        target_q_sa = batch_r + (self.gamma * max_a_q_sp * (1 - batch_tr))
        q_sa = self.online_model(batch_s).gather(1, batch_a)

        value_loss = self.criterion(q_sa, target_q_sa)
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()
    
    def save_model(self, model_name='torch_prj/DuelingDQN/models/dueling_dqn_net.pt'):
        model_scripted = torch.jit.script(self.online_model)
        model_scripted.save(model_name)

#create env, model, optimizer and memory buffer
env_name = 'CartPole-v1'
env = gym.make(env_name)
EPISODES = 500        

#create fn that generates model, because we need 2 nets: target and online
if __name__ == '__main__':
    mk_model_fn = lambda: FCNDueling(env.observation_space.shape[0], env.action_space.n, n_hidden=(512, 256, 128))
    mk_optimizer_fn = lambda model: optim.RMSprop(params=model.parameters(), lr=0.0005)
    memory = ReplayMemory()

    #agent itself
    dqn_agent = DQN(mk_model_fn=mk_model_fn, env=env, training_strategy=e_lin_greedy(), \
                    mk_optimizer_fn=mk_optimizer_fn, replay_buffer=memory, num_episodes=EPISODES)#if you haven't gpu set device = 'cpu' 

    dqn_agent.train() # start training
    dqn_agent.save_model()

    rewards = np.array(dqn_agent.rewards)

    plt.plot(rewards[np.arange(0, len(rewards))[::20]])
    plt.savefig('torch_prj/DuelingDQN/images/rewards')