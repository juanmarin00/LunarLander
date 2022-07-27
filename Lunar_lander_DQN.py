#Imports to avoid errors from numpy and pandas
import warnings
warnings.filterwarnings('ignore')

#Imports for visualization and data processing
import gym
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Imports for NN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple


#Lets create our environment!
env = gym.make('LunarLander-v2')


#Set our device for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Class for our Q and target network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_space):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space)
        
    def forward(self, state):
        forward_pass = self.fc1(state)
        forward_pass = F.relu(forward_pass)
        forward_pass = self.fc2(forward_pass)
        forward_pass = F.relu(forward_pass)
        return self.fc3(forward_pass)




#Class to create replay buffers of fixed size (buffer_size)
class ReplayBuffer:
    def __init__(self, action_space, buffer_size, minibatch_length):
        self.action_space = action_space
        self.memory = deque(maxlen=buffer_size)  
        self.minibatch_length = minibatch_length
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "terminal"])
    
    def __len__(self):
        return len(self.memory)

    def sample(self):
    #creates minibath of the specified length

        minibatch = random.sample(self.memory, k=self.minibatch_length)

        states = torch.from_numpy(np.vstack([e.state for e in minibatch if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in minibatch if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in minibatch if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in minibatch if e is not None])).float().to(device)
        terminals = torch.from_numpy(np.vstack([e.terminal for e in minibatch if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, terminals)
    
    def append_experience(self, state, action, reward, next_state, terminal):
        #append experienced state action reward next-state tuple to buffer
        e = self.experience(state, action, reward, next_state, terminal)
        self.memory.append(e)

   


#This function creates a random agent which runs for n_episodes with max_t timestepsilon per episode iof no terminal state is reached
def random_agent(n_episodes=2000, max_t=1000):
    action_space = 4
    
    scores = []                       
    running_avg_scores = deque(maxlen=100)  
    for episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = random.choice(np.arange(action_space))
            next_state, reward, terminal, _ = env.step(action)
            
            state = next_state
            score += reward

            if terminal:
                break 
        running_avg_scores.append(score)       
        scores.append(score)              
      
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(running_avg_scores)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(running_avg_scores)))
        if np.mean(running_avg_scores)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(running_avg_scores)))
            break
    return scores
        



#Class for our "smart" agent used for DQN
class Agent():

    def __init__(self,loss_formula, buffer_size, minibatch_length, gamma, LR):
        self.state_size = 8
        self.action_space = 4  
        self.loss = loss_formula
        self.buffer_size = buffer_size
        self.minibatch_length = minibatch_length
        self.gamma = gamma
        self.LR = LR

        # Replay memory
        self.memory = ReplayBuffer(self.action_space, self.buffer_size, self.minibatch_length)
        self.t_step = 0

        # Q-Network
        self.q_network = QNetwork(self.state_size, self.action_space).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_space).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.LR)
    

    def step(self, state, action, reward, next_state, terminal):
        #append experience to memory and update counter to update after every 4 calls of this function

        # Save experience in replay memory
        self.memory.append_experience(state, action, reward, next_state, terminal)
        
        # Learn every 4 time steps.
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.minibatch_length:
                minibatch = self.memory.sample()
                self.learn(minibatch, self.gamma)

    def act(self, state, epsilon):
        #This function will select an action based on the current state
        #Similar to most act functions we have used in the past
        #it will use random to get a random number in a distribution between 0 and 1
        #if that random numnber is less than epsilon it will select a random action from action space
        #if the number is greater then it will chose optimal action
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()

        # epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_space))

    def learn(self, minibatch, gamma):

        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, terminals = minibatch

        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #Calculate  value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - terminals)
        #Calculate expected value from local network
        q_expected = self.q_network(states).gather(1, actions)
        
        ### Loss calculation, default value is huber, can change to mse for comparison in performance
        if self.loss == 'MSE':
            loss = F.mse_loss(q_expected, q_targets)
        else:
            loss = F.huber_loss(q_expected, q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #update target network
        self.soft_update(self.q_network, self.qnetwork_target)                     


    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(0.001*local_param.data + (1.0-0.001)*target_param.data)





def dqn_fixed_epsilon(dqn_agent,n_episodes=3000, max_t=1000, epsilon = 0.15):

    scores = []                       
    running_avg_scores = deque(maxlen=100)                 
    print("fixed epsilon", epsilon)
    for episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = dqn_agent.act(state, epsilon)
            next_state, reward, terminal, _ = env.step(action)
            dqn_agent.step(state, action, reward, next_state, terminal)
            state = next_state
            score += reward
            if terminal:
                break 
        running_avg_scores.append(score)       
        scores.append(score)              

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(running_avg_scores)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(running_avg_scores)))
        if np.mean(running_avg_scores)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(running_avg_scores)))
            break
    return scores





def dqn(dqn_agent,n_episodes=2000, max_t=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    #This function will take an agent with its parameyters as an argument and will handle the running of episodes
    #and interaction between agent and environment

    #parameters have default values for number of episodes, max timesteps, starting finishing and rate of decsay for epsilon
    #if epsilon_start = epsilon_end and epsilon_deay = 1 the epsilon will be fixed

    scores = []                       
    per_100_scores = {}
    running_avg_scores = deque(maxlen=100)  
    epsilon = epsilon_start                    
    for episode in range(n_episodes):
        state = env.reset()
        score = 0
        for _ in range(max_t):
            action = dqn_agent.act(state, epsilon)
            next_state, reward, terminal, _ = env.step(action)
            dqn_agent.step(state, action, reward, next_state, terminal)
            state = next_state
            score += reward

            if terminal:
                break 

        running_avg_scores.append(score)      
        scores.append(score)              
        epsilon = max(epsilon_end, epsilon_decay*epsilon) 
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(running_avg_scores)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(running_avg_scores)))
            per_100_scores[episode] = np.mean(running_avg_scores)
       
    return [scores, per_100_scores]