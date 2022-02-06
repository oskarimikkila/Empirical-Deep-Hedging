import numpy as np
from sklearn import preprocessing
import joblib
import torch
import os

def maybe_make_dirs():
    def make(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
    required = ['results/testing/', 'model/', 'data/', 'settings/']
    for r in required:
        make(r)

def get_model_number(m_name):
    f_name = 'model/' + m_name +'.txt'
    if os.path.isfile(f_name):
        with open(f_name, 'r') as f:
            x = f.readline()
            x = int(x.strip()) + 1
    else:
        x = 0
    with open(f_name, 'w') as f:
        f.write(str(x))

    return x

class StatePrepare():
    def __init__(self, env, action_size, model_name):
        self.name = model_name
        self.process = env.process
        
        state = env.reset() 
        
        self.state_size = state.shape[0]
        self.scaler = self.get_scaler(env, action_size)
    
    def transform(self, x):
        x = self.scaler.transform(np.reshape(x, (1, -1)))
        x = x.reshape((1, self.state_size))
        
        return x
        
    def get_scaler(self, env, action_size, n_samples = 1000):   
        samples = []    
        while len(samples) < n_samples:
            state = env.reset()  
            samples.append(state)
            done = False
            while not done:
                state, _, done, _ = env.step(np.random.rand()*action_size)
                samples.append(state)
        
        scaler = preprocessing.StandardScaler()
        scaler.fit(samples)
        
        return scaler
    
    def load(self, model_name = None):
        if model_name == None:
            model_name = self.name
        self.scaler = joblib.load('model/' + model_name + '_scaler')
    
    def save(self, model_name = None):
        if model_name == None:
            model_name = self.name
        joblib.dump(self.scaler, 'model/' + model_name + '_scaler')

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.empty()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def empty(self):
        
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))
        
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
