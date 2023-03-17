import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, filename = None):
        self.max_length_ = 8
        if not filename:
            sys.exit(1)
        else:
            self.prefix = '../sim/saved_data_table/'
            self.data_ = set()
            with open(filename, 'r') as f:
                data = f.readlines()
                for t in range(0, len(data),2):
                    f1 = data[t][:-1]
                    f2 = data[t+1][:-1]
                    seq_length = np.load(self.prefix + f1 + '/' + f2 + '/coverage_rate.npy')
                    self.data_.add((f1, f2, len(seq_length)))
            self.data_ = list(self.data_)

    def __getitem__(self, idx):
        f1, f2, length = self.data_[idx]
        states = np.zeros(shape = (self.max_length_, 1, 101, 121, 86))
        #states = np.zeros(shape = (self.max_length_, 4))
        actions = np.zeros(shape = (self.max_length_, 7))
        rewards = np.zeros(shape = (self.max_length_, 1))
        for t in range(length):
            state_file = self.prefix + f1 + '/' + f2 + '/' + 'env' + f2 + '_sequence' + str(t) + '_scene.npy'
            state_data = np.load(state_file)
            state_data = np.transpose(state_data, (3, 0, 1, 2))
            state_data = np.where(state_data == -1, 255, state_data)
            #state_file = self.prefix + 'envs/' + f2 + '_config.npy'
            #state_data = np.load(state_file)
            #state_data = state_data[:4]
            states[t,:] = state_data
            action_file = self.prefix + f1 + '/' + f2 + '/' + 'env' + f2 + '_sequence' + str(t) + '_camera.npy'
            action_data = np.load(action_file)
            actions[t, :] = action_data[:7]
            rewards[t+1] = action_data[7]
        #for t in range(length, self.max_length_):
        #    states[t, :] = states[t-1, :]
        #    actions[t, :] = actions[t-1, :]
        #for t in range(length+1, self.max_length_):
        #    rewards[t] = rewards[t-1]


        return states, actions, rewards

    def __len__(self):
        return len(self.data_)



if __name__ == '__main__':
    seq = SeqDataset('good_data.txt')
    dataloader = DataLoader(seq, batch_size = 1, shuffle = True)
    for i, (states, actions, rewards) in enumerate(dataloader):
        print (rewards)
        print (actions)

    print (states.shape)

