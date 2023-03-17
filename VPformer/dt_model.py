import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class decoder_block(nn.Module):
    def __init__(self):
        super(decoder_block, self).__init__()
        self.seq_length = 8

        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.attention = nn.MultiheadAttention(\
                         embed_dim = 256,
                         num_heads = 8,
                         dropout = 0.1,
                         bias = True,
                         batch_first = True)


        mask = (1 - torch.tril(torch.ones(self.seq_length*3, self.seq_length*3))).to(dtype = torch.bool)
        #for t in range(20, 60):
        #    for k in range(20):
        #        mask[t][k] = True
        #for t in range(40, 60):
        #    for k in range(20, 40):
        #        mask[t][k] = True
        self.register_buffer("mask", \
                              mask,)

        self.mlp = nn.Sequential(\
                   nn.Linear(256, 4*256), \
                   nn.GELU(),
                   nn.Linear(4*256, 256),
                   nn.Dropout(0.1))

    def forward(self, x):
        batch_size, token_size, embed_size = x.shape

        out1 = self.ln1(x)

        mask = self.mask

        attn_out, _ = self.attention(out1, out1, out1, attn_mask = mask, need_weights = False)

        out1 = out1 + attn_out

        out1 = out1 + self.mlp(self.ln2(out1))

        return out1



class NBV_decision_transformer(nn.Module):

    def __init__(self):
        super(NBV_decision_transformer, self).__init__()


        self.seq_length = 8
        #transfer scene to a tensor
        self.conv1 = nn.Conv3d(1, 16, kernel_size = (4, 4, 4), stride = 2)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(16, 32, kernel_size = (4, 4, 4))
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size = (4, 4, 4))
        self.pool3 = nn.MaxPool3d(2)

        #transfer action to a tensor
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 512)

        #self.fc3 = nn.Linear(4, 64)
        #self.fc4 = nn.Linear(64, 128)

    
        self.embed_timestep = nn.Embedding(self.seq_length, 256)
        self.embed_scene = nn.Linear(2880, 256)
        self.embed_action = nn.Linear(512, 256)
        self.embed_reward = nn.Linear(1, 256)

        self.embed_ln = nn.LayerNorm(256)

        self.blocks = nn.Sequential(
                *[
                  decoder_block()
                  for _ in range(6)
                ]
                )

        self.predict_actions = nn.Linear(256, 7)

        self.predict_stds = nn.Linear(256, 7)
    
    def forward(self, states, actions, rewards):
        batch_size, seq_length = states.shape[0], states.shape[1]
        timestep = torch.tensor(np.array([x for x in range(self.seq_length)]))
        timestep = timestep.to('cuda', dtype = torch.int)

        time_embedding = self.embed_timestep(timestep)
        time_embedding = torch.unsqueeze(time_embedding, 0)

        action_latent = F.relu(self.fc1(actions))
        action_latent = F.relu(self.fc2(action_latent))
        action_latent = F.relu(self.fc3(action_latent))
        action_embedding = self.embed_action(action_latent)

        #scene_latent = self.fc3(states)
        #scene_latent = self.fc4(scene_latent)
        #scene_embedding = self.embed_scene(scene_latent)

        reward_embedding = self.embed_reward(rewards)

        scene_embedding = None
        for t in range(batch_size):
            out1 = F.relu(self.conv1(states[t]))
            out2 = self.pool1(out1)

            out3 = F.relu(self.conv2(out2))
            out4 = self.pool2(out3)

            out5 = F.relu(self.conv3(out4))
            out6 = self.pool3(out5)
             
            out7 = out6.view(-1, 2880)

            out8 = self.embed_scene(out7)
            out8 = torch.unsqueeze(out8, 0)

            if scene_embedding == None:
                scene_embedding = out8
            else:
                scene_embedding = torch.cat((scene_embedding, out6), 0)

        scene_embedding += time_embedding
        action_embedding += time_embedding
        reward_embedding += time_embedding

        stacked_inputs = torch.stack((reward_embedding, 
                                      scene_embedding,
                                      action_embedding), dim = 1)\
                         .permute(0, 2, 1, 3).reshape(batch_size, self.seq_length*3, 256)

        #stacked_inputs = torch.stack((reward_embedding, action_embedding), dim = 1).permute(0, 2, 1, 3).reshape(batch_size, self.seq_length*2, 256)
        
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_inputs = self.blocks(stacked_inputs)
 
        result = stacked_inputs.reshape(batch_size, self.seq_length, 3, 256).permute(0, 2, 1, 3)

        extract = result[:,1].clone().detach().requires_grad_(True)
        
        means = torch.tanh(self.predict_actions(extract))

        stds = torch.sigmoid(self.predict_stds(extract))

        return means, stds
            
if __name__ == '__main__':
    model = NBV_decision_transformer()
    states = torch.randn(4, 8, 1, 101, 121, 86)
    actions = torch.randn(4, 8, 7)
    rewards = torch.randn(4, 8, 1)
    test_input = torch.randn(4, 24, 256)
    out = model(states, actions, rewards)
    #model2 = decoder_block()
    #out = model2(test_input)

