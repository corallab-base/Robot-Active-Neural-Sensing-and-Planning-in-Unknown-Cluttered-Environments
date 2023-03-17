import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
from SeqDataset import SeqDataset
from dt_model import NBV_decision_transformer
import torch.optim as optim
import torch.nn as nn
import math
import numpy as np
from torch.distributions.normal import Normal


def nbv_feed_forward(states, actions, rewards, timestep):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = NBV_decision_transformer()
    model = model.to(device)

    model.load_state_dict(torch.load('/home/corallab3/Documents/active_sensing/decision_transformer/saved_MODEL/CamScoreNet_best.pth'))
    model.eval()
    
    total_loss = 0

    states = torch.from_numpy(states)
    actions = torch.from_numpy(actions)
    rewards = torch.from_numpy(rewards)

    states = states.to(device, dtype = torch.float)
    actions = actions.to(device, dtype = torch.float)
    rewards = rewards.to(device, dtype = torch.float)

    with torch.no_grad():
        means, stds = model(states, actions, rewards)

    return means[0, timestep].detach().cpu().numpy(), stds[0, timestep].detach().cpu().numpy()

def main():

    parser = argparse.ArgumentParser(description = 'Convolutional Neural Network')
    parser.add_argument('--num_epochs', default = 10, type = int, \
                        help = 'number of epochs. default: 10')
    parser.add_argument('--learning_rate', default = 1e-3, type = float, \
                        help = 'learning rate. default: 1e-3')
    parser.add_argument('--batch_size', default = 32, type = int, \
                        help = 'batch size. default: 32')
    parser.add_argument('--optimizer', default = 'Adam', choices = ['SGD', 'Adam'], \
                        help = 'optimizer choices. default: Adam')
    parser.add_argument('--model_path', default = '', type = str, \
                        help = 'path of saved model.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_length = 8


    #set hyperparameters
    if not args.model_path:
        num_epoches = args.num_epochs
        learning_rate = args.learning_rate
        batch_size = args.batch_size


        train_dataset = SeqDataset('train_data.txt')
        validation_dataset = SeqDataset('vali_data.txt')
        #test_dataset = SceneDataset('./scene_test/')


        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True)
        #test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

        num_steps = len(train_loader)

        model = NBV_decision_transformer()
        model = model.to(device)

        criterion = nn.MSELoss()

        optimizer = None
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        else:
            sys.exit(1)

        track_train = []
        track_validation = []
    
        best_vali_loss = sys.maxsize
        penalty_weights = 10
        for t in range(num_epoches):
            model.train()
            print (f'Epoch: {t+1}/{num_epoches}')
            total_loss_train = 0
            total_num_train = 0
            total_loss_validation = 0
            total_num_validation = 0
            for i, (states, actions, rewards) in enumerate(train_loader):
                loss = 0
                states = states.to(device, dtype = torch.float)
                actions = actions.to(device, dtype = torch.float)
                rewards = rewards.to(device, dtype = torch.float)


                means, stds = model(states, actions, rewards)

                for k in range(max_length):
                    if sum(actions[0, k]) != 0:
                        loss += criterion(means[0, k], actions[0, k])
                    #m = Normal(means[0, k], stds[0, k])
                    #loss -= sum(m.entropy())
                    #loss += sum(-m.log_prob(actions[0, k]))
                    #loss += penalty_weights*sum(stds[0, k])

                total_loss_train += loss
                total_num_train += 1

                optimizer.zero_grad()

                loss.backward(retain_graph = True)
                optimizer.step()

                if (i + 1) % 10 == 0:
                    print (f'step: {i+1}/{num_steps}, Loss: {math.sqrt(loss.item()/8.0)*10000:.4f}')

            with torch.no_grad():
                for i, (states, actions, rewards) in enumerate(validation_loader):
                    loss = 0
                    states = states.to(device, dtype = torch.float)
                    actions = actions.to(device, dtype = torch.float)
                    rewards = rewards.to(device, dtype = torch.float)


                    means, stds = model(states, actions, rewards)

                    for k in range(max_length):
                        if sum(actions[0, k]) != 0:
                            loss += criterion(means[0, k], actions[0, k])
                        #m = Normal(means[0, k], stds[0, k])
                        #loss -= sum(m.entropy())
                        #loss += sum(-m.log_prob(actions[0, k]))
                        #loss += penalty_weights*sum(stds[0, k])

                    total_loss_validation += loss
                    total_num_validation += 1

            average_loss_train = total_loss_train*1.0/total_num_train
            average_loss_validation = total_loss_validation*1.0/total_num_validation
            track_train.append(average_loss_train.detach().cpu().numpy().item())
            track_validation.append(average_loss_validation.detach().cpu().numpy().item())

            if average_loss_validation < best_vali_loss:
                best_vali_loss = average_loss_validation
                print('Saving best model to ./saved_MODEL')
                file_name = 'CamScoreNet_best.pth'
                torch.save(model.state_dict(), './saved_MODEL/' + file_name)
            print (f"After epoch: {t + 1}, average training loss: {math.sqrt(average_loss_train/8.0)*10000:.4f}, average validation loss: {math.sqrt(average_loss_validation/8.0)*10000:.4f}")


        print (track_train)
        print (track_validation)
        with open("track_train_loss.txt", 'w') as f:
            for element in track_train:
                f.write(str(element)+'\n')
        with open("track_validation_loss.txt", 'w') as f:
            for element in track_validation:
                f.write(str(element)+'\n')
        print(f'smallest validation error : {min(track_validation):.4f}')


    else:

        batch_size = args.batch_size

        criterion = nn.MSELoss()
        
        test_dataset = SeqDataset('test_data.txt')

        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

        model = NBV_decision_transformer()
        model = model.to(device)

        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        
        total_loss = 0

        with torch.no_grad():
            total_loss = 0
            num_samples = 0
            for i, (states, actions, rewards) in enumerate(test_loader):
                loss = 0
                states = states.to(device, dtype = torch.float)
                actions = actions.to(device, dtype = torch.float)
                rewards = rewards.to(device, dtype = torch.float)


                means, stds = model(states, actions, rewards)

                print (i)
                for k in range(max_length):
                    m = Normal(means[0, k], stds[0, k])
                    #loss -= sum(m.entropy())
                    loss += sum(-m.log_prob(actions[0, k]))
                    print ('**************')
                    print (means[0, k])
                    #print (stds[0, k])
                    print (actions[0, k])
                    print ('**************')

                total_loss += loss
                num_samples += 1
        print (f'Average loss in test data: {total_loss*1.0/num_samples}')
        



if __name__ == '__main__':
    main()
