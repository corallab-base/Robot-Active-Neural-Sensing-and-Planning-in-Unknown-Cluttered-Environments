import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
from SceneDataLoader import SceneDataset
from model import CamScoreNet
import torch.optim as optim
import torch.nn as nn
import math
import numpy as np


def feed_forward(scene, camera_pose):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scene = np.where(scene == -1, 255, scene)
    scene = np.transpose(scene, (3, 0, 1, 2))
    scene = torch.from_numpy(np.array([scene]))
    camera_pose = torch.from_numpy(np.array([camera_pose]))
    

    model = CamScoreNet()
    model = model.to(device)

    model.load_state_dict(torch.load('/home/corallab3/Documents/active_sensing/learning/saved_MODEL/CamScoreNet_best.pth'))
    model.eval()
    
    total_loss = 0

    with torch.no_grad():
        num_samples = 0
        scene = scene.to(device, dtype = torch.float)
        camera_pose = camera_pose.to(device, dtype = torch.float)

        outputs = model(scene, camera_pose)

    return outputs[0][0].item()

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


    #set hyperparameters
    if not args.model_path:
        num_epoches = args.num_epochs
        learning_rate = args.learning_rate
        batch_size = args.batch_size


        train_dataset = SceneDataset_object_focus('./scene_train/')
        validation_dataset = SceneDataset_object_focus('./scene_validation/')
        test_dataset = SceneDataset_object_focus('./scene_test/')

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

        num_steps = len(train_loader)

        model = CamScoreNet()
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
        for t in range(num_epoches):
            model.train()
            print (f'Epoch: {t+1}/{num_epoches}')
            total_loss_train = 0
            total_num_train = 0
            total_loss_validation = 0
            total_num_validation = 0
            for i, (scene, camera_pose, score) in enumerate(train_loader):
                scene = scene.to(device, dtype = torch.float)
                camera_pose = camera_pose.to(device, dtype = torch.float)
                score = score.to(device, dtype = torch.float)

                outputs = model(scene, camera_pose)

                loss = criterion(outputs, score)
                total_loss_train += loss
                total_num_train += 1


                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                if (i + 1) % 10 == 0:
                    print (f'step: {i+1}/{num_steps}, Loss: {loss.item()*10000:.4f}')

            with torch.no_grad():
                for i, (scene, camera_pose, score) in enumerate(validation_loader):
                    scene = scene.to(device, dtype = torch.float)
                    camera_pose = camera_pose.to(device, dtype = torch.float)
                    score = score.to(device, dtype = torch.float)

                    outputs = model(scene, camera_pose)

                    total_loss_validation += criterion(outputs, score)
                    total_num_validation += 1
            average_loss_train = math.sqrt(total_loss_train*1.0/total_num_train)*100
            average_loss_validation = math.sqrt(total_loss_validation*1.0/total_num_validation)*100
            track_train.append(average_loss_train)
            track_validation.append(average_loss_validation)

            if average_loss_validation < best_vali_loss:
                best_vali_loss = average_loss_validation
                print('Saving best model to ./saved_MODEL_object_focus')
                file_name = 'CamScoreNet_best.pth'
                torch.save(model.state_dict(), './saved_MODEL_object_focus/' + file_name)
            print (f"After epoch: {t + 1}, average training loss: {average_loss_train:.4f}, average validation loss: {average_loss_validation:.4f}")


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
        
        test_dataset = SceneDataset('./scene_test/')

        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

        model = CamScoreNet()
        model = model.to(device)

        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        
        total_loss = 0

        with torch.no_grad():
            num_samples = 0
            for i, (scene, camera_pose, score) in enumerate(test_loader):
                scene = scene.to(device, dtype = torch.float)
                camera_pose = camera_pose.to(device, dtype = torch.float)
                score = score.to(device, dtype = torch.float)

                outputs = model(scene, camera_pose)

                num_samples += 1

                total_loss += criterion(outputs, score)
                m, n = score.shape

                for k in range(m):
                    if score[k][0] < 0.5:
                        print (f"Check view: {score[k][0]}, {outputs[k][0]}")
                print (f"True percentage: {score[0][0]}, predicted percentage: {outputs[0][0]}")
        print (f'Average loss in test data: {math.sqrt(total_loss*1.0/num_samples)*100}')
        



if __name__ == '__main__':
    main()
