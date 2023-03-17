import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class CamScoreNet(nn.Module):
    def __init__(self):
        super(CamScoreNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size = (4,4,4), stride = 2)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(16, 32, kernel_size = (4,4,4))
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size = (4,4,4))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(2)


        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 512)

        self.fc4 = nn.Linear(512 + 2880, 1024)
        self.fc5 = nn.Linear(1024, 256)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, 1)

    def forward(self, x, y):
        out1 = F.relu(self.conv1(x))
        out2 = self.bn1(out1)
        out3 = self.pool1(out2)

        out4 = F.relu(self.conv2(out3))
        out5 = self.bn2(out4)
        out6 = self.pool2(out5)

        out7 = F.relu(self.conv3(out6))
        out8 = self.bn3(out7)
        out_x = self.pool3(out8)

        out9 = F.relu(self.fc1(y))
        out10 = F.relu(self.fc2(out9))
        out_y = F.relu(self.fc3(out10))
    
        out_x = out_x.view(-1, 2880)
        out_cat = torch.cat((out_x, out_y), 1)
        
        out11 = F.relu(self.fc4(out_cat))
        out12 = F.relu(self.fc5(out11))
        out13 = F.relu(self.fc6(out12))
        out_final = torch.sigmoid(self.fc7(out13))

        return out_final


if __name__ == '__main__':
    model = CamScoreNet()
    rand_input_x = torch.randn(4, 1, 101, 121, 86)
    rand_input_y = torch.randn(4, 7)
    output = model(rand_input_x, rand_input_y)
    print (output)

