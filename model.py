import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim


class NoduleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv3d(1, 16, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size = 2)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size = 2)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size = 2)
        
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(64 * 8 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x