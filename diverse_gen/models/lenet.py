import torch
import torch.nn as nn
import torch.nn.functional as F

# define network (LeNet? what does divdis use?)
class LeNet(nn.Module):

    def __init__(self, num_classes=1,dropout_p=0.0) -> None:
        super().__init__()
        self.droput_p = dropout_p
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 56, kernel_size=5)
        self.fc1 = nn.Linear(2016, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.avgpool_2 = nn.AvgPool2d(kernel_size=2)
        self.avgpool_3 = nn.AvgPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor, dropout=True) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = F.dropout(x, p=self.droput_p, training=dropout)
        x = self.avgpool_2(x)
        x = self.relu(self.conv2(x))
        x = F.dropout(x, p=self.droput_p, training=dropout)
        x = self.avgpool_3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, p=self.droput_p, training=dropout)
        x = self.fc2(x)
        x = self.relu(x)
        x = F.dropout(x, p=self.droput_p, training=dropout)
        x = self.fc3(x)
        return x