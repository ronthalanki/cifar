from re import A
import torch.nn as nn
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)

class PreTrainedConvNet(nn.Module):
    def __init__(self):
        super(PreTrainedConvNet, self).__init__()

        self.stack = nn.Sequential(
            resnet18,
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.stack(x)