import torch.nn as nn
import torchvision.models as models

resnet50 = models.resnet50(pretrained=True)

class PreTrainedConvNet(nn.Module):
    def __init__(self):
        super(PreTrainedConvNet, self).__init__()

        self.stack = nn.Sequential(
            resnet50,
            nn.Linear(1000, 4196),
            nn.ReLU(),
            nn.Linear(4196, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
    
    def forward(self, x):
        return self.stack(x)
