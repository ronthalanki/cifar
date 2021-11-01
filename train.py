import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Lambda
from torchvision.transforms.transforms import ColorJitter

from helper import visualize_one, train_loop, validate_loop
from model import PreTrainedConvNet

transform = Compose([Resize(224), ToTensor()])

target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(y)), value=1))

learning_rate = 1e-3
batch_size = 64
epochs = 100

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform, target_transform=target_transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False)

visualizeOne = True 
if visualizeOne:
    dataiter = iter(trainloader)
    visualize_one(dataiter.next())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = PreTrainedConvNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(trainloader, model, loss_fn, optimizer)
    validate_loop(testloader, model, loss_fn)

torch.save(model.state_dict(), 'out/conv_model_weights.pth')
print("Done Training Model!")