import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Lambda
from torchvision.transforms.transforms import ColorJitter

import matplotlib.pyplot as plt

from helper import visualize_one, train_loop, validate_loop
from model import PreTrainedConvNet

transform = Compose([Resize(224), ToTensor()])

target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(int(y)), value=1))

learning_rate = 1e-3
batch_size = 64 
epochs = 20

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform, target_transform=target_transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False)

visualizeOne = False 
if visualizeOne:
    dataiter = iter(trainloader)
    visualize_one(dataiter.next())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = PreTrainedConvNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

validation_correct_per_epoch = []
validation_loss_per_epoch = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(trainloader, model, loss_fn, optimizer)
    correct, loss = validate_loop(testloader, model, loss_fn)
    validation_correct_per_epoch.append(correct)
    validation_loss_per_epoch.append(loss)

torch.save(model.state_dict(), 'out/conv_model_weights.pth')
plt.plot(validation_correct_per_epoch)
print(validation_correct_per_epoch)
print(validation_loss_per_epoch)
plt.savefig('out/accuracy_loss_plot.jpg')
print("Done Training Model!")