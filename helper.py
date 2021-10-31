import numpy as np
import matplotlib.pyplot as plt
import torch


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def visualize_one(data):
    image, label = data

    image = np.transpose(unnormalize(image[0].numpy()), (1, 2, 0))
    label = label[0].item()

    plt.axis(False)
    plt.imshow(image)
    plt.title(f'Class: {classes[label]}')
    plt.savefig('out/test.jpg')


def unnormalize(img):
    return img / 2 + 0.5


def train_loop(dataloader, model, loss_fn, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, torch.max(y, 1)[1])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>6d}/{size:>5d}]")

def validate_loop(dataloader, model, loss_fn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, torch.max(y, 1)[1]).item()
            correct += (pred.argmax(1) == torch.max(y, 1)[1]).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
