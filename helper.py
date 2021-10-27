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
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, torch.max(y, 1)[1])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
