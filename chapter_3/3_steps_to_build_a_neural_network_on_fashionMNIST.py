from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset')  # This can be any directory you

# want to download FMNIST to
fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images = fmnist.data
tr_targets = fmnist.targets

class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float()/255
        x = x.view(-1, 28*28)
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ix):
        x , y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)
    
def get_data():
    train = FMNISTDataset(tr_images, tr_targets)
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    return train_dl

from torch.optim import SGD

def get_model():
    model = nn.Sequential(
        nn.Linear(28*28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    )
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)
    
    return model, loss_fn, optimizer

def train_batch(x, y, model, opt, loss_fn):
    model.train() # <- let's hold on to this until we reach
    # droupout section
    # call your model like any python function on your batch
    # of inputs
    prediction = model(x)

    # compute loss
    batch_loss = loss_fn(prediction, y)

    # based on the forward pass in `model(x)` compute all the
    # gradients of 'model.parameters()'
    batch_loss.backward()

    # apply new-weights = f(old-weights, old-weight-gradients)
    # where "f" is the optimizer
    opt.step()

    # Flush gradients memory for next batch of calculations
    opt.zero_grad()

    return batch_loss.item()

# since there's no need for updating weights,
# we might as well not compute the gradients.
# Using this '@' decorator on top of functions
# will disable gradient computation in the entire function
@torch.no_grad()
def accuracy(x, y, model):
    model.eval()  # <- let's wait till we get to dropout
    # section
    
    # get the prediction matrix for a tensor of `x` images
    prediction = model(x)

    # compute if the location of maximum in each row
    # coincides with ground truth
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    
    return is_correct.cpu().numpy().tolist()

trn_dl = get_data()
model, loss_fn, optimizer = get_model()

losses, accuracies = [], []

for epoch in range(5):
    print(f'Epoch: {epoch}')
    epoch_losses, epoch_accuracies = [], []
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        epoch_losses.append(batch_loss)

    epoch_loss = np.array(epoch_losses).mean()

    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        epoch_accuracies.extend(is_correct)
    
    epoch_accuracy = np.mean(epoch_accuracies)

    print(f'Loss: {epoch_loss}, Accuracy: {epoch_accuracy}')
    
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)

epochs = np.arange(5) + 1
plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.title('Loss over increasing epochs')
plt.plot(epochs, losses, label='Tranning Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy over increasing epochs')
plt.plot(epochs, accuracies, label='Training Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()
plt.show()