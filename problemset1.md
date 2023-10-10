# Load MNIST and show montage

import numpy as np # basic python
import matplotlib.pyplot as plt # for plots
import torch
from torchvision import datasets # get the datasets to use
from skimage.util import montage # to show montage of the images
from skimage.io import imread

# definitions

def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))

def plot(x):
    if type(x) == torch.Tensor :
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(7, 7)
    plt.show()

def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))

train_set = datasets.KMNIST('./data', train=True, download=True)
test_set = datasets.KMNIST('./data', train=False, download=True)

# train_set is to be a PyTorch dataset containing training data.
# .data extracts the data portion from the dataset.
# .numpy() converts the data from a PyTorch tensor to a NumPy array.
# X is assigned this NumPy array, contains the training images.
X = train_set.data.numpy()

# test_set is to be a PyTorch dataset containing test data.
# .data extracts the data portion from the test dataset.
# .numpy() converts the data from a PyTorch tensor to a NumPy array.
# X_test is assigned this NumPy array, which likely contains the test images.
X_test = test_set.data.numpy()

# train_set is to be a PyTorch dataset containing training data.
# .targets extracts the target labels from the training dataset.
# .numpy() converts the target labels from a PyTorch tensor to a NumPy array.
# Y is assigned this NumPy array, which likely contains the labels for the training data.
Y = train_set.targets.numpy()

# test_set is assumed to be a PyTorch dataset containing test data.
# .targets extracts the target labels from the test dataset.
# .numpy() converts the target labels from a PyTorch tensor to a NumPy array.
# Y_test is assigned this NumPy array, which likely contains the labels for the test data.
Y_test = test_set.targets.numpy()

# adds an extra dimension to the X array at index 1 (i.e., it adds a channel dimension).
# This is typically done for compatibility with convolutional neural networks (CNNs) which
# expect input data in the format [batch_size, channels, height, width].
X = X[:, None, :, :] / 255

X_test = X_test[:, None, :, :] / 255
X.shape
montage_plot(X[125:150, 0, :, :])  # this actually plots it

## Run random y=mx model on MNIST
# Reshape image data tensor from (60000, 1, 28, 28) to (60000, 784)
X = X.reshape(X.shape[0], 784)

# Reshape image test data tensor from (60000, 1, 28, 28) to (60000, 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X.shape

X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)

X.shape

X = X.T

X.shape

"""# Run random y=mx model on MNIST"""

x = X[:, 0:1]
x.shape
M = GPU(np.random.rand(10, 784))

# This line performs matrix multiplication between M and x.
# It's essentially a linear transformation where M acts as weights and x is the input.
# The result is assigned to the variable y.
y = M @ x

batch_size = 64

x = X[:, 0:batch_size]

# Similar to line 3, this line initializes a new random matrix M with the same dimensions
M = GPU(np.random.rand(10, 784))

y = M @ x

y = torch.argmax(y, 0)

torch.sum((y == Y[0:batch_size])) / batch_size

"""# Train random walk model

"""

m_best = 0 # best weight matrix found during the search
acc_best = 0 # best accuracy achieved during the search

for i in range(100000): # Start a loop that will run 100,000 iterations

    step = 0.0000000001 # Define a small step size that will be used to update the m_best matrix in each iteration. It's a small value, likely for fine-grained adjustments

    m_random = GPU_data(np.random.randn(10, 784)) # Define a small step size that will be used to update the m_best matrix in each iteration. It's a small value, likely for fine-grained adjustments

    m = m_best  + step * m_random # Update the weight matrix m by adding a small step (step) times the random matrix (m_random) to the current m_best. This is part of the random search process, exploring different weight matrices.

    y = m @ X

    # Calculate the maximum value along the 0th axis (likely representing different classes or predictions) using PyTorch's argmax function. This is done to determine the predicted class for each input.
    y = torch.argmax(y, axis=0)

    acc = ((y == Y)).sum() / len(Y)

    if acc > acc_best:
        print(acc.item())
        m_best = m
        acc_best = acc

