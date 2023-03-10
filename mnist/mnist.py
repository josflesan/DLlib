import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

# Define hyperparameters
epochs = 5
batch_train = 64
batch_test = 1000
learning_rate = 1e-2
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False  # disable cuDNN nondeterminstic algorithms
torch.manual_seed(random_seed)  # Set manual random seed

# Load Data
loader_train = DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,)  # Global mean and std. deviation of MNIST
                                    )
                                ])),
                                batch_size=batch_train, shuffle=True
)

loader_test = DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,)  # Global mean and std. deviation of MNIST
                                    )
                                ])),
                                batch_size=batch_test, shuffle=True
)

# Visualize some examples
examples = enumerate(loader_test)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap = 'gray', interpolation='none')
    plt.title(f"Ground Truth: {example_targets[i]}")
    plt.xticks([])
    plt.yticks([])

# fig.show()


# Build Network
class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # First 2D convolution from 1 channel (B&W) to 10 channels
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # Second 2D convolution from 10 channels to 20
        self.conv2_drop = nn.Dropout2d()  # Network dropout for regularization
        self.fc1 = nn.Linear(320, 50)  # Fully-connected layer from output of convolution to 50 nodes
        self.fc2 = nn.Linear(50, 10)  # Fully-connected layer from 50 nodes to 10 desired outputs

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Pooling layer with ReLU activation
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # Pooling layer with dropout of second convolution with ReLU activation
        x = x.view(-1, 320)  # Convert output into 2D tensor with same batch size and 320 channels
        x = F.relu(self.fc1(x))  # Fully-connected forward computation
        x = F.dropout(x, training=self.training)  # Apply dropout regularization to x during training
        x = self.fc2(x)  # Second fully-connected layer
        return F.log_softmax(x, -1)  # Apply softmax to final layer to create probability distribution

# Initialize network and optimizer
network = NeuralNet()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)  # The momentum of the optimizer determines weight given to previous updates in current (higher = faster convergence)

# Training the Model
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(loader_train.dataset) for i in range(epochs + 1)]

def train(epoch):
    network.train()  # ?
    for batch_idx, (data, target) in enumerate(loader_train):
        optimizer.zero_grad()  # Make gradients 0 as pytorch accumulates gradients by default
        output = network(data)  # Compute forward pass on data
        loss = F.nll_loss(output, target)  # Compute negative log-likelihood loss between nn output and gold label
        loss.backward()  # Autograd (Backpropagation)
        optimizer.step()  # Propagate new gradients into each of the parameters

        if batch_idx % log_interval == 0:
            print(f"Epoch: {epoch} [{batch_idx * len(data)}/{len(loader_train.dataset)} ({(100. * batch_idx / len(loader_train)):.2f}%)]\tLoss: {round(loss.item(), 6)}")
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(loader_train.dataset))
            )
            torch.save(network.state_dict(), './results/model.pt')  # Save weights to output file
            torch.save(optimizer.state_dict(), './results/optimizer.pt')  # Save optimizer parameters to output file

def test():
    network.eval()  # ?
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader_test:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]  # Get the prediction by getting the maximal value in column
            correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(loader_test)
        test_losses.append(test_losses)        
        print(f'\nTest set: Avg. Loss: {test_loss:.0f}, Accuracy: {correct}/{len(loader_test.dataset)} ({(100. * correct / len(loader_test.dataset)):.2f}%)')

test()
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
