# By Jeff Yuanbo Han (u6617017), 2018-05-01.
import torch
from torch.autograd import Variable
from load_data import train_data

# Split x (features) and y (targets)
x_array = train_data[:, 1:]
y_array = train_data[:, 0]

# Create Tensors to hold inputs and outputs, and wrap them in Variables
X = Variable(torch.Tensor(x_array).float())
Y = Variable(torch.Tensor(y_array).float())

# Define the number of neurons for input layer, hidden layer and output layer
input_neurons = 126
hidden_neurons = 1
#hidden_neurons = 3
output_neurons = 1
# Define learning rate and number of epoch on training
learning_rate = 1
num_epoch = 3000
# lambda for auxiliary term in cost function
lamb1 = 0
lamb2 = 0.001
# Threshold for classification
thresh = 0.4


# Define a customised neural network structure
net = torch.nn.Sequential(
    torch.nn.Linear(input_neurons, hidden_neurons, bias=False),
    #torch.nn.Tanh(),
    torch.nn.Linear(hidden_neurons, output_neurons, bias=False),
    torch.nn.Sigmoid()
)


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Y_pred, Y, net, lamb1, lamb2):
        for w in net.parameters():
            # Auxillary term
            c = lamb2 / 2 * torch.sum(w*w*(w-1)*(w-1)*(w+1)*(w+1))\
                + lamb1 / 2 * torch.sum(w*w)
            break

        new_loss = torch.nn.MSELoss()(Y_pred, Y) + c
        return new_loss


# Define loss function
loss_func = Loss()

# Define optimiser
child_counter = 0
for child in net.children():
    if child_counter == 1:
        for param in child.parameters():
            param.data = torch.ones([output_neurons, hidden_neurons])
            param.requires_grad = False
    elif child_counter == 0:
        optimiser = torch.optim.SGD(child.parameters(), lr=learning_rate)
    child_counter += 1

# Store all losses for visualisation
all_losses = []

# Train a neural network
for epoch in range(num_epoch):
    # Perform forward pass: compute predicted y by passing x to the model.
    Y_pred = net(X)[:,0]

    # Compute loss
    #lamb2 = epoch/num_epoch * (0.01-0.001) + 0.001
    loss = loss_func(Y_pred, Y, net, lamb1, lamb2)
    all_losses.append(loss.data)

    # Print progress
    if epoch % 200 == 0:
        # Convert three-column predicted Y values to one column for comparison
        #_, predicted = torch.max(F.softmax(Y_pred), 1)
        predicted = (torch.sign(Y_pred-thresh) + 1) / 2

        # Calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.6f %%'
              % (epoch + 1, num_epoch, loss.data, 100 * sum(correct)/total))

    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its parameters
    optimiser.step()


for w_mat in net.parameters():
    a = torch.FloatTensor(w_mat.data)
    i = 0
    for w_row in w_mat:
        j = 0
        for w in w_row:
            if w.data.numpy() >= 0.3:
                a[i,j] = 1
            elif w.data.numpy() <= -0.3:
                a[i,j] = -1
            else:
                a[i,j] = 0
            j += 1
        i += 1
    w_mat.data = a


Y_pred = net(X)[:,0]
loss = loss_func(Y_pred, Y, net, lamb1, lamb2)
predicted = (torch.sign(Y_pred-thresh) + 1) / 2

# Calculate and print accuracy
total = predicted.size(0)
correct = predicted.data.numpy() == Y.data.numpy()

print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.6f %%'
      % (epoch + 1, num_epoch, loss.data, 100 * sum(correct)/total))

# Save weights
torch.save(net.state_dict(), 'net_weights')
