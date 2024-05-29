import torch
from torch import nn, optim
from torchgen import model

from models.kan import InitializeKan
from models.mlp import MLP
from models.utils import load_images

# # 5 is the output layer for geometric figures
images, labels = load_images(5, "geometric_figures")
#
# # testing MLP
# hidden_layers = [20]
# mlpModel = MLP(hidden_layers, images, labels)
# # train model
# mlpModel.train(50, learning_rate=0.01)
# # test on single images
# mlpModel.test_single_image()

# Testing KAN
input_size = images.shape[1]
hidden_layers = [64, 32, 16, 5]  # last one matches the output layer
kanModel = InitializeKan(hidden_layers, input_size)

# define criteria for loss function and learning rate
criterion = nn.MSELoss()
optimizer = optim.Adam(kanModel.parameters(), lr=0.001)

# train KAN

# Convert data to PyTorch tensors
X = torch.tensor(images, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = kanModel(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

while True:
    index = int(input("Enter a number (0 - 49): "))
    if index < 0 or index >= 50:
        print("Invalid index, please enter a number between 0 and 49.")
        continue

    input_data = X[index].unsqueeze(0)  # Add a batch dimension
    output = kanModel(input_data)
    # whichever index of the list is the highest that corresponds to the answer
    print(f"Predicted output: {output.squeeze().tolist()}")
