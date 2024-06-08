import math
import sys
import uuid

import numpy
import torch
from matplotlib import pyplot as plt

from models.mlp import MLP
from torch import nn, optim
from models.kan import InitializeKan


def model_analysis(images, labels, epochs, hidden_layers_mlp, hidden_layers_kan, learning_rate, eye=5):
    # MLP
    hidden_layers_mlp = hidden_layers_mlp
    mlpModel = MLP(hidden_layers_mlp, images, labels)
    # train model
    print("started training,Method - MLP")
    mlpModel.train(epochs, learning_rate=learning_rate)

    # KAN
    input_size = images.shape[1]
    hidden_layers_kan.append(eye)  # last one matches the output layer
    kanModel = InitializeKan(hidden_layers_kan, input_size)
    # define criteria for loss function and learning rate
    criterion = nn.MSELoss()
    optimizer = optim.Adam(kanModel.parameters(), lr=learning_rate)

    # train KAN
    # Convert data to PyTorch tensors
    X = torch.tensor(images, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    print("started training,Method - KAN")
    log_file = open('kan_logs.txt', 'w')
    original_stdout = sys.stdout
    try:
        sys.stdout = log_file
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = kanModel(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    finally:
        sys.stdout = original_stdout
        log_file.close()

    # outputs
    while True:
        # index upper limit depends on the input given to the models
        index = int(input("Enter a number (0 - 49): "))
        if index < 0 or index >= 50:
            print("Invalid index, please enter a number between 0 and 49.")
            continue
        input_data = X[index].unsqueeze(0)  # Add a batch dimension
        output = kanModel(input_data)
        predicted_kan = assign_polygon(numpy.argmax(output.squeeze().tolist()))
        img = images[index]
        predicted_mlp = assign_polygon(mlpModel.predict(img))
        plt.imshow(img.reshape(28, 28), cmap="Greys")
        plt.title(f"KanModel prediction: {predicted_kan}, MLPModel prediction: {predicted_mlp}")
        plt.figtext(0.5, 0.01,
                    f'epochs-{epochs},hidden layers for KAN-{hidden_layers_kan[:-1]},MLP-{hidden_layers_mlp}',
                    ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 3})
        random_string = str(uuid.uuid4())
        plt.savefig(f'results/{random_string}.png')
        plt.show()


def assign_polygon(polygon_id):
    match polygon_id:
        case 0:
            return "triangle"
        case 1:
            return "heptagon"
        case 2:
            return "rombus"
        case 3:
            return "hexagon"
        case 4:
            return "pentagon"
        case _:
            return ""
