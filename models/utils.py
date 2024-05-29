import numpy as np
from PIL import Image


def initialize_weights_biases_mlp(hidden_layers, input_size, output_size):
    weights = []
    biases = []

    # First hidden layer weights and biases
    w = np.random.uniform(-0.5, 0.5, (hidden_layers[0], input_size))
    weights.append(w)
    b = np.zeros((hidden_layers[0], 1))
    biases.append(b)

    # Hidden layers weights and biases
    for i in range(1, len(hidden_layers)):
        w = np.random.uniform(-0.5, 0.5, (hidden_layers[i], hidden_layers[i - 1]))
        weights.append(w)
        b = np.zeros((hidden_layers[i], 1))
        biases.append(b)

    # Output layer weights and biases
    w = np.random.uniform(-0.5, 0.5, (output_size, hidden_layers[-1]))
    weights.append(w)
    b = np.zeros((output_size, 1))
    biases.append(b)

    return weights, biases


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def load_images(eye,dir):
    """

    :param eye: output layer
    :return:
    """
    images = []
    labels = []
    # Loop through the range of values for i and j
    for i in range(5):  # i from 0 to 4
        for j in range(10):  # j from 0 to 9
            # Construct the file path for the image
            filename = f"data/{dir}/{i}p{j}.png"
            try:
                # Load the image and append it to the list
                image = np.array(Image.open(filename).convert("L"), dtype="float32") / 255
                image = image.flatten()  # Flatten the image to a 1D array
                images.append(image)
                labels.append(i)
            except FileNotFoundError:
                print(f"File {filename} not found.")

    return np.array(images), np.eye(eye)[labels]
