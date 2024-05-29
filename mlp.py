import numpy as np
import matplotlib.pyplot as plt

from utils import *


class MLP:
    def __init__(self, hidden_layers, input_data, data_labels):
        self.hidden_layers = hidden_layers
        self.input_data = input_data
        self.data_labels = data_labels
        input_size = self.input_data.shape[1]
        output_size = self.data_labels.shape[1]
        self.weights, self.biases = initialize_weights_biases_mlp(self.hidden_layers, input_size, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, epochs, learning_rate=0.01):
        correct_guesses = 0

        for epoch in range(epochs):
            for img, l in zip(self.input_data, self.data_labels):
                img = img.reshape(-1, 1)
                l = l.reshape(-1, 1)

                # Forward propagation
                activations = [img]
                for w, b in zip(self.weights, self.biases):
                    z = np.dot(w, activations[-1]) + b
                    activation = self.sigmoid(z)
                    activations.append(activation)

                # Cost / Error calculation
                output = activations[-1]
                error = 1 / len(output) * np.sum((output - l) ** 2, axis=0)
                correct_guesses += int(np.argmax(output) == np.argmax(l))

                # Backpropagation
                delta = output - l
                for i in reversed(range(len(self.weights))):
                    delta_w = np.dot(delta, activations[i].T)
                    delta_b = delta
                    self.weights[i] -= learning_rate * delta_w
                    self.biases[i] -= learning_rate * delta_b

                    if i != 0:
                        delta = np.dot(self.weights[i].T, delta) * self.sigmoid_derivative(activations[i])

            # Show accuracy for this epoch
            accuracy = round((correct_guesses / self.input_data.shape[0]) * 100, 2)
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}%")
            correct_guesses = 0

    def test_single_image(self):
        while True:
            index = int(input("Enter a number (0 - 59999): "))
            img = self.input_data[index]
            plt.imshow(img.reshape(28, 28), cmap="Greys")

            img.shape += (1,)
            # Forward propagation
            activations = [img]
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w, activations[-1]) + b
                activation = self.sigmoid(z)
                activations.append(activation)

            # Output result
            output = activations[-1]
            plt.title(f"Its  {output.argmax()}")
            plt.show()


# Example usage (assuming images and labels are already loaded):
hidden_layers = [20]  # Example hidden layers
input_data, data_labels = load_images()

nn = MLP(hidden_layers, input_data, data_labels)
nn.train(epochs=50, learning_rate=0.01)
nn.test_single_image()
