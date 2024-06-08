import sys

import matplotlib.pyplot as plt
import numpy as np

from models.utils import initialize_weights_biases_mlp


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
        log_file = open('mlp_logs.txt', 'w')
        original_stdout = sys.stdout

        try:
            sys.stdout = log_file
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
        finally:
            sys.stdout = original_stdout
            log_file.close()

    def predict(self, img):
        img.shape += (1,)
        # Forward propagation
        activations = [img]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            activation = self.sigmoid(z)
            activations.append(activation)
        return activations[-1].argmax()
