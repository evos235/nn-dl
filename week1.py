

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights and biases for the layers

        # Input to hidden layer weights (random initialization)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        # Hidden layer biases
        self.bias_hidden = np.zeros((1, hidden_size))

        # Hidden to output layer weights (random initialization)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        # Output layer biases
        self.bias_output = np.zeros((1, output_size))

        # Print the initialized weights and biases
        print("Initialized weights and biases:")
        print("Weights (Input -> Hidden):\n", self.weights_input_hidden)
        print("Biases (Hidden Layer):\n", self.bias_hidden)
        print("Weights (Hidden -> Output):\n", self.weights_hidden_output)
        print("Biases (Output Layer):\n", self.bias_output)

    def forward(self, X):
        # Forward pass through the network

        # Input to hidden layer activation (using sigmoid for example)
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Hidden to output layer activation
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)

        return self.output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage:
input_size = 3    # Number of input neurons
hidden_size = 5   # Number of neurons in the hidden layer
output_size = 2   # Number of output neurons

# Create the neural network instance
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Example input data (for a batch of 2 samples with 3 features each)
X_input = np.array([[0.1, 0.5, 0.9], [0.2, 0.8, 0.6]])

# Perform a forward pass with the input data
output = nn.forward(X_input)

print("\nOutput of the neural network for the input data:")
print(output)
