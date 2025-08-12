import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize the weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))

        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

        # Learning rate for gradient descent
        self.learning_rate = learning_rate

    def forward(self, X):
        # Forward pass through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)

        return self.output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function
        return x * (1 - x)

    def compute_loss(self, y_true, y_pred):
        # Mean Squared Error (MSE) loss function
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y_true, y_pred):
        # Calculate the loss gradient with respect to the output layer
        output_error = y_pred - y_true
        output_delta = output_error * self.sigmoid_derivative(y_pred)

        # Calculate the loss gradient with respect to the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases using the gradients
        self.weights_hidden_output -= self.learning_rate * self.hidden_output.T.dot(output_delta)
        self.bias_output -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weights_input_hidden -= self.learning_rate * X.T.dot(hidden_delta)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y_true, epochs=1000):
        # Train the neural network using forward and backward propagation
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y_true, y_pred)

            # Backward pass (backpropagation)
            self.backward(X, y_true, y_pred)

            # Print the loss at each epoch
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss}")

# Example usage:
input_size = 3    # Number of input neurons
hidden_size = 5   # Number of neurons in the hidden layer
output_size = 1   # Number of output neurons (for regression, it can be 1)

# Create the neural network instance
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.01)

# Example input data (for a batch of 4 samples with 3 features each)
X_input = np.array([[0.1, 0.5, 0.9],
                    [0.2, 0.8, 0.6],
                    [0.9, 0.1, 0.4],
                    [0.5, 0.3, 0.8]])

# Actual target values (e.g., for regression)
y_true = np.array([[0.5], [0.7], [0.2], [0.9]])

# Train the neural network
nn.train(X_input, y_true, epochs=1000)

# After training, let's test the network with the same input data
y_pred = nn.forward(X_input)

print("\nPredicted output after training:")
print(y_pred)


