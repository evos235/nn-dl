import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))

        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

        # Learning rate for gradient descent
        self.learning_rate = learning_rate

    def forward(self, X):
        # Forward pass: Calculate the activations for hidden and output layers
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)

        return self.output

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function
        return x * (1 - x)

    def compute_loss(self, y_true, y_pred):
        # Mean Squared Error loss function
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y_true, y_pred):
        # Backward pass: Calculate the gradients and update weights using gradient descent

        # Calculate output layer error and delta
        output_error = y_pred - y_true
        output_delta = output_error * self.sigmoid_derivative(y_pred)

        # Calculate hidden layer error and delta
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases using the gradients
        self.weights_hidden_output -= self.learning_rate * self.hidden_output.T.dot(output_delta)
        self.bias_output -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weights_input_hidden -= self.learning_rate * X.T.dot(hidden_delta)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y_true, epochs=10000):
        # Train the neural network using forward and backward propagation
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y_true, y_pred)

            # Backward pass (backpropagation)
            self.backward(X, y_true, y_pred)

            # Print loss every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} | Loss: {loss}")

    def test(self, X):
        # Test the neural network with the input data
        return self.forward(X)

# Example usage for the XOR problem:
# Input data for XOR (4 examples, 2 features each)
X_input = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

# Output data for XOR (4 examples, 1 output each)
y_true = np.array([[0], [1], [1], [0]])

# Create a neural network instance
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)

# Train the neural network
print("Training the neural network...")
nn.train(X_input, y_true, epochs=10000)

# After training, test the neural network on the XOR input
print("\nTesting the neural network on the XOR input:")
y_pred = nn.test(X_input)

print("\nPredicted output after training:")
print(y_pred)

