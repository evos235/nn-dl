import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_function='sigmoid', loss_function='mse'):
        # Initialize the weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))

        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

        # Set the activation function
        self.activation_function = activation_function
        self.loss_function = loss_function

    def forward(self, X):
        # Forward pass through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.apply_activation(self.hidden_input)

        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.apply_activation(self.output_input)

        return self.output

    def apply_activation(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_function == 'relu':
            return self.relu(x)
        elif self.activation_function == 'tanh':
            return self.tanh(x)
        elif self.activation_function == 'leaky_relu':
            return self.leaky_relu(x)
        elif self.activation_function == 'softmax':
            return self.softmax(x)
        elif self.activation_function == 'softplus':
            return self.softplus(x)
        else:
            raise ValueError(f"Activation function {self.activation_function} not supported.")

    # Activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def softplus(self, x):
        return np.log(1 + np.exp(x))

    # Loss function
    def compute_loss(self, y_true, y_pred):
        if self.loss_function == 'mse':
            return self.mean_squared_error(y_true, y_pred)
        elif self.loss_function == 'cross_entropy':
            return self.cross_entropy_loss(y_true, y_pred)
        else:
            raise ValueError(f"Loss function {self.loss_function} not supported.")

    # Mean Squared Error (MSE)
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Cross-Entropy Loss (for classification)
    def cross_entropy_loss(self, y_true, y_pred):
        # To avoid log(0), we add a small epsilon value
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Example usage:
input_size = 3    # Number of input neurons
hidden_size = 5   # Number of neurons in the hidden layer
output_size = 2   # Number of output neurons

# Create a neural network instance
nn = NeuralNetwork(input_size, hidden_size, output_size, activation_function='sigmoid', loss_function='mse')

# Example input data (for a batch of 2 samples with 3 features each)
X_input = np.array([[0.1, 0.5, 0.9], [0.2, 0.8, 0.6]])

# Actual target values (e.g., for regression, use continuous values)
y_true = np.array([[0.5, 0.2], [0.7, 0.3]])

# Perform a forward pass with the input data
output = nn.forward(X_input)

# Calculate the loss using the true values and the predicted output
loss = nn.compute_loss(y_true, output)

print("\nPredicted output:")
print(output)
print("\nLoss (MSE):")
print(loss)

