import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights and bias
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def forward(self, X):
        # non Linear combination + step function
        non_linear_output = np.dot(X, self.weights) + self.bias
        return self.step_function(non_linear_output)

    def step_function(self, x):
        # Binary step activation function
        return np.where(x >= 0, 1, 0)

    def train(self, X, y, epochs=100):
        print("Training perceptron for non-linearly separable problem...")

        for epoch in range(epochs):
            total_error = 0

            for i in range(len(X)):
                # Forward pass
                prediction = self.forward(X[i])

                # Calculate error
                error = y[i] - prediction
                total_error += abs(error)

                # Update weights and bias (Perceptron Learning Rule)
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

            # Stop early if converged
            if total_error == 0:
                print(f"Converged at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Total Error: {total_error}")

    def predict(self, X):
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
nn = Perceptron(input_size=2, learning_rate=0.1)

# Train the neural network
print("Training the perceptron...")
nn.train(X_input, y_true.flatten(), epochs=10000)

# After training, test the neural network on the XOR input
print("\nTesting the perceptron on the XOR input:")
y_pred = nn.predict(X_input)
print(y_pred)
