import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

print("Loading MNIST...")

# Load MNIST
(X_train, _), (X_test, _) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

print("Training Autoencoder...")

# Architecture sizes
input_size = 784
hidden1_size = 128
bottleneck_size = 64

learning_rate = 0.01
epochs = 10
batch_size = 128

# Initialize weights
W1 = np.random.randn(input_size, hidden1_size) * 0.01
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, bottleneck_size) * 0.01
b2 = np.zeros((1, bottleneck_size))

W3 = np.random.randn(bottleneck_size, hidden1_size) * 0.01
b3 = np.zeros((1, hidden1_size))

W4 = np.random.randn(hidden1_size, input_size) * 0.01
b4 = np.zeros((1, input_size))


# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Training loop
for epoch in range(epochs):
    total_loss = 0

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]

        # Forward pass
        Z1 = X_batch.dot(W1) + b1
        A1 = relu(Z1)

        Z2 = A1.dot(W2) + b2
        A2 = relu(Z2)

        Z3 = A2.dot(W3) + b3
        A3 = relu(Z3)

        Z4 = A3.dot(W4) + b4
        output = sigmoid(Z4)

        # Loss (MSE)
        loss = np.mean((X_batch - output) ** 2)
        total_loss += loss

        # Backpropagation
        dZ4 = (output - X_batch) * output * (1 - output)
        dW4 = A3.T.dot(dZ4)
        db4 = np.sum(dZ4, axis=0, keepdims=True)

        dA3 = dZ4.dot(W4.T)
        dZ3 = dA3 * relu_derivative(Z3)
        dW3 = A2.T.dot(dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3.dot(W3.T)
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2.dot(W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = X_batch.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update
        W4 -= learning_rate * dW4
        b4 -= learning_rate * db4
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


print("Testing Reconstruction...")

# Test reconstruction
Z1 = X_test.dot(W1) + b1
A1 = relu(Z1)
Z2 = A1.dot(W2) + b2
A2 = relu(Z2)
Z3 = A2.dot(W3) + b3
A3 = relu(Z3)
Z4 = A3.dot(W4) + b4
reconstructed = sigmoid(Z4)

# Show original vs reconstructed
n = 5
plt.figure(figsize=(10,4))
for i in range(n):
    # Original
    plt.subplot(2, n, i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')
    plt.axis('off')

    # Reconstructed
    plt.subplot(2, n, i+n+1)
    plt.imshow(reconstructed[i].reshape(28,28), cmap='gray')
    plt.axis('off')

plt.show()