import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

#Load Dataset 
print("Loading MNIST...")

mnist = fetch_openml('mnist_784', as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

# Normalize
X = X / 255.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training started...")

#Hyperparameters
input_size = 784
hidden_size = 128
output_size = 10

learning_rate = 0.01
epochs = 10
batch_size = 64

#Initialize Weights
np.random.seed(42)

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

#Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

#One-hot Encoding
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

#Training Loop
loss_history = []
accuracy_history = []

for epoch in range(epochs):

    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    for i in range(0, X_train.shape[0], batch_size):

        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        y_batch_onehot = one_hot(y_batch, output_size)

        # Forward pass
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)

        Z2 = np.dot(A1, W2) + b2
        y_pred = softmax(Z2)

        # Backpropagation
        m = X_batch.shape[0]

        dZ2 = y_pred - y_batch_onehot
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X_batch.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    #  End of Epoch 

    # Full training evaluation
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    y_pred = softmax(Z2)

    y_train_onehot = one_hot(y_train, output_size)

    loss = -np.mean(np.sum(y_train_onehot * np.log(y_pred + 1e-8), axis=1))
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y_train)

    loss_history.append(loss)
    accuracy_history.append(accuracy)

    print(f"Epoch {epoch+1}/{epochs}  Loss: {loss:.4f}  Accuracy: {accuracy:.4f}")

#Test Accuracy
Z1 = np.dot(X_test, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2
y_pred = softmax(Z2)

test_predictions = np.argmax(y_pred, axis=1)
test_accuracy = np.mean(test_predictions == y_test)

print("\nFinal Test Accuracy:", test_accuracy)

#Plot Results
plt.figure()
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_plot.png")
plt.show()

plt.figure()
plt.plot(accuracy_history)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("accuracy_plot.png")

plt.show()
