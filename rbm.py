import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

print("Loading MNIST...")

# ===============================
# 1. Load and Prepare Data
# ===============================
(X_train, _), (X_test, _) = mnist.load_data()

# Normalize and flatten
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Convert to binary (RBM works better with binary units)
X_train = (X_train > 0.5).astype(np.float32)
X_test = (X_test > 0.5).astype(np.float32)

print("Data Loaded Successfully!")

# ===============================
# 2. RBM Parameters
# ===============================
visible_units = 784
hidden_units = 256
learning_rate = 0.05
epochs = 20
batch_size = 64

# Initialize weights
W = np.random.normal(0, 0.01, (visible_units, hidden_units))
bv = np.zeros((1, visible_units))
bh = np.zeros((1, hidden_units))

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ===============================
# 3. Training (Contrastive Divergence - CD1)
# ===============================
print("Training RBM...")

for epoch in range(epochs):
    np.random.shuffle(X_train)
    loss = 0
    
    for i in range(0, X_train.shape[0], batch_size):
        v0 = X_train[i:i+batch_size]

        # Positive phase
        h_prob = sigmoid(np.dot(v0, W) + bh)
        h_state = (h_prob > np.random.rand(*h_prob.shape)).astype(np.float32)

        # Negative phase
        v_prob = sigmoid(np.dot(h_state, W.T) + bv)
        v_state = (v_prob > np.random.rand(*v_prob.shape)).astype(np.float32)

        h_prob_neg = sigmoid(np.dot(v_state, W) + bh)

        # Update weights and biases
        W += learning_rate * (
            np.dot(v0.T, h_prob) - np.dot(v_state.T, h_prob_neg)
        ) / batch_size

        bv += learning_rate * np.mean(v0 - v_state, axis=0, keepdims=True)
        bh += learning_rate * np.mean(h_prob - h_prob_neg, axis=0, keepdims=True)

        loss += np.mean((v0 - v_prob) ** 2)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

print("Training Complete!")

# ===============================
# 4. Testing Reconstruction
# ===============================
print("Testing Reconstruction...")

test_sample = X_test[:5]

h = sigmoid(np.dot(test_sample, W) + bh)
v_reconstructed = sigmoid(np.dot(h, W.T) + bv)

# ===============================
# 5. Plot Results
# ===============================
plt.figure(figsize=(10,4))

for i in range(5):
    # Original
    plt.subplot(2, 5, i+1)
    plt.imshow(test_sample[i].reshape(28,28), cmap='gray')
    plt.axis('off')

    # Reconstructed
    plt.subplot(2, 5, i+6)
    plt.imshow(v_reconstructed[i].reshape(28,28), cmap='gray')
    plt.axis('off')

plt.suptitle("RBM Reconstruction")
plt.show()