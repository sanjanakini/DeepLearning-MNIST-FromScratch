Project Title
# Deep Learning Models from Scratch using NumPy (MNIST)

Project Description
This project implements core deep learning models from scratch using NumPy:

• Multilayer Perceptron (MLP)
• Dense Autoencoder
• Restricted Boltzmann Machine (RBM)

The models are trained and tested on the MNIST handwritten digit dataset.

DATASET:
Dataset: MNIST Handwritten Digits
Total Samples: 70,000
Classes: 10 (Digits 0–9)
Image Size: 28x28

Models Implemented:
1. MLP (784 → 128 → 10)
   - ReLU + Softmax
   - Cross-Entropy Loss

2. Autoencoder (784 → 128 → 64 → 128 → 784)
   - ReLU + Sigmoid
   - MSE Loss

3. RBM (784 → 256)
   - Contrastive Divergence (CD-1)

Results:
• MLP achieved good classification accuracy.
• Autoencoder reconstructed digits with slight blurring.
• RBM produced sharper reconstructed images.

How to Run:
Install dependencies:
pip install numpy matplotlib scikit-learn keras

Run:
1)python mlp.py
2)python autoencoder.py
3)python rbm.py

Author
Name: Sanjana Kini
Course: Deep Learning (IS3332-1)
Department: Information Science & Engineering
