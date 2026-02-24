             Neural Network + Autoencoder + RBM Project



This project contains implementation of a Neural Network (MLP), Autoencoder and RBM using Python and NumPy. The models are trained on the MNIST dataset for digit classification and reconstruction.



Files included:


1.mlp.py

This file contains a 2-layer neural network built from scratch.

It includes forward pass, backpropagation and SGD update.

ReLU is used in hidden layer and Softmax in output layer.


2.autoencoder.py

This file contains an undercomplete autoencoder.

It is trained to reconstruct the input images.

Reconstruction error is used for checking performance and simple outlier detection.


3.rbm.py

This file contains Restricted Boltzmann Machine implementation.

It is trained on MNIST and shows reconstructed outputs.


--->Report.pdf

Contains explanation of models, hyperparameters used, training graphs, and observations.



Requirements:


-Python 3.x

-NumPy

-Matplotlib




To install libraries:

pip install numpy matplotlib



How to run:


Open terminal in this folder and run:


python mlp.py

python autoencoder.py

python rbm.py



Each file will train the model and display results like loss, accuracy or reconstruction output.



Dataset:

MNIST dataset is loaded inside the code.





