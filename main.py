"""
Linear Regression Module

This module implements a simple linear regression model using TensorFlow.
It generates synthetic data, trains a linear model, and visualizes the results.
The implementation includes data normalization to prevent numerical instability.
"""

import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for consistent performance
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 5) Implement python code for hypothesis, cost function, and optimizer

# Hypothesis (prediction)
def hypothesis(W, x, b):
    """
    Compute the linear regression hypothesis (prediction).

    Args:
        W (tf.Variable): The weight parameter of the linear model.
        x (tf.Tensor): Input feature values.
        b (tf.Variable): The bias parameter of the linear model.

    Returns:
        tf.Tensor: Predicted values using the linear equation y = W*x + b.
    """
    return W * x + b

# Cost function (Mean Squared Error)
def cost_function(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) cost function.

    Args:
        y_true (tf.Tensor): True target values.
        y_pred (tf.Tensor): Predicted values from the model.

    Returns:
        tf.Tensor: Mean squared error between true and predicted values.
    """
    return tf.reduce_mean(tf.square(y_pred - y_true))

def optimizer_init(learning_rate):
    """
    Initialize the Stochastic Gradient Descent (SGD) optimizer.

    Args:
        learning_rate (float): The learning rate for gradient updates.

    Returns:
        tf.optimizers.SGD: Configured SGD optimizer instance.
    """
    return tf.optimizers.SGD(learning_rate)


def main():
    """
    Main function that executes the linear regression training pipeline.

    This function performs the following steps:
    1. Sets random seeds for reproducibility
    2. Generates synthetic linear data with noise
    3. Plots the training data
    4. Normalizes the data to prevent numerical instability
    5. Initializes model parameters (weights and bias)
    6. Trains the linear regression model using gradient descent
    7. Prints final training results
    8. Plots the fitted regression line

    The function uses TensorFlow for automatic differentiation and optimization.
    """
    # Set seeds for reproducibility
    np.random.seed(101)
    tf.random.set_seed(101)

    # Generate random linear data
    x = np.linspace(0, 50, 50)
    y = np.linspace(0, 50, 50)

    # Add noise
    x += np.random.uniform(-4, 4, 50)
    y += np.random.uniform(-4, 4, 50)

    # 1) Plot the training data
    plt.scatter(x, y, label="Training Data")
    plt.figure("Training Data")
    plt.title("Training Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

    # 2) Create TensorFlow X and Y tensors
    X = tf.constant(x, dtype=tf.float32)
    Y = tf.constant(y, dtype=tf.float32)

    # Normalize data to prevent NaN values during training
    # I had to implement normalization to prevent NaN values during training,
    # which can occur when the cost function becomes too large. Normalization
    # helps to keep the values in a manageable range, ensuring stable training.
    X_mean, X_std = tf.reduce_mean(X), tf.math.reduce_std(X)
    Y_mean, Y_std = tf.reduce_mean(Y), tf.math.reduce_std(Y)
    X_normalized = (X - X_mean) / (X_std + 1e-7)
    Y_normalized = (Y - Y_mean) / (Y_std + 1e-7)

    # 3) Initialize trainable variables (weights and bias)
    W = tf.Variable(np.random.randn(), dtype=tf.float32)
    b = tf.Variable(np.random.randn(), dtype=tf.float32)

    # 4) Hyperparameters
    learning_rate = 0.01
    training_epochs = 1000

    # Optimizer
    optimizer = optimizer_init(learning_rate)

    # 6) Implement the training process in TensorFlow
    for epoch in range(training_epochs):
        with tf.GradientTape() as tape:
            y_pred = hypothesis(W, X_normalized, b)
            cost = cost_function(Y_normalized, y_pred)

        # Compute gradients
        gradients = tape.gradient(cost, [W, b])

        # Apply gradients
        optimizer.apply_gradients(zip(gradients, [W, b]))

        # Print progress occasionally
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Training Cost = {cost.numpy():.4f}, Weight (W) = {W.numpy():.4f}, Bias (b) = {b.numpy():.4f}")

    # 7) Print out the results for the training cost, weight, and bias
    print("\nTraining complete!")
    print(f"Final Training Cost: {cost.numpy():.4f}")
    print(f"Final Weight (W): {W.numpy():.4f}")
    print(f"Final Bias (b): {b.numpy():.4f}")

    # 8) Plot fitted line
    y_pred_normalized = hypothesis(W, X_normalized, b)
    y_pred_original = y_pred_normalized * Y_std + Y_mean  # Transform back to original scale
    plt.scatter(x, y, label="Original Data")
    plt.plot(x, y_pred_original, color='red', label="Fitted Line")
    plt.figure("Linear Regression Fit")
    plt.title("Linear Regression Fit")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
