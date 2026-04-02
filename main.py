import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for consistent performance
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(101)
tf.random.set_seed(101)

# Generate random linear data
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

# Add noise
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

n = len(x)

# 1) Plot the training data
plt.scatter(x, y, label="Training Data")
plt.title("Training Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Convert to TensorFlow tensors
X = tf.constant(x, dtype=tf.float32)
Y = tf.constant(y, dtype=tf.float32)

# Normalize data to prevent NaN values during training
X_mean, X_std = tf.reduce_mean(X), tf.math.reduce_std(X)
Y_mean, Y_std = tf.reduce_mean(Y), tf.math.reduce_std(Y)
X_normalized = (X - X_mean) / (X_std + 1e-7)
Y_normalized = (Y - Y_mean) / (Y_std + 1e-7)

# 3) Initialize trainable variables (weights and bias)
W = tf.Variable(0.0, dtype=tf.float32)
b = tf.Variable(0.0, dtype=tf.float32)

# 4) Hyperparameters
learning_rate = 0.01
training_epochs = 1000

# 5) Define model components

# Hypothesis (prediction)
def hypothesis(x):
    return W * x + b

# Cost function (Mean Squared Error)
def cost_function(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# 6) Training process
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        y_pred = hypothesis(X_normalized)
        cost = cost_function(Y_normalized, y_pred)
    
    # Compute gradients
    gradients = tape.gradient(cost, [W, b])
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    # Print progress occasionally
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: Cost = {cost.numpy():.4f}, W = {W.numpy():.4f}, b = {b.numpy():.4f}")

# 7) Final results
print("\nTraining complete!")
print(f"Final Cost: {cost.numpy():.4f}")
print(f"Final Weight (W): {W.numpy():.4f}")
print(f"Final Bias (b): {b.numpy():.4f}")

# 8) Plot fitted line
y_pred_normalized = hypothesis(X_normalized)
y_pred_original = y_pred_normalized * Y_std + Y_mean  # Transform back to original scale
plt.scatter(x, y, label="Original Data")
plt.plot(x, y_pred_original, color='red', label="Fitted Line")
plt.title("Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
