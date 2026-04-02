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

# 3) Initialize trainable variables (weights and bias)
W = tf.Variable(np.random.randn(), dtype=tf.float32)
b = tf.Variable(np.random.randn(), dtype=tf.float32)

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
        y_pred = hypothesis(X)
        cost = cost_function(Y, y_pred)
    
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
plt.scatter(x, y, label="Original Data")
plt.plot(x, hypothesis(X), color='red', label="Fitted Line")
plt.title("Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
