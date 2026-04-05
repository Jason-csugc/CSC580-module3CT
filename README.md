# CSC580-module3CT

CSC580 Module 3 Critical Thinking

## Product Overview

This repository contains a TensorFlow-based linear regression implementation and analysis for CSC580 Module 3 (Critical Thinking). It includes utilities for generating synthetic data, training/evaluating a linear model, implementing data normalization, and visualizing training results.

## Features

• `hypothesis()`: compute linear regression predictions
• `cost_function()`: calculate Mean Squared Error loss
• `optimizer_init()`: initialize SGD optimizer
• `main()`: orchestrate complete training pipeline
• Data normalization to prevent numerical instability
• Training progress visualization and final results plotting
• Experimentation with learning rates and training epochs

## Getting Started

1. Create and activate virtual environment:  python3 -m venv mod3ct
source mod3ct/bin/activate
pip install -r requirements.txt
2. Run the project:  python main.py

## Notes

• Designed for linear regression on synthetic 1D data with noise.
• Provides structured evaluation for coursework questions.
• GUI / plotting uses `matplotlib` for data and regression line visualization.

## Outputs

1. Command line results

```
Epoch 100: Training Cost = 0.1234, Weight (W) = 0.9876, Bias (b) = 0.5432
Epoch 200: Training Cost = 0.0567, Weight (W) = 0.9765, Bias (b) = 0.4321
...
Training complete!
Final Training Cost: 0.0123
Final Weight (W): 0.9654
Final Bias (b): 0.3210
```

2. Images of training data and fitted regression line

## Additional Links

- [Code](https://github.com/Jason-csugc/CSC580-module3CT)
- [Issues](https://github.com/Jason-csugc/CSC580-module3CT/issues)
- [Pull requests](https://github.com/Jason-csugc/CSC580-module3CT/pulls)
- [Actions](https://github.com/Jason-csugc/CSC580-module3CT/actions)
- [Projects](https://github.com/Jason-csugc/CSC580-module3CT/projects)
- [Security and quality](https://github.com/Jason-csugc/CSC580-module3CT/security)