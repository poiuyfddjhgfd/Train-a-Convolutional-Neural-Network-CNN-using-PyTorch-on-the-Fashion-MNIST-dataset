Notebook Overview
1. Import Libraries
Essential libraries for data handling, model building, training, and visualization are imported:

pandas, sklearn, torch, matplotlib

2. Set Random Seed and Device
Ensures reproducibility by setting a manual seed.

Checks for GPU availability and sets the device accordingly.

3. Load and Explore Data
Loads the Fashion-MNIST dataset from a CSV file.

Displays the first few rows and visualizes the first 16 images in a 4x4 grid.

4. Data Preprocessing
Splits the data into training and testing sets.

Normalizes pixel values to the range [0, 1].

5. Custom Dataset Class
Defines a CustomDataset class to handle data loading and transformation.

Converts data into PyTorch tensors and reshapes images to (1, 28, 28).

6. Neural Network Model
Defines a custom neural network class MYNN with configurable hidden layers, neurons, and dropout.

The model supports:

Multiple hidden layers with Batch Normalization and ReLU activation.

Dropout for regularization.

Flexible input and output dimensions.

7. Hyperparameter Tuning with Optuna
Uses Optuna to optimize hyperparameters such as:

Number of hidden layers

Neurons per layer

Learning rate, batch size, optimizer, etc.

The objective function defines the trial parameters and model training process.

Key Features
Data Visualization
Displays sample images from the dataset with their corresponding labels.

Custom Dataset Handling
Efficiently loads and preprocesses data using PyTorch's Dataset class.

Flexible Model Architecture
The MYNN class allows dynamic configuration of the network architecture.

Automated Hyperparameter Optimization
Leverages Optuna to find the best hyperparameters for the model.

Usage
Run the notebook cells sequentially to load data, define the model, and start training.

Adjust hyperparameters manually or use Optuna for automated tuning.

Visualize results using the provided plotting functions.

Requirements
Python 3.x

PyTorch

Pandas

Scikit-learn

Matplotlib

Optuna

Install the required packages using:

bash
pip install torch pandas scikit-learn matplotlib optuna
Notes
The notebook is designed to run on GPU for faster training (if available).

The Fashion-MNIST dataset is used for multi-class classification (10 classes).

The model includes techniques like Batch Normalization and Dropout to improve generalization.

This notebook serves as a comprehensive guide to building a CNN in PyTorch, from data loading to hyperparameter optimization, making it suitable for both beginners and experienced practitioners.

