# Multi-Layer Perceptron (MLP) Implementation in C

### Overview

This project implements a simple **Multi-Layer Perceptron (MLP)** neural network in C to solve the XOR problem. The code includes forward propagation, backpropagation, and weight updates to train the MLP using gradient descent. The network is designed with one hidden layer and is flexible enough to handle different activation functions.

---

## Table of Contents

1. [Introduction to MLP](#introduction-to-mlp)
2. [Backpropagation Explanation](#backpropagation-explanation)
3. [Code Structure](#code-structure)
4. [Setup and Usage](#setup-and-usage)
5. [References](#references)

---

## Introduction to MLP

A **Multi-Layer Perceptron (MLP)** is a type of feedforward neural network consisting of at least three layers:
- **Input layer**: Receives the input features.
- **Hidden layer**: Consists of neurons that learn to represent complex patterns from the input data.
- **Output layer**: Produces predictions based on the learned patterns.

In this implementation:
- The MLP consists of **one hidden layer** with **4 neurons**.
- The **activation function** used for both the hidden and output layers is a sigmoid for simplicity.

The **XOR problem** is used as the example dataset. This is a classical binary classification problem where the goal is to predict the XOR relationship between two binary inputs.

---

## Backpropagation Explanation

**Backpropagation** is the algorithm used to train the MLP by updating the weights and biases to minimize the error between the predicted and expected outputs. Here's a breakdown of the forward and backward passes:

### 1. **Forward Propagation**:
   In forward propagation, the network computes the output for a given input by passing data through the hidden layer and then the output layer.

   For each layer:
   - The **weighted sum of inputs** is calculated.
   - An **activation function** is applied to produce the output.

   Formally, for a given layer:
   ![z = W \cdot x + b](https://latex.codecogs.com/svg.latex?z%20%3D%20W%20%5Ccdot%20x%20%2B%20b)

   where \(W\) is the weight matrix, \(x\) is the input, and \(b\) is the bias vector.

   The result is then passed through an activation function:
   ![y = \sigma(z)](https://latex.codecogs.com/svg.latex?y%20%3D%20%5Csigma%28z%29)

   where \(\sigma\) is the activation function (ReLU or sigmoid in this case).

### 2. **Backpropagation**:
   Backpropagation is used to adjust the weights and biases based on the error between the predicted and expected outputs. The total loss is calculated using the **Mean Squared Error (MSE)** loss function:
   
   ![\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{pred},i} - y_{\text{true},i})^2](https://latex.codecogs.com/svg.latex?%5Ctext%7BMSE%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%28y_%7B%5Ctext%7Bpred%7D%2Ci%7D%20-%20y_%7B%5Ctext%7Btrue%7D%2Ci%7D%29%5E2)

   **Backpropagation Steps**:
   - **Step 1**: Compute the error at the output layer:
     ![\delta_{\text{output}} = (y_{\text{pred}} - y_{\text{true}}) \times \sigma'(z_{\text{output}})](https://latex.codecogs.com/svg.latex?%5Cdelta_%7B%5Ctext%7Boutput%7D%7D%20%3D%20%28y_%7B%5Ctext%7Bpred%7D%7D%20-%20y_%7B%5Ctext%7Btrue%7D%7D%29%20%5Ctimes%20%5Csigma%27%28z_%7B%5Ctext%7Boutput%7D%7D%29)
     
     where ![\sigma'(z_{\text{output}})](https://latex.codecogs.com/svg.latex?%5Csigma%27%28z_%7B%5Ctext%7Boutput%7D%7D%29) is the derivative of the output layer’s activation function (e.g., sigmoid).
   
   - **Step 2**: Compute the error at the hidden layer:
     ![\delta_{\text{hidden}} = (\delta_{\text{output}} \cdot W_{\text{output}}) \times \sigma'(z_{\text{hidden}})](https://latex.codecogs.com/svg.latex?%5Cdelta_%7B%5Ctext%7Bhidden%7D%7D%20%3D%20%28%5Cdelta_%7B%5Ctext%7Boutput%7D%7D%20%5Ccdot%20W_%7B%5Ctext%7Boutput%7D%7D%29%20%5Ctimes%20%5Csigma%27%28z_%7B%5Ctext%7Bhidden%7D%7D%29)
   
   - **Step 3**: Update the weights and biases using gradient descent:

     ![W_{\text{new}} = W_{\text{old}} - \alpha \cdot \delta \cdot x](https://latex.codecogs.com/svg.latex?W_%7B%5Ctext%7Bnew%7D%7D%20%3D%20W_%7B%5Ctext%7Bold%7D%7D%20-%20%5Calpha%20%5Ccdot%20%5Cdelta%20%5Ccdot%20x)
     
     ![b_{\text{new}} = b_{\text{old}} - \alpha \cdot \delta](https://latex.codecogs.com/svg.latex?b_%7B%5Ctext%7Bnew%7D%7D%20%3D%20b_%7B%5Ctext%7Bold%7D%7D%20-%20%5Calpha%20%5Ccdot%20%5Cdelta)
     
     where ![\alpha](https://latex.codecogs.com/svg.latex?%5Calpha) is the learning rate, ![\delta](https://latex.codecogs.com/svg.latex?%5Cdelta) is the error term (computed earlier), and \(x\) is the input to the layer.

   These steps are repeated for each epoch until the loss converges.

---

## Code Structure

The project is divided into several functions to handle different aspects of the MLP, including initialization, forward propagation, backpropagation, and training.

- **`initialize_layer(LinearLayer *layer, int input_size, int output_size)`**: Initializes weights and biases for a layer.
- **`forward_propagation(LinearLayer *layer, double inputs[], double outputs[], double (*activation_func)(double))`**: Performs forward propagation through one layer.
- **`backpropagation(NeuralNetwork *nn, double inputs[], double hidden_outputs[], double output_outputs[], double errors[], double *delta_hidden, double *delta_output)`**: Computes the gradients for weights and biases using backpropagation.
- **`update_weights_biases(LinearLayer *layer, double inputs[], double deltas[])`**: Updates the weights and biases of a layer based on computed deltas.
- **`train(NeuralNetwork *nn, const double inputs[][NUM_INPUTS], const double expected_outputs[][NUM_OUTPUTS])`**: Trains the MLP using the XOR dataset.
- **`test(NeuralNetwork *nn, const double inputs[][NUM_INPUTS], const double expected_outputs[][NUM_OUTPUTS])`**: Tests the trained MLP.

---

## Setup and Usage

### **Prerequisites**
- This code is written in C and requires a basic C compiler to run.

### **Compilation**

Use the following command to compile the program:

```bash
gcc -o mlp mlp_xor.c -lm
```

There is also a Makefile for your convenience allowing you to compile with 

```bash
make compile
```

### **Running**

```
./mlp
```

This should give the expected output 

```
Epoch 1000, Error: 0.250322
Epoch 2000, Error: 0.250300
Epoch 3000, Error: 0.250277
Epoch 4000, Error: 0.250254
Epoch 5000, Error: 0.250231

....

Epoch 998000, Error: 0.000115
Epoch 999000, Error: 0.000115
Epoch 1000000, Error: 0.000115

Testing the trained network:
Input: 0.0, 0.0, Expected Output: 0.0, Predicted Output: 0.011
Input: 0.0, 1.0, Expected Output: 1.0, Predicted Output: 0.989
Input: 1.0, 0.0, Expected Output: 1.0, Predicted Output: 0.989
Input: 1.0, 1.0, Expected Output: 0.0, Predicted Output: 0.010
```

## References
![Hidden Layer By Hand](https://aibyhand.substack.com/p/w8-hidden-layer)
![Backpropagation By Hand](https://aibyhand.substack.com/p/7-can-you-calculate-a-transformer?utm_source=publication-search)
![Backpropagation](https://www.youtube.com/watch?v=tIeHLnjs5U8)
![Neural Networks From Scratch - Python](https://www.kaggle.com/code/ancientaxe/simple-neural-network-from-scratch-in-python)