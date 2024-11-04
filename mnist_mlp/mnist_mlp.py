import math
import random
import time

# Constants
NUM_INPUTS = 784       # 28x28 pixels
NUM_HIDDEN = 128       # Number of hidden neurons
NUM_OUTPUTS = 10       # Digits 0-9
LEARNING_RATE = 0.01   # Learning rate
EPOCHS = 10            # Number of training epochs
BATCH_SIZE = 64        # Mini-batch size

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def softmax(inputs):
    max_input = max(inputs)
    exp_values = [math.exp(i - max_input) for i in inputs]
    total = sum(exp_values)
    return [v / total for v in exp_values]

# Initialize layer
def initialize_layer(input_size, output_size, activation='relu'):
    limit = math.sqrt(6 / (input_size + output_size))
    weights = [[random.uniform(-limit, limit) for _ in range(output_size)] for _ in range(input_size)]
    biases = [0.0 for _ in range(output_size)]
    return weights, biases, activation

# Forward pass for a single layer
def linear_layer_forward(weights, biases, inputs, activation):
    outputs = [sum(w * i for w, i in zip(weights[j], inputs)) + biases[j] for j in range(len(biases))]
    
    if activation == 'sigmoid':
        return [sigmoid(x) for x in outputs]
    elif activation == 'relu':
        return [relu(x) for x in outputs]
    elif activation == 'softmax':
        return softmax(outputs)
    return outputs

# Forward propagation through the network
def forward(hidden_weights, hidden_biases, output_weights, output_biases, x):
    hidden_output = linear_layer_forward(hidden_weights, hidden_biases, x, 'relu')
    output_output = linear_layer_forward(output_weights, output_biases, hidden_output, 'softmax')
    return hidden_output, output_output

# Backpropagation
def backward(hidden_weights, hidden_biases, output_weights, output_biases, x, hidden_output, output_output, expected_output):
    # Compute output layer delta
    delta_output = [output_output[i] - expected_output[i] for i in range(NUM_OUTPUTS)]

    # Compute hidden layer delta
    delta_hidden = [
        sum(delta_output[j] * output_weights[i][j] for j in range(NUM_OUTPUTS)) * relu_derivative(hidden_output[i])
        for i in range(NUM_HIDDEN)
    ]

    # Update output layer weights and biases
    for i in range(NUM_HIDDEN):
        for j in range(NUM_OUTPUTS):
            output_weights[i][j] -= LEARNING_RATE * delta_output[j] * hidden_output[i]
    for j in range(NUM_OUTPUTS):
        output_biases[j] -= LEARNING_RATE * delta_output[j]

    # Update hidden layer weights and biases
    for i in range(NUM_INPUTS):
        for j in range(NUM_HIDDEN):
            hidden_weights[i][j] -= LEARNING_RATE * delta_hidden[j] * x[i]
    for j in range(NUM_HIDDEN):
        hidden_biases[j] -= LEARNING_RATE * delta_hidden[j]

    return hidden_weights, hidden_biases, output_weights, output_biases

# Cross-entropy loss
def cross_entropy_loss(predicted, expected):
    return -sum(expected[i] * math.log(predicted[i] + 1e-9) for i in range(len(expected)))

# Data loading
def read_mnist_images(filename, num_images):
    with open(filename, 'rb') as f:
        f.read(16)  # Skip the header
        data = f.read(num_images * NUM_INPUTS)
        return [[pixel / 255.0 for pixel in data[i * NUM_INPUTS:(i + 1) * NUM_INPUTS]] for i in range(num_images)]

def read_mnist_labels(filename, num_labels):
    with open(filename, 'rb') as f:
        f.read(8)  # Skip the header
        labels = f.read(num_labels)
        return [label for label in labels]

# Training function
def train(hidden_weights, hidden_biases, output_weights, output_biases, images, labels):
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        indices = list(range(len(images)))
        random.shuffle(indices)

        for start in range(0, len(images), BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]
            batch_images = [images[i] for i in batch_indices]
            batch_labels = [labels[i] for i in batch_indices]

            for x, label in zip(batch_images, batch_labels):
                y_true = [1.0 if i == label else 0.0 for i in range(NUM_OUTPUTS)]  # One-hot encode label
                hidden_output, output_output = forward(hidden_weights, hidden_biases, output_weights, output_biases, x)

                # Calculate loss
                loss = cross_entropy_loss(output_output, y_true)
                total_loss += loss

                # Backward pass
                hidden_weights, hidden_biases, output_weights, output_biases = \
                    backward(hidden_weights, hidden_biases, output_weights, output_biases, x, hidden_output, output_output, y_true)

        # Calculate epoch duration
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(images)}, Time: {duration:.2f} seconds")

# Testing function
def test(hidden_weights, hidden_biases, output_weights, output_biases, images, labels):
    correct = 0
    for x, label in zip(images, labels):
        _, output_output = forward(hidden_weights, hidden_biases, output_weights, output_biases, x)
        predicted_label = output_output.index(max(output_output))
        if predicted_label == label:
            correct += 1
    accuracy = correct / len(images) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

# Load MNIST data from binary files
train_images = read_mnist_images('train-images.idx3-ubyte', 60000)
train_labels = read_mnist_labels('train-labels.idx1-ubyte', 60000)
test_images = read_mnist_images('t10k-images.idx3-ubyte', 10000)
test_labels = read_mnist_labels('t10k-labels.idx1-ubyte', 10000)

# Initialize layers
hidden_weights, hidden_biases, _ = initialize_layer(NUM_INPUTS, NUM_HIDDEN, 'relu')
output_weights, output_biases, _ = initialize_layer(NUM_HIDDEN, NUM_OUTPUTS, 'softmax')

# Train and test the neural network
train(hidden_weights, hidden_biases, output_weights, output_biases, train_images, train_labels)
test(hidden_weights, hidden_biases, output_weights, output_biases, test_images, test_labels)