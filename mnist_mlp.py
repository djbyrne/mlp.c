import math
import time
import struct
import random

# Constants
NUM_INPUTS = 784       # 28x28 pixels
NUM_HIDDEN = 128       # Number of hidden neurons
NUM_OUTPUTS = 10       # Digits 0-9
TRAIN_SAMPLES = 20000  # Number of training samples
TEST_SAMPLES = 5000    # Number of test samples
LEARNING_RATE = 0.01   # Learning rate
EPOCHS = 10            # Number of training epochs
BATCH_SIZE = 64        # Mini-batch size

# Activation Functions
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def relu(x):
    return x if x > 0 else 0

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def softmax(inputs):
    max_input = max(inputs)
    exp_values = [math.exp(i - max_input) for i in inputs]
    sum_exp = sum(exp_values)
    return [val / (sum_exp + 1e-9) for val in exp_values]

# Linear Layer Class
class LinearLayer:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Xavier Initialization
        limit = math.sqrt(6.0 / (input_size + output_size))
        self.weights = [[random.uniform(-limit, limit) for _ in range(output_size)] for _ in range(input_size)]
        self.biases = [0.0 for _ in range(output_size)]
    
    def forward(self, inputs):
        z = []
        for i in range(self.output_size):
            activation_sum = self.biases[i]
            for j in range(self.input_size):
                activation_sum += inputs[j] * self.weights[j][i]
            z.append(activation_sum)
        if self.activation == 'sigmoid':
            a = [sigmoid(val) for val in z]
        elif self.activation == 'relu':
            a = [relu(val) for val in z]
        elif self.activation == 'softmax':
            a = softmax(z)
        else:
            a = z  # No activation
        return a, z

# Neural Network Class
class NeuralNetwork:
    def __init__(self):
        self.hidden_layer = LinearLayer(NUM_INPUTS, NUM_HIDDEN, 'relu')
        self.output_layer = LinearLayer(NUM_HIDDEN, NUM_OUTPUTS, 'softmax')
    
    def forward(self, x):
        hidden_activations, hidden_z = self.hidden_layer.forward(x)
        output_activations, output_z = self.output_layer.forward(hidden_activations)
        return hidden_activations, hidden_z, output_activations, output_z

# Cross-Entropy Loss Function
def cross_entropy_loss(predicted, expected):
    loss = 0.0
    for i in range(len(predicted)):
        loss -= expected[i] * math.log(predicted[i] + 1e-9)
    return loss

# Backward Propagation
def backward(nn, inputs, hidden_outputs, output_outputs, expected_outputs):
    # Output layer delta
    delta_output = [output_outputs[i] - expected_outputs[i] for i in range(NUM_OUTPUTS)]  # For softmax with cross-entropy

    # Hidden layer delta
    delta_hidden = [0.0 for _ in range(NUM_HIDDEN)]
    for i in range(NUM_HIDDEN):
        error = 0.0
        for j in range(NUM_OUTPUTS):
            error += delta_output[j] * nn.output_layer.weights[i][j]
        if nn.hidden_layer.activation == 'sigmoid':
            activation_derivative = sigmoid_derivative(hidden_outputs[i])
        elif nn.hidden_layer.activation == 'relu':
            activation_derivative = relu_derivative(hidden_outputs[i])
        delta_hidden[i] = error * activation_derivative

    # Update weights and biases for output layer
    for i in range(NUM_HIDDEN):
        for j in range(NUM_OUTPUTS):
            nn.output_layer.weights[i][j] -= LEARNING_RATE * delta_output[j] * hidden_outputs[i]
    for i in range(NUM_OUTPUTS):
        nn.output_layer.biases[i] -= LEARNING_RATE * delta_output[i]

    # Update weights and biases for hidden layer
    for i in range(NUM_INPUTS):
        for j in range(NUM_HIDDEN):
            nn.hidden_layer.weights[i][j] -= LEARNING_RATE * delta_hidden[j] * inputs[i]
    for i in range(NUM_HIDDEN):
        nn.hidden_layer.biases[i] -= LEARNING_RATE * delta_hidden[i]

# Training Function
def train(nn, inputs, labels):
    num_samples = len(inputs)
    with open('./logs/training_loss_py.txt', 'w') as loss_file:
        for epoch in range(EPOCHS):
            total_loss = 0.0
            start_time = time.time()
            
            # Shuffle the dataset
            indices = list(range(num_samples))
            random.shuffle(indices)
            shuffled_inputs = [inputs[i] for i in indices]
            shuffled_labels = [labels[i] for i in indices]
            
            # Mini-batch training
            for batch_start in range(0, num_samples, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_samples)
                batch_inputs = shuffled_inputs[batch_start:batch_end]
                batch_labels = shuffled_labels[batch_start:batch_end]
                
                for idx in range(len(batch_inputs)):
                    x = batch_inputs[idx]
                    label = batch_labels[idx]
                    expected_output = [0.0 for _ in range(NUM_OUTPUTS)]
                    expected_output[label] = 1.0
                    
                    # Forward pass
                    hidden_outputs, hidden_z, output_outputs, output_z = nn.forward(x)
                    
                    # Compute loss
                    loss = cross_entropy_loss(output_outputs, expected_output)
                    total_loss += loss
                    
                    # Backward pass
                    backward(nn, x, hidden_outputs, output_outputs, expected_output)
                    
            end_time = time.time()
            duration = end_time - start_time
            average_loss = total_loss / num_samples
            print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}, Time: {duration:.2f}s")
            loss_file.write(f"{epoch+1},{average_loss},{duration}\n")

# Testing Function
def test(nn, inputs, labels):
    num_samples = len(inputs)
    correct_predictions = 0
    
    for idx in range(num_samples):
        x = inputs[idx]
        label = labels[idx]
        
        hidden_outputs, hidden_z, output_outputs, output_z = nn.forward(x)
        predicted_label = output_outputs.index(max(output_outputs))
        
        if predicted_label == label:
            correct_predictions += 1
    
    accuracy = correct_predictions / num_samples * 100.0
    print(f"Test Accuracy: {accuracy:.2f}%")

# Read MNIST Images
def read_mnist_images(filename, num_images):
    images = []
    with open(filename, 'rb') as f:
        f.read(16)  # Skip the header
        for _ in range(num_images):
            image = []
            for _ in range(NUM_INPUTS):
                pixel = f.read(1)
                if not pixel:
                    break
                pixel = struct.unpack('>B', pixel)[0]
                image.append(pixel / 255.0)  # Normalize pixel values
            images.append(image)
    return images

# Read MNIST Labels
def read_mnist_labels(filename, num_labels):
    labels = []
    with open(filename, 'rb') as f:
        f.read(8)  # Skip the header
        for _ in range(num_labels):
            label = f.read(1)
            if not label:
                break
            label = struct.unpack('>B', label)[0]
            labels.append(label)
    return labels

# Main Function
def main():
    # Read training data
    print("Loading training data...")
    train_images = read_mnist_images('./data/train-images.idx3-ubyte', TRAIN_SAMPLES)
    train_labels = read_mnist_labels('./data/train-labels.idx1-ubyte', TRAIN_SAMPLES)
    
    # Read test data
    print("Loading test data...")
    test_images = read_mnist_images('./data/t10k-images.idx3-ubyte', TEST_SAMPLES)
    test_labels = read_mnist_labels('./data/t10k-labels.idx1-ubyte', TEST_SAMPLES)
    
    # Initialize neural network
    print("Initializing neural network...")
    nn = NeuralNetwork()
    
    # Train the neural network
    print("Training the neural network...")
    train(nn, train_images, train_labels)
    
    # Test the neural network
    print("Testing the neural network...")
    test(nn, test_images, test_labels)

if __name__ == "__main__":
    main()