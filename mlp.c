#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_INPUTS 2      // Number of input neurons
#define NUM_HIDDEN 4      // Number of hidden neurons
#define NUM_OUTPUTS 1     // Number of output neurons
#define NUM_SAMPLES 4     // Number of training samples
#define LEARNING_RATE 0.01 // Learning rate
#define EPOCHS 1000000    // Number of training epochs

// Activation function and its derivative
double sigmoid(double x);
double sigmoid_derivative(double x);

// Data structure for a layer in the neural network
typedef struct {
    int input_size;
    int output_size;
    double **weights; // [input_size][output_size]
    double *biases;   // [output_size]
} LinearLayer;

// Data structure for the neural network
typedef struct {
    LinearLayer hidden_layer;
    LinearLayer output_layer;
} NeuralNetwork;

// Function prototypes
void initialize_layer(LinearLayer *layer, int input_size, int output_size);
void free_layer(LinearLayer *layer);

void initialize_network(NeuralNetwork *nn);
void free_network(NeuralNetwork *nn);

void forward_propagation(LinearLayer *layer, double inputs[], double outputs[]);
void backpropagation(NeuralNetwork *nn, double inputs[], double hidden_outputs[], double output_outputs[], double expected_outputs[], double delta_hidden[], double delta_output[]);

void update_weights_biases(LinearLayer *layer, double inputs[], double deltas[]);
void train(NeuralNetwork *nn, double inputs[][NUM_INPUTS], double expected_outputs[][NUM_OUTPUTS]);
void test(NeuralNetwork *nn, double inputs[][NUM_INPUTS], double expected_outputs[][NUM_OUTPUTS]);


// Activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of activation function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Initialize a layer
void initialize_layer(LinearLayer *layer, int input_size, int output_size) {
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Allocate memory for weights
    layer->weights = (double **)malloc(input_size * sizeof(double *));
    for (int i = 0; i < input_size; i++) {
        layer->weights[i] = (double *)malloc(output_size * sizeof(double));
    }

    // Allocate memory for biases
    layer->biases = (double *)malloc(output_size * sizeof(double));

    // Initialize weights and biases with random values
    for (int i = 0; i < input_size; i++)
        for (int j = 0; j < output_size; j++)
            layer->weights[i][j] = ((double)rand() / RAND_MAX) - 0.5;

    for (int i = 0; i < output_size; i++)
        layer->biases[i] = ((double)rand() / RAND_MAX) - 0.5;
}

// Free memory allocated for a layer
void free_layer(LinearLayer *layer) {
    for (int i = 0; i < layer->input_size; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
}

// Initialize the neural network
void initialize_network(NeuralNetwork *nn) {
    srand(time(NULL));
    initialize_layer(&nn->hidden_layer, NUM_INPUTS, NUM_HIDDEN);
    initialize_layer(&nn->output_layer, NUM_HIDDEN, NUM_OUTPUTS);
}

// Free memory allocated for the neural network
void free_network(NeuralNetwork *nn) {
    free_layer(&nn->hidden_layer);
    free_layer(&nn->output_layer);
}

// Forward propagation for a single layer
void forward_propagation(LinearLayer *layer, double inputs[], double outputs[]) {
    for (int i = 0; i < layer->output_size; i++) {
        double activation = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++) {
            activation += inputs[j] * layer->weights[j][i];
        }
        outputs[i] = sigmoid(activation);
    }
}

// Backpropagation
void backpropagation(NeuralNetwork *nn, double inputs[], double hidden_outputs[], double output_outputs[], double expected_outputs[], double delta_hidden[], double delta_output[]) {
    // Calculate output layer error and delta
    for (int i = 0; i < nn->output_layer.output_size; i++) {
        double error = expected_outputs[i] - output_outputs[i];
        delta_output[i] = error * sigmoid_derivative(output_outputs[i]);
    }

    // Calculate hidden layer error and delta
    for (int i = 0; i < nn->hidden_layer.output_size; i++) {
        double error = 0.0;
        for (int j = 0; j < nn->output_layer.output_size; j++) {
            error += delta_output[j] * nn->output_layer.weights[i][j];
        }
        delta_hidden[i] = error * sigmoid_derivative(hidden_outputs[i]);
    }

    // Update weights and biases
    update_weights_biases(&nn->output_layer, hidden_outputs, delta_output);
    update_weights_biases(&nn->hidden_layer, inputs, delta_hidden);
}

// Update weights and biases for a layer
void update_weights_biases(LinearLayer *layer, double inputs[], double deltas[]) {
    // Update weights
    for (int i = 0; i < layer->input_size; i++) {
        for (int j = 0; j < layer->output_size; j++) {
            layer->weights[i][j] += LEARNING_RATE * deltas[j] * inputs[i];
        }
    }

    // Update biases
    for (int i = 0; i < layer->output_size; i++) {
        layer->biases[i] += LEARNING_RATE * deltas[i];
    }
}

// Training function
void train(NeuralNetwork *nn, double inputs[][NUM_INPUTS], double expected_outputs[][NUM_OUTPUTS]) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_error = 0.0;
        for (int sample = 0; sample < NUM_SAMPLES; sample++) {
            double hidden_outputs[NUM_HIDDEN];
            double output_outputs[NUM_OUTPUTS];

            // Forward propagation
            forward_propagation(&nn->hidden_layer, inputs[sample], hidden_outputs);
            forward_propagation(&nn->output_layer, hidden_outputs, output_outputs);

            // Calculate error
            for (int i = 0; i < NUM_OUTPUTS; i++) {
                double error = expected_outputs[sample][i] - output_outputs[i];
                total_error += error * error;
            }

            // Backpropagation
            double delta_hidden[NUM_HIDDEN];
            double delta_output[NUM_OUTPUTS];
            backpropagation(nn, inputs[sample], hidden_outputs, output_outputs, expected_outputs[sample], delta_hidden, delta_output);
        }

        // Optional: Print error every 1000 epochs
        if ((epoch + 1) % 1000 == 0) {
            printf("Epoch %d, Error: %f\n", epoch + 1, total_error);
        }
    }
}

// Testing function
void test(NeuralNetwork *nn, double inputs[][NUM_INPUTS], double expected_outputs[][NUM_OUTPUTS]) {
    printf("\nTesting the trained network:\n");
    for (int sample = 0; sample < NUM_SAMPLES; sample++) {
        double hidden_outputs[NUM_HIDDEN];
        double output_outputs[NUM_OUTPUTS];

        // Forward propagation
        forward_propagation(&nn->hidden_layer, inputs[sample], hidden_outputs);
        forward_propagation(&nn->output_layer, hidden_outputs, output_outputs);

        // Print the result
        printf("Input: %.1f, %.1f, Expected Output: %.1f, Predicted Output: %.3f\n",
               inputs[sample][0], inputs[sample][1],
               expected_outputs[sample][0], output_outputs[0]);
    }
}

// Main function
int main() {
    // Training dataset for XOR problem
    double inputs[NUM_SAMPLES][NUM_INPUTS] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    double expected_outputs[NUM_SAMPLES][NUM_OUTPUTS] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Initialize neural network
    NeuralNetwork nn;
    initialize_network(&nn);

    // Train the neural network
    train(&nn, inputs, expected_outputs);

    // Test the trained network
    test(&nn, inputs, expected_outputs);

    // Free allocated memory
    free_network(&nn);

    return 0;
}