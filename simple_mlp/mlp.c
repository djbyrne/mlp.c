#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_INPUTS 2       // Number of input neurons
#define NUM_HIDDEN 4       // Number of hidden neurons
#define NUM_OUTPUTS 1      // Number of output neurons
#define NUM_SAMPLES 4      // Number of training samples
#define LEARNING_RATE 0.01 // Learning rate
#define EPOCHS 1000000     // Number of training epochs

// Sigmoid activation function and its derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}


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
void backward(NeuralNetwork *nn, double inputs[], double hidden_outputs[], double output_outputs[],
                     double expected_outputs[], double delta_hidden[], double delta_output[]);

void update_weights_biases(LinearLayer *layer, double inputs[], double deltas[]);
void train(NeuralNetwork *nn, double inputs[][NUM_INPUTS], double expected_outputs[][NUM_OUTPUTS]);
void test(NeuralNetwork *nn, double inputs[][NUM_INPUTS], double expected_outputs[][NUM_OUTPUTS]);

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
    // Initialize layers
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
        // Calculate the weighted sum of inputs
        double activation = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++) {
            activation += inputs[j] * layer->weights[j][i];
        }
        // Apply the activation function
        outputs[i] = sigmoid(activation);

    }
}

void backward(NeuralNetwork *nn, double inputs[], double hidden_outputs[], double output_outputs[],
                     double errors[], double *delta_hidden, double *delta_output) {
    // Output layer delta
    for (int i = 0; i < nn->output_layer.output_size; i++) {
        delta_output[i] = errors[i] * sigmoid_derivative(output_outputs[i]);
    }

    // Hidden layer delta
    for (int i = 0; i < nn->hidden_layer.output_size; i++) {
        double error = 0.0;
        for (int j = 0; j < nn->output_layer.output_size; j++) {
            error += delta_output[j] * nn->output_layer.weights[i][j];
        }
        delta_hidden[i] = error * sigmoid_derivative(hidden_outputs[i]);
    }
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
    // Allocate memory for deltas
    double *delta_hidden = (double *)malloc(NUM_HIDDEN * sizeof(double));
    double *delta_output = (double *)malloc(NUM_OUTPUTS * sizeof(double));
    double errors[NUM_OUTPUTS];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0.0;
        double total_errors = 0.0;
        for (int sample = 0; sample < NUM_SAMPLES; sample++) {
            double hidden_outputs[NUM_HIDDEN];
            double output_outputs[NUM_OUTPUTS];

            // Forward propagation
            forward_propagation(&nn->hidden_layer, inputs[sample], hidden_outputs);
            forward_propagation(&nn->output_layer, hidden_outputs, output_outputs);

            // Loss function: Mean Squared Error
            for (int i = 0; i < NUM_OUTPUTS; i++) {
                errors[i] = expected_outputs[sample][i] - output_outputs[i];
                total_errors += errors[i] * errors[i];
            }

            // Backpropagation (compute deltas)
            backward(nn, inputs[sample], hidden_outputs, output_outputs,
                            errors, delta_hidden, delta_output);

            // Update weights and biases
            update_weights_biases(&nn->output_layer, hidden_outputs, delta_output);
            update_weights_biases(&nn->hidden_layer, inputs[sample], delta_hidden);
        }

        // Optional: Print error every 1000 epochs
        loss = total_errors / NUM_SAMPLES;
        if ((epoch + 1) % 1000 == 0) {
            printf("Epoch %d, Error: %f\n", epoch + 1, loss);
        }
    }

    // Free allocated memory for deltas
    free(delta_hidden);
    free(delta_output);
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
    // seed the random number generator
    srand(42);

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