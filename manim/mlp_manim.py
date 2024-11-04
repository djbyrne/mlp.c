from manim import *

class MLPVisualization(Scene):
    def construct(self):
        # Parameters for the neural network
        input_neurons = 3
        hidden_neurons = 4
        output_neurons = 2

        # Colors for the neurons
        neuron_color = BLUE

        # Spacing of the neurons
        input_layer_position = LEFT * 4
        hidden_layer_position = ORIGIN
        output_layer_position = RIGHT * 4

        # Creating the input layer
        input_layer = VGroup(*[Dot(radius=0.2, color=neuron_color) for _ in range(input_neurons)]).arrange(DOWN, buff=1)
        input_layer.shift(input_layer_position)

        # Creating the hidden layer
        hidden_layer = VGroup(*[Dot(radius=0.2, color=neuron_color) for _ in range(hidden_neurons)]).arrange(DOWN, buff=1)
        hidden_layer.shift(hidden_layer_position)

        # Creating the output layer
        output_layer = VGroup(*[Dot(radius=0.2, color=neuron_color) for _ in range(output_neurons)]).arrange(DOWN, buff=1)
        output_layer.shift(output_layer_position)

        # Adding layers to the scene
        self.play(Create(input_layer), Create(hidden_layer), Create(output_layer))
        self.wait(1)

        # Connecting the input layer to the hidden layer
        input_to_hidden_connections = VGroup()
        for input_neuron in input_layer:
            for hidden_neuron in hidden_layer:
                connection = Line(input_neuron.get_center(), hidden_neuron.get_center(), color=WHITE)
                input_to_hidden_connections.add(connection)
        
        # Connecting the hidden layer to the output layer
        hidden_to_output_connections = VGroup()
        for hidden_neuron in hidden_layer:
            for output_neuron in output_layer:
                connection = Line(hidden_neuron.get_center(), output_neuron.get_center(), color=WHITE)
                hidden_to_output_connections.add(connection)

        # Animating the connections
        self.play(Create(input_to_hidden_connections), Create(hidden_to_output_connections))
        self.wait(1)

        # Simulating a forward pass: Highlight neurons and connections
        self.simulate_forward_pass(input_layer, hidden_layer, output_layer, input_to_hidden_connections, hidden_to_output_connections)

    def simulate_forward_pass(self, input_layer, hidden_layer, output_layer, input_to_hidden_connections, hidden_to_output_connections):
        # Highlight the input neurons
        self.play(*[neuron.animate.set_fill(YELLOW) for neuron in input_layer])
        self.wait(0.5)

        # Highlight the connections from input to hidden
        self.play(*[connection.animate.set_color(YELLOW) for connection in input_to_hidden_connections])
        self.wait(0.5)

        # Highlight the hidden neurons
        self.play(*[neuron.animate.set_fill(YELLOW) for neuron in hidden_layer])
        self.wait(0.5)

        # Highlight the connections from hidden to output
        self.play(*[connection.animate.set_color(YELLOW) for connection in hidden_to_output_connections])
        self.wait(0.5)

        # Highlight the output neurons
        self.play(*[neuron.animate.set_fill(YELLOW) for neuron in output_layer])
        self.wait(1)