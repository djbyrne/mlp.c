# from manim import *

# class MLPDiagramAnimation(Scene):
#     def construct(self):
#         # Step 1: Display the Input Matrix
#         input_matrix = MathTable(
#             [[2], [1], [3]],
#             include_outer_lines=True
#         ).scale(0.5).shift(LEFT * 5 + UP * -0.5)

#         input_label = Text("Input Layer").scale(0.5).next_to(input_matrix, UP)
        
#         # Step 2: Display Hidden Layer 1 Weights (W1) and Biases (b1)
#         w1_matrix = MathTable(
#             [[1, -1, 1 ], [1, 1, 0]],
#             include_outer_lines=True
#         ).scale(0.5).next_to(input_matrix, RIGHT, buff=1)
        
#         b1_matrix = MathTable(
#             [[-5], [0], [1]],
#             include_outer_lines=True
#         ).scale(0.5).next_to(w1_matrix, RIGHT, buff=1)

#         w1_label = Text("W1").scale(0.5).next_to(w1_matrix, UP)
#         b1_label = Text("b1").scale(0.5).next_to(b1_matrix, UP)
        
#         # # Step 3: Result of W1 * Input + b1
#         result_matrix_1 = MathTable(
#             [[-1], [3]],
#             include_outer_lines=True
#         ).scale(0.5).next_to(b1_matrix, RIGHT, buff=1)

#         relu_result_1 = MathTable(
#             [[0], [3]],
#             include_outer_lines=True
#         ).scale(0.5).next_to(result_matrix_1, RIGHT, buff=1)
        
#         result_label_1 = Text("Output 1").scale(0.5).next_to(result_matrix_1, UP)
#         relu_label_1 = Text("ReLU").scale(0.5).next_to(relu_result_1, UP)

#         # # Step 4: Hidden Layer 2 weights and biases
#         # w2_matrix = MathTable(
#         #     [[0, 0, 1, 1], [-1, 1, 0, 2], [1, 0, 1, 1], [-1, 0, 1, 1]],
#         #     include_outer_lines=True
#         # ).scale(0.7).shift(DOWN * 2 + LEFT * 5)

#         # b2_matrix = MathTable(
#         #     [[1], [2], [1], [1]],
#         #     include_outer_lines=True
#         # ).scale(0.7).next_to(w2_matrix, RIGHT, buff=1)

#         # w2_label = Text("W2").next_to(w2_matrix, UP)
#         # b2_label = Text("b2").next_to(b2_matrix, UP)
        
#         # # Step 5: Result of W2 * Hidden Layer 1 Output + b2
#         # result_matrix_2 = MathTable(
#         #     [[0], [3], [4], [0]],
#         #     include_outer_lines=True
#         # ).scale(0.7).next_to(b2_matrix, RIGHT, buff=1)

#         # relu_result_2 = MathTable(
#         #     [[0], [3], [4], [0]],
#         #     include_outer_lines=True
#         # ).scale(0.7).next_to(result_matrix_2, RIGHT, buff=1)
        
#         # result_label_2 = Text("Hidden Layer 2").next_to(result_matrix_2, UP)
#         # relu_label_2 = Text("ReLU").next_to(relu_result_2, UP)

#         # # Step 6: Hidden Layer 3 weights and biases
#         # w3_matrix = MathTable(
#         #     [[0, 1, -1], [1, 0, 0], [1, 1, 0], [-1, 0, 1]],
#         #     include_outer_lines=True
#         # ).scale(0.7).shift(DOWN * 2 + LEFT * 5)

#         # b3_matrix = MathTable(
#         #     [[1], [2], [1], [1]],
#         #     include_outer_lines=True
#         # ).scale(0.7).next_to(w3_matrix, RIGHT, buff=1)

#         # # Step 7: Repeat for Hidden Layer 3, Output Layer and Sigmoid Activation
#         # # (similar to above steps for w3, b3, w4, b4, and final output).

#         # Step 8: Animations
#         self.play(Create(input_matrix), Write(input_label))
#         self.wait(1)

#         # Animate the multiplication of W1 * input + b1
#         self.play(Create(w1_matrix), Write(w1_label), Create(b1_matrix), Write(b1_label))
#         self.wait(1)

#         self.play(Create(result_matrix_1), Write(result_label_1))
#         self.wait(1)

#         self.play(Create(relu_result_1), Write(relu_label_1))
#         self.wait(1)

#         # # Continue with Hidden Layer 2, 3, and Output Layer animations...


from manim import *

class MLPAnimation(Scene):
    def construct(self):
        # Define the input vector
        input_vector = Matrix([[2], [1], [3]], left_bracket="[", right_bracket="]")
        input_label = Text("Input", font_size=24).next_to(input_vector, UP)

        # Define the weights matrix
        weights_matrix = Matrix([[1, -1, 1], [1, 1, 0]])
        weights_label = Text("Weights", font_size=24).next_to(weights_matrix, UP)

        # Define the bias vector
        bias_vector = Matrix([[-5], [0]], left_bracket="[", right_bracket="]")
        bias_label = Text("Bias", font_size=24).next_to(bias_vector, UP)

        # Position the matrices on the screen
        input_group = VGroup(input_label, input_vector).to_edge(LEFT, buff=1)
        weights_group = VGroup(weights_label, weights_matrix).next_to(input_group, RIGHT, buff=2)
        bias_group = VGroup(bias_label, bias_vector).next_to(weights_group, RIGHT, buff=2)

        # Display the input vector, weights matrix, and bias vector
        self.play(Write(input_group))
        self.play(Write(weights_group))
        self.play(Write(bias_group))

        # Compute the dot product of the input and weights
        # The result is a 2x1 vector
        # Calculation:
        # [ [1, -1, 1],     [2]     [ (1*2) + (-1*1) + (1*3) ]     [ 4 ]
        #   [1,  1, 0]  x   [1]  =   [ (1*2) +  (1*1) + (0*3) ]  = [ 3 ]
        #                       [3]

        # Create the product vector
        product_vector = Matrix([[4], [3]], left_bracket="[", right_bracket="]")
        product_label = Text("Product", font_size=24).next_to(product_vector, UP)
        product_group = VGroup(product_label, product_vector)

        # Position the product vector below the weights matrix
        product_group.next_to(weights_group, DOWN, buff=1)

        # Animate the multiplication
        self.play(Write(product_label))
        self.play(ReplacementTransform(VGroup(input_vector.copy(), weights_matrix.copy()), product_vector))

          # Make the previous matrices disappear
        self.play(
            FadeOut(input_group),
            FadeOut(weights_group),
        )

        # Show the addition of the bias
        result_vector = Matrix([[-1], [3]], left_bracket="[", right_bracket="]")
        result_label = Text("Result", font_size=24).next_to(result_vector, UP)
        result_group = VGroup(result_label, result_vector)

        # Position the result vector below the product vector
        result_group.next_to(product_group, RIGHT, buff=1)

        # Animate the addition
        self.play(Write(result_label))
        self.play(ReplacementTransform(VGroup(product_vector.copy(), bias_vector.copy()), result_vector))

        # Make the previous matrices disappear
        self.play(
            FadeOut(bias_group),
            FadeOut(product_group),
        )

        # Move the result to the center of the screen
        self.play(result_group.animate.move_to(ORIGIN))

        # Wait for a moment before ending the scene
        self.wait(2)