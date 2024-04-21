from typing import Sequence, Callable, List, Tuple

import jax
import jax.numpy as jnp


def main(argv=None):
    """Back Propagation Application"""

    # Neural Network Architecture:
    hidden_layers = [32, 32, 32]
    input_size = 32
    output_size = 2

    # Build Neural Network:
    def initialize_network(
        hidden_layers: Sequence[int],
        input_size: int,
        output_size: int,
    ) -> Tuple[Callable, List[jnp.ndarray], List[jnp.ndarray]]:
        """
            Initializes Neural Network Parameters
            and Returns Forward Pass Function.

            Args:
                hidden_layers: Sequence of integers representing the size
                                of each hidden layer
                input_size: Integer representing the size of the input
                output_size: Integer representing the size of the output

            Output:
                forward_pass: Function that computes the forward pass
                                of the neural network given an input and
                                the weights and biases
                weights: List of randomized initial weights for each layer
                biases: List of randomized biases for each layer
        """

        # Initialize Weights and Biases:
        weights = []
        biases = []
        for i, hidden_layer in enumerate(hidden_layers):
            if i == 0:
                w = jax.random.normal(
                    jax.random.PRNGKey(0), (input_size, hidden_layer),
                )
                b = jax.random.normal(
                    jax.random.PRNGKey(0), (hidden_layer,),
                )
            elif i == len(hidden_layers) - 1:
                w = jax.random.normal(
                    jax.random.PRNGKey(0), (hidden_layer, output_size),
                )
                b = jax.random.normal(
                    jax.random.PRNGKey(0), (output_size,),
                )
            else:
                w = jax.random.normal(
                    jax.random.PRNGKey(0), (hidden_layer, hidden_layer),
                )
                b = jax.random.normal(
                    jax.random.PRNGKey(0), (hidden_layer,),
                )

            # Append Weights and Biases:
            weights.append(w)
            biases.append(b)

        # Neural Network Layer:
        def layer(x, w, b):
            return x @ w + b

        # Activation Function:
        def activation(x):
            return jax.nn.relu(x)

        # Forward Pass Function:
        def forward_pass(x, weights, biases):
            for i, (w, b) in enumerate(zip(weights, biases)):
                x = layer(x, w, b)
                if i != len(hidden_layers) - 1:
                    x = activation(x)
                else:
                    return x

        return forward_pass, weights, biases

    # Loss Function: Mean Squared Error
    def loss_fn(x, y, weights, biases):
        y_pred = forward_pass(x, weights, biases)
        return jnp.mean(jnp.square(y - y_pred))

    # Initialize Neural Network:
    forward_pass, weights, biases = initialize_network(
        hidden_layers, input_size, output_size,
    )

    # Learn trivial solution to random desired output:
    x = jax.random.normal(jax.random.PRNGKey(0), (input_size,))
    desired_output = jax.random.normal(jax.random.PRNGKey(0), (output_size,))

    # Compute gradient of the loss function relative to the Weights and Biases:
    grad_fn = jax.value_and_grad(loss_fn, (2, 3))

    # Learning Parameters:
    learning_rate = 1e-5
    learning_iterations = 100

    # Training Loop:
    for i in range(learning_iterations):
        # Compute Gradient:
        loss, (weight_gradients, bias_gradients) = grad_fn(
            x, desired_output, weights, biases,
        )

        # Gradient Descent Algorithm:
        gradient_descent = lambda x, dx: x - learning_rate * dx

        # Update Weights and Biases:
        # Iterates over all values and gradients and
        # applies them to the gradient descent function
        weights = jax.tree_map(gradient_descent, weights, weight_gradients)
        biases = jax.tree_map(gradient_descent, biases, bias_gradients)

        print(f'Iteration: {i} \t Loss: {loss}')

    # Verify that the Neural Network has learned the trivial solution:
    y_pred = forward_pass(x, weights, biases)
    print(
        f'Trained Model Verification \n'
        f'Predicted Output: {y_pred} \n'
        f'Desired Output: {desired_output} \n'
        f'Mean Squared Error: {jnp.mean(jnp.square(y_pred - desired_output))}'
    )


if __name__ == '__main__':
    main()
