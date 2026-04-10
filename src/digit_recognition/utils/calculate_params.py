# Calculate the number of parameters in a neural network of a given shape

from typing import Sequence

def calculate_num_params(network_shape: Sequence[int]) -> int:
    total_params = 0

    for i in range(len(network_shape) - 1):
        n_init = network_shape[i]
        n_next = network_shape[i+1]

        num_weights = n_init * n_next
        num_biases = n_next

        total_params += num_weights + num_biases

    return total_params

candidate_shapes = [
    [784, 16, 16, 10],
    [784, 32, 32, 10],
    [784, 32, 32, 32, 10],
    [784, 100, 100, 10]  # this might be a nice one!
]

def _test():
    for shape in candidate_shapes:
        print(f"Number of parameters for shape {shape}: {calculate_num_params(shape):,}")

if __name__ == "__main__":
    _test()
