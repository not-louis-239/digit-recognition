# Digit Recogniser

Project to simulate the evolution of a digit recogniser AI.

## Details

The AI is trained to recognise digits from 0-9 from a 28x28 image.


The AI is trained using a simple feedforward neural network with two hidden layers.
The weights and biases start randomly generated, then "evolve" over time.

This is done by:
* evaluate how "bad" each model is based on training data
* select the best models and discard the rest
* mutate the best models (change weights and biases randomly) to create new ones
* repeat for a given number of generations

Why did I decide to make this? Because I was bored lol.

## Installation

To run, simply run the `main.py` file. It will generate a model for you.
Then have fun making your own AI models.

## Licence

Licensed under the Apache Licence v2.0. See LICENSE for details.
