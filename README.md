# Python-Neural-Network

## About

A Neural Network written in Python.  
No dependencies -- pure python.   
## Quick Start

First clone this repository or download the `NeuralNetwork.py` file and put it alongside your code.  
Your project must look like this :

```

My-Project
├── NeuralNetwork.py
├── Your_Code.py
└── ---Your-Other-Files---

```

Write this code block into your code :

```python

from NeuralNetwork import Neural_Network

if __name__ == "__main__":
	# XOR truth table
	TABLE = [
	([0, 0], [0]),
	([0, 1], [1]),
	([1, 0], [1]),
	([1, 1], [0])
	]

	nn = Neural_Network(2, [2], 1, activation="leaky-relu")
	nn.Train(TABLE, epoches=100000, lr=0.05)

	print("--- predictions ---")
	for inp, target in TABLE:
		out = nn.Forward(inp)[0]
		print(f"input {inp}  target {target[0]}  output {out:.4f}")

```

## Data format

`TABLE` is a list of `(inputs, targets)` tuples. Each tuple is one case.
+ `inputs` is a list of numbers — one per network input
+ `targets` is a list of numbers — one per network output

```python
TABLE = [
	([0, 0], [0]),   # input [0, 0]  ->  target [0]
	([0, 1], [1]),   # input [0, 1]  ->  target [1]
	([1, 0], [1]),   # input [1, 0]  ->  target [1]
	([1, 1], [0])    # input [1, 1]  ->  target [0]
]
```

## Activation

You have 7 activation methods to choose from. Pass one to `activation` parameter of `Neural_Network`.

| Name       | Formula                         | Range        |
| ---------- | ------------------------------- | ------------ |
| Sigmoid    | $1/(1+e^{-x})$                  | $(0, 1)$     |
| Tanh       | $(e^x - e^{-x})/(e^x + e^{-x})$ | $(-1, 1)$    |
| ReLU       | $max(0, x)$                     | $[0, ∞)$     |
| Leaky-ReLU | $\alpha x+(1-\alpha)max(x,0)$   | $(-∞, ∞)$    |
| Soft-Plus  | $ln(1+e^x)$                     | $(0, ∞)$     |
| Linear     | $x$                             | $(-∞, ∞)$    |
| Silu       | $x\cdot \sigma(x)$              | $(-0.28, ∞)$ |

>[!Note]
>The `activation` parameter of `Neural_Network` is Case-Insensitive.  
> `(` or `)` — the value is not included  
> `[` or `]` — the value is included  
> `∞` is always open on its side.  


## API

### Constructor

`Neural_Network(...)` — Creates a neural network.

| Parameter      | Type      | Default   | Description                        |
| -------------- | --------- | --------- | ---------------------------------- |
| `input_size`   | int       | —         | Number of input neurons            |
| `hiddens_size` | list[int] | —         | Neurons per hidden layer           |
| `output_size`  | int       | —         | Number of output neurons           |
| `activation`   | str       | "Sigmoid" | Type of activation used in network |
| `alpha`        | float     | 0.01      | Negative slope for `Leaky-ReLU`    |

`hiddens_size`:
- Each item indicates number of neurons in the corresponding layer.
- The length of this list shows the number of hidden layers.
```python
# Example: 2 inputs -> hidden layers of 5, 4, 3 -> 1 output
nn = Neural_Network(2, [5, 4, 3], 1)
```

`alpha`:
- Only applies if `activation="Leaky-ReLU"`.

### .Train(...)

Trains the network on the given data.

| Parameter       | Type                    | Description                         |
| --------------- | ----------------------- | ----------------------------------- |
| `data`          | list[(inputs, targets)] | Table to train on                   |
| `epoches`       | int                     | Number of full passes over the data |
| `learning_rate` | float                   | Learning rate                       |

>[!NOTE]
>`data`: see [Data Format](#data-format) section.[^1]

### .Forward(...)

Moves inputs accross network and returns the results.

| Parameter | Type        | Description                        |
| --------- | ----------- | ---------------------------------- |
| `inputs`  | list[float] | Input values, one per input neuron |
`inputs`:  length must match `input_size`

Returns a list[float] with the length equal to `output_size`

>[!NOTE]
>`input_size`, `output_size`: see [Constructor](#constructor) section[^2].

### .Backward(...)

Backpropagates the error from the last forward pass and updates weights and biases in place.

| Parameter       | Type        | Description                             |
| --------------- | ----------- | --------------------------------------- |
| `answers`       | list[float] | Expected answers, one per output neuron |
| `learning_rate` | float       | Learning rate                           |
`answers`: length must match `output_size`
returns `None`,  updates weights and biases in place.

>[!NOTE]
>`output_size`: look at [Constructor](#constructor)[^2].

>[!WARNING]
>`Forward` must be called before `Backward`. It uses the cached values from the last forward pass.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

[^1]: If not working use this link [[#Data format]]
[^2]: If not working use this link [[#Constructor]]
