import numpy as np
import random
import linalg


def sigmoid(z):
    """The sigmoid function.
    Сигмоида
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function.
    Производная сигмоиды по e (шутка). По x
    """
    return sigmoid(z) * (1 - sigmoid(z))


def cost_function(network, test_data, onehot=True):
    c = 0
    for example, y in test_data:
        if not onehot:
            y = np.eye(3, 1, k=-int(y))
        yhat = network.feedforward(example)
        c += np.sum((y - yhat) ** 2)
    return c / len(test_data)


arr = [2, 3, 2, 1]
num_layers = len(arr)
bias = [np.random.randn(y, 1) for y in arr[1:]]

print(*bias)

weights = [np.random.randn(y, x) for y, x in zip(arr[1:], arr[:-1])]
print(f'Weights')
print(*weights)


def forward_pass(input):
    for b, w in zip(bias, weights):
        input = sigmoid(w.dot(input) + b)
    return input


def cost_derivative(outputactivations, y):
    return (outputactivations - y)


input = np.array([1, 2]).reshape(2, 1)

print(input)
print(forward_pass(input))
print('@@@@@@@@@@@@@@@')


def backprop(input, y):
    nabla_b = [np.zeros(b.shape) for b in bias]
    nabla_w = [np.zeros(w.shape) for w in weights]
    activation = input
    activations = [input]
    zs = []

    for b, w in zip(bias, weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

    delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-1 - 1].transpose())

    for l in range(2, num_layers):
        delta = np.dot(weights[-l + 1].transpose(), delta) * sigmoid_prime(zs[-l])  # ошибка на слое L-l
        nabla_b[-l] = delta  # производная J по смещениям L-l-го слоя
        nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())  # производная J по весам L-l-го слоя
    return nabla_b, nabla_w


# print(f"Значения выходного слоя ")
# print(*zs)
# print(f"Значения функции активации ")
# print(*activations)

y = 0
print('nabla_b')
print(backprop(input, y)[0])
print('nabla_w')
print(backprop(input, y)[1])

