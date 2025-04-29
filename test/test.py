import numpy as np
from matplotlib import pyplot as plt


def test_sigmoid():
    x = np.linspace(-5, 5, 1000)
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_derivative = sigmoid * (1 - sigmoid)

    plt.plot(x, sigmoid_derivative)
    plt.xlabel('x')
    plt.ylabel('Sigmoid Derivative')
    plt.title('Sigmoid Derivative Function')
    plt.show()

if __name__ == '__main__':
    test_sigmoid()