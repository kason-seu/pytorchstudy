from __future__ import print_function
from utils import relu
import utils
import numpy as np

def test_relu():
    X = np.random.random((3, 2)) - 0.5
    print(X)
    b = relu(X)
    print(b)

    s = utils.sigmod(X)
    print(s)



if __name__ == "__main__":
    test_relu()