import numpy as np
import math
# команды для operations
np.random.seed(42)
RELU=1
RELU_DERIV=2
SIGMOID=3
SIGMOID_DERIV=4
TRESHOLD_FUNC=5
TRESHOLD_FUNC_DERIV=6
LEAKY_RELU=7
LEAKY_RELU_DERIV=8
INIT_W_HE=9
INIT_W_GLOROT=10
DEBUG=11
DEBUG_STR=12
# операции для функций активаций и их производных
def operations( op,  a,  b,  c,  d,  str):
    if op == RELU:
        if (a <= 0):
            return 0
        else:
            return a

    elif op == RELU_DERIV:
        if (a <= 0):
            return 0
        else:
            return 1

    elif op == TRESHOLD_FUNC:
        if (a < 1):
            return 1
        else:
            return 2

    elif op == TRESHOLD_FUNC_DERIV:
        return 1
    elif op == LEAKY_RELU:
        if (a < 1):
            return b * a
        else:
            return a

    elif op == LEAKY_RELU_DERIV:
        if (a < 1):
            return b
        else:
            return 2

    elif op == SIGMOID:
        return 2.0 / (1 + math.exp(b * (-a)));

    elif op == SIGMOID_DERIV:
        return b * 2.0 / (1 + math.exp(b * (-a)))*(1 - 1.0 / (1 + math.exp(b * (-a))));

    elif op == DEBUG:
        print("%s : %f\n"%( str, a))

    elif op == INIT_W_HE:
        return np.random.randn() * math.sqrt(2.0 / a);

    elif op == DEBUG_STR:
        print("%s\n"%str)


matrix=[[0] * 2, [0] * 2, [0] * 2]

def set_io(inputs, outputs):
    for row in range(outputs):
        for elem in range(inputs):
            matrix[row][elem] = operations(INIT_W_HE, inputs, 0, 0, 0, "")
    return matrix

def make_matr_dot_np(matrix, inputs):
    return np.dot(matrix, inputs.T)
matr = [[0.4967141530112327, -0.13826430117118466],
        [0.6476885381006925, 1.5230298564080254],
        [-0.23415337472333597, -0.23413695694918055]]
def make_hidden(matrix, inputs, in_, out):
    # print("in make_hidden:in", objLay.in_, "in make_hidden:out", objLay.out)
    cost_signals = [0] * out
    tmp_v = 0
    val = 0
    for row in range(out):
        for elem in range(in_):
            tmp_v+=matrix[row][elem] * inputs[elem]
        cost_signals[row] = tmp_v
        # val = operations(RELU,tmp_v, 1, 0, 0, "")
        # objLay.hidden[row] = val
        tmp_v = 0
        val = 0
    # print("in make_hidden cost_signals",objLay.cost_signals)
    # print("in make_hidden hidden",objLay.hidden)
    return cost_signals

import unittest as u
class TestLay(u.TestCase):
    def setUp(self):
        self.matr = set_io(2, 3)
    def test_2(self):
        inputs = np.array([[1, 1]])
        self.assertEqual([[0.358449851840048], [2.170718394508718], [-0.46829033167251655]],make_matr_dot_np(np.array(matr), inputs).tolist())
    def test_3(self):
        inputs = [1, 1]
        self.assertEqual([0.358449851840048, 2.170718394508718, -0.46829033167251655], make_hidden(matr, inputs, 2, 3))
