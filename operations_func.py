from nn_constants import RELU_DERIV, RELU, TRESHOLD_FUNC, TRESHOLD_FUNC_DERIV, LEAKY_RELU, LEAKY_RELU_DERIV,\
SIGMOID, SIGMOID_DERIV, DEBUG, DEBUG_STR, INIT_W_HE, INIT_W_GLOROT
import numpy as np
import math
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
        if (a <= 0):
            return 1
        else:
            return 2

    elif op == TRESHOLD_FUNC_DERIV:
        return 1

    elif op == LEAKY_RELU:
        if (a <= 0):
            return b * a
        else:
            return a

    elif op == LEAKY_RELU_DERIV:
        if (a <= 0):
            return b
        else:
            return 2
    elif op == SIGMOID:
        return 2.0 / (1 + math.exp(b * (-a)))

    elif op == SIGMOID_DERIV:
        return b * 2.0 / (1 + math.exp(b * (-a)))*(1 - 1.0 / (1 + math.exp(b * (-a))))

    elif op == DEBUG:
        print("%s : %f\n"%( str, a))
    elif op == INIT_W_HE:
        return np.random.randn() * math.sqrt(2 / a)

    elif op == DEBUG_STR:
        print("%s\n"%str)
