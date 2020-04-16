from nn_constants import RELU_DERIV, RELU, TRESHOLD_FUNC, TRESHOLD_FUNC_DERIV, LEAKY_RELU, LEAKY_RELU_DERIV,\
SIGMOID, SIGMOID_DERIV, DEBUG, DEBUG_STR, INIT_W_HE, INIT_W_GLOROT_V1, INIT_W_HABR, INIT_W_MY, INIT_W_UNIFORM,\
    TAN, TAN_DERIV
import numpy as np
import math
np.random.seed(42)
ready = False
f=0
# операции для функций активаций и их производных
def operations( op,  a,  b,  c,  d,  str):
    global ready, f
    alpha = 1.7159
    beta = 2 / 3

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
        return b * 2.0 / (1 + math.exp(b * (-a)))*(1 - 2.0 / (1 + math.exp(b * (-a))))
    elif op == DEBUG:
        print("%s : %f\n"%( str, a))
    elif op == INIT_W_HABR:
        return 2 * np.random.random() - 1
    elif op == INIT_W_HE:
        return np.random.randn() * math.sqrt(2 / a)
    elif op == INIT_W_MY:
        if ready:
            ready = False
            return -0.01
        ready = True
        return 0.01
    elif op ==INIT_W_GLOROT_V1:
        return 2 / (a + b)
    elif op == INIT_W_UNIFORM:
        print("in op  INIT_W_UNIFORM a=",a,"b=",b)
        return a + np.random.random() * (b - a)
    elif op == TAN:
        f = alpha * math.tanh(beta * a)
        return  f
    elif op == TAN_DERIV:
        return beta / alpha * (alpha * alpha - f * f)
    elif op == DEBUG_STR:
        print("%s\n"%str)
