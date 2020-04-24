from NN_params import NnParams   # импортруем параметры сети
from serial_deserial_func import deserializ
from nn_constants import bc_bufLen, RELU, LEAKY_RELU, SIGMOID, TAN
import unittest as u
#----------------------Основные параметры сети----------------------------------
# создать параметры сети
def create_nn_params():
    return NnParams()
class TestLay(u.TestCase):
    def setUp(self) -> None:
        self.nn_params = create_nn_params()
        self.nn_params.with_bias = False
        self.nn_params.with_adap_lr = True
        self.nn_params.lr = 0.01
        self.nn_params.act_fu = RELU
    def test_7(self):
        nn_map = (2, 3, 1)
        X = [[1, 1], [1, 0], [0, 1], [0, 0]]
        Y_or = [[1], [1], [1], [0]]
        Y_and = [[1], [0], [0], [0]]
        b_c = [0] * bc_bufLen  # буффер для сериализации матричных элементов и входов
        # X = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0],
        #      [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1],
        #      [1, 1, 1, 0], [1, 1, 1, 1]]
        # Y = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0],
        #      [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1],
        #      [1, 1, 1, 0], [1, 1, 1, 1]]

        b_c = [0] * bc_bufLen  # буффер для сериализации матричных элементов и входов
        initiate_layers(self.nn_params, nn_map, len(nn_map))
        learn(b_c, self.nn_params, 7, X, Y_and, 100)
        compil_serializ(self.nn_params, b_c, self.nn_params.list_, len(nn_map) - 1, "weight2" )
        print("in test_7 after learn matr")
        for i in self.nn_params.list_:
            print(i.matrix)

    def test_8(self):
        nn_params1=create_nn_params()
        deserializ(nn_params1, nn_params1.list_, "weight2")
        print("in test 8",nn_params1.with_bias)
        for i in nn_params1.list_:
            print(i.matrix)
        print(answer_nn_direct(nn_params1, [1, 1], 1))
        # print(answer_nn_direct(nn_params1, [0, 1], 1))
        print("*ON CONTRARY*")
        answer_nn_direct_on_contrary(nn_params1, [0], 1)
if __name__ == '__main__':
    t = TestLay()
    u.main()
