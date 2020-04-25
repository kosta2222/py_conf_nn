from NN_params import NnParams   # импортруем параметры сети
from serial_deserial_func import deserializ
from nn_constants import bc_bufLen, RELU, LEAKY_RELU, SIGMOID, TAN
from lear_func import initiate_layers, answer_nn_direct, answer_nn_direct_on_contrary
from serial_deserial_func import compil_serializ
from fit import fit
import unittest as u
#----------------------Основные параметры сети----------------------------------
# создать параметры сети
def create_nn_params():
    return NnParams()
class TestLay(u.TestCase):
    def setUp(self) -> None:
      pass

    def test_1(self):
        f = open("test_out.txt", "w")
        nn_params = create_nn_params()
        nn_params.with_bias = False
        nn_params.with_adap_lr = True
        nn_params.lr = 0.01
        nn_params.act_fu = RELU
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
        initiate_layers(nn_params, nn_map, len(nn_map))
        fit(b_c, nn_params, 7, X, Y_and, 100)
        compil_serializ(nn_params, b_c, nn_params.list_, len(nn_map) - 1, "weight2" )
        print("in test_7 after learn matr", file=f)
        for i in nn_params.list_:
            print(i.matrix,file=f)
        nn_params1=create_nn_params()
        deserializ(nn_params1, nn_params1.list_, "weight2")
        print("in test 8 with bias %s"%str(nn_params1.with_bias),file=f)
        for i in nn_params1.list_:
            print(i.matrix,file=f)
        print(answer_nn_direct(nn_params1, [1, 1], 1))
        # print(answer_nn_direct(nn_params1, [0, 1], 1))
        print("*ON CONTRARY*")
        answer_nn_direct_on_contrary(nn_params1, [0], 1)
        f.close()

    def test_2(self):
        pass

if __name__ == '__main__':
    t = TestLay()
    u.main()

