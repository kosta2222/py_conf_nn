from cv import cross_validation
from serial_deserial_func import compil_serializ
from nn_app import train, initiate_layers, get_min_square_err
from NN_params import NnParams   # импортруем параметры сети
from nn_constants import bc_bufLen
#----------------------Основные параметры сети----------------------------------


# создать параметры сети
def create_nn_params():
    return NnParams()
nn_params = create_nn_params()
nn_params.inputNeurons = 2
nn_params.outputNeurons = 1
nn_params.lr = None
nn_map = (2, 3, 1)

def learn(b_c:list, nn_params, l_r, epochcs, train_set:list, target_set:list):
    error = 0.0
    iteration: int = 0
    n_epochs = []
    n_mse = []
    nn_params.lr = l_r
    while (iteration < epochcs):
        print("epocha:", iteration)
        for i in range(len(target_set)):
            X = train_set[i]
            Y = target_set[i]
            print("in learn X",X)
            print("in learn Y",Y)
            train(nn_params, X, Y, 1)
            mse = get_min_square_err(nn_params.list_[nn_params.nlCount - 1].hidden, Y, nn_params.outputNeurons)
            print("in learn mse",mse)
        if mse == 0:
            break
        iteration+=1
    cross_validation(nn_params, train_set, target_set)
    compil_serializ(b_c, nn_params.list_,len(nn_map)-1,"wei_wei")

import unittest as u
class TestLay(u.TestCase):
    # def setUp(self) -> None:
    #     self.lay=create_one_lay()
    def test_7(self):
        X = [[1, 1], [1, 0], [0, 1], [0, 0]]
        Y = [[1], [1], [1], [0]]
        b_c = [0] * bc_bufLen  # буффер для сериализации матричных элементов и входов
        initiate_layers(nn_params, nn_map, len(nn_map))
        learn(b_c,nn_params, 0.7, 7, X, Y)
    # def test_8(self):
    #     X = [[1, 1], [1, 0], [0, 1], [0, 0]]
    #     out_nn = [[1], [1], [1], [0]]
    #     Y = [[1], [1], [1], [0]]
    #     cross_validation(X, Y)
    # def test_9(self):
    #     deserializ(nn_params.list_, "wei")
    #     for i in nn_params.list_:
    #         print(i.matrix)

