from cv import cross_validation
from serial_deserial_func import compil_serializ
from nn_app import train, initiate_layers, get_min_square_err, answer_nn_direct, answer_nn_direct_on_contrary
from NN_params import NnParams   # импортруем параметры сети
from serial_deserial_func import deserializ
from nn_constants import bc_bufLen
#----------------------Основные параметры сети----------------------------------
# создать параметры сети
def create_nn_params():
    return NnParams()


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
    # compil_serializ(b_c, nn_params.list_,len(nn_map)-1,"wei_wei")


import unittest as u
class TestLay(u.TestCase):
    def setUp(self) -> None:
         self.nn_params = create_nn_params()
    def test_7(self):
        nn_map = (2, 3, 1)
        X = [[1, 1], [1, 0], [0, 1], [0, 0]]
        Y_or = [[1], [1], [1], [0]]
        Y_and = [[1], [0], [0], [0]]
        b_c = [0] * bc_bufLen  # буффер для сериализации матричных элементов и входов
        initiate_layers(self.nn_params, nn_map, len(nn_map))
        learn(b_c,self.nn_params, 0.07, 7, X, Y_and)
        compil_serializ(b_c, self.nn_params.list_, len(nn_map) - 1, "weight2" )
        print("in test_7 after learn matr")
        for i in self.nn_params.list_:
             print(i.matrix)

    def test_8(self):
       nn_params1=create_nn_params()
       deserializ(nn_params1, nn_params1.list_, "weight2")
       for i in nn_params1.list_:
          print(i.matrix)
       print(answer_nn_direct(nn_params1, [1, 1], 1))
       answer_nn_direct_on_contrary(nn_params1, [1], 1)
    # def test_9(self):
