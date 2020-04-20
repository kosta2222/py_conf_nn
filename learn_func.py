from cv import cross_validation
from serial_deserial_func import compil_serializ
from nn_app import train, initiate_layers, get_min_square_err, answer_nn_direct, answer_nn_direct_on_contrary,\
get_mean
from NN_params import NnParams   # импортруем параметры сети
from serial_deserial_func import deserializ
from nn_constants import bc_bufLen, RELU, LEAKY_RELU, SIGMOID, TAN
#----------------------Основные параметры сети----------------------------------
# создать параметры сети
def create_nn_params():
    return NnParams()


def learn(b_c:list, nn_params, epochcs, train_set:list, target_set:list):
    error = 0.0
    iteration: int = 0
    n_epochs = []
    n_mse = []
    A = nn_params.lr  # A - здесь коэффициент обучения
    acc_shurenss = 100
    acc = 0
    alpha=0.99
    beta=1.01
    gama=1.01
    delta_E_spec=0
    E_spec=0
    E_spec_t_minus_1=0
    A_t_minus_1=0
    with_adap_lr = True
    with_bias = False
    out_nn:list=None
    x:list = None  # 1d вектор из матрицы обучения
    y:list = None  # 1d вектор из матрицы ответов от учителя
    hei_target_set = len(target_set)
    while True:
        print("epocha:", iteration)
        for i in range(hei_target_set):
            if iteration == 0:
                E_spec_t_minus_1 = E_spec
                A_t_minus_1 = A
            nn_params.lr = A
            x = train_set[i]
            y = target_set[i]
            print("in learn x",x)
            print("in learn y",y)
            train(nn_params, x, y, 1)
            out_nn = nn_params.list_[nn_params.nlCount - 1].hidden
            print("in learn",out_nn)
            mse = get_min_square_err(out_nn, y, nn_params.outputNeurons)
            print("in learn mse",mse)
            if mse == 0:
               # break
               pass
            if nn_params.with_adap_lr:
                E_spec = get_mean(out_nn, y, len(y))
                delta_E_spec = E_spec - gama * E_spec_t_minus_1
                if delta_E_spec > 0:
                    A = alpha * A_t_minus_1
                else:
                    A = beta * A_t_minus_1
                print("A",A)
                A_t_minus_1 = A
                E_spec_t_minus_1 = E_spec
        acc = cross_validation(nn_params, train_set, target_set)
        if acc == acc_shurenss:
            break
        iteration+=1
    print("***CV***")
    cross_validation(nn_params, train_set, target_set)
    # compil_serializ(b_c, nn_params.list_,len(nn_map)-1,"wei_wei")


import unittest as u
class TestLay(u.TestCase):
    def setUp(self) -> None:
         self.nn_params = create_nn_params()
         self.nn_params.with_bias = False
         self.nn_params.with_adap_lr = True
         self.nn_params.lr = 0.07
    def test_7(self):
        nn_map = (2, 3, 1)
        X = [[1, 1], [1, 0], [0, 1], [0, 0]]
        Y_or = [[1], [1], [1], [0]]
        # Y_and = [[1], [0], [0], [0]]
        # X = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0],
        #      [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1],
        #      [1, 1, 1, 0], [1, 1, 1, 1]]
        # Y = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0],
        #      [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1],
        #      [1, 1, 1, 0], [1, 1, 1, 1]]

        b_c = [0] * bc_bufLen  # буффер для сериализации матричных элементов и входов
        initiate_layers(self.nn_params, nn_map, len(nn_map))
        learn(b_c,self.nn_params, 7, X, Y_or)
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
       print(answer_nn_direct(nn_params1, [0, 1], 1))
       print("*ON CONTRARY*")
       answer_nn_direct_on_contrary(nn_params1, [0], 1)
    # def test_9(self):
if __name__ == '__main__':
    t = TestLay()
    u.main()
