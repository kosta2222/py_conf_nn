from cv import cross_validation
from serial_deserial_func import compil_serializ
from nn_app import train, initiate_layers, get_min_square_err, answer_nn_direct, answer_nn_direct_on_contrary,\
get_mean
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
    A = 0.01
    exit_flag = False
    acc_shurenss = 100
    acc = 0
    alpha=0.99
    beta=1.01
    gama=1.01
    delta_E_spec=0
    Z=0
    Z_t_minus_1=0
    A_t_minus_1=0
    with_adap_lr = True
    out_nn:list=None
    while True:#(iteration < epochcs):
        print("epocha:", iteration)
        for i in range(len(target_set)):
            if iteration == 0:
                Z_t_minus_1 = Z
                A_t_minus_1 = A
            nn_params.lr = A
            X = train_set[i]
            Y = target_set[i]
            print("in learn X",X)
            print("in learn Y",Y)
            train(nn_params, X, Y, 1)
            out_nn = nn_params.list_[nn_params.nlCount - 1].hidden
            print("in learn",out_nn)
            mse = get_min_square_err(out_nn, Y, nn_params.outputNeurons)
            print("in learn mse",mse)
            if mse == 0:
            # break
               pass
            if with_adap_lr:
                Z = get_mean(out_nn, Y, len(Y))
                delta_E_spec = Z - gama * Z_t_minus_1
                if delta_E_spec > 0:
                    A = alpha * A_t_minus_1
                else:
                    A = beta * A_t_minus_1
                print("A",A)
            A_t_minus_1 = A
            Z_t_minus_1 = Z
        acc = cross_validation(nn_params, train_set, target_set)
        if acc == acc_shurenss:
            exit_flag = True
            break
            pass
        # if exit_flag == True:
        #     break
        iteration+=1
    print("***CV***")
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
        learn(b_c,self.nn_params, 0.07, 7, X, Y_or)
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
       print("*ON CONTRARY*")
       answer_nn_direct_on_contrary(nn_params1, [0], 1)
    # def test_9(self):
