from NN_params import NnParams   # импортруем параметры сети
from serial_deserial_func import deserializ
from nn_constants import bc_bufLen, RELU, LEAKY_RELU, SIGMOID, TAN
from lear_func import initiate_layers, answer_nn_direct, answer_nn_direct_on_contrary
from serial_deserial_func import compil_serializ
from fit import fit
"""
X и Y означают двухмернй список обучения и ответов соответственно
x_* и y_* - просто списки из этих матриц
"""


# создать параметры сети
def create_nn_params():
    return NnParams()


def turn_on_lamp():
    print("Lampa vkluchena!")


len_=10

push_i = 0
push_fl = 1
push_str = 2
calc_vecs = 3
fit_ = 4
recogn = 5
stop = 6

ops=["push_i","push_fl", "push_str", "calc_vecs","fit","recogn"]
def console():
        b_c = [0] * len_* 3
        input_ = ""
        # splitted_cmd и splitted_cmd_src - т.к. работаем со статическим массивом
        splitted_cmd: list = [''] * 2
        splitted_cmd_src:list = None
        main_cmd = '<uninitialized>'
        par_cmd = '<uninitialized>'
        idex_of_bytecode_is_bytecode = 0
        cmd_in_ops = '<uninitialized>'
        pos_bytecode = -1
        shell_is_running = True
        # exit_flag = False
        # while shell_is_running:
        print("Zdravstvuite ya sostavitel bait-coda dla etoi programmi")
        print("r vipolnit")
        print("Naberite exit dlya vihoda")
        print("Dostupnie codi:")
        for c in ops:
            print(c, end=' ')
        print()
        while shell_is_running:

            input_ = input(">>>")
            # полностью выходим из программы
            if input_== "exit":
                # exit_flag = True
                break
            # выполняем байткод вирт-машиной
            elif input_== "r":
                pos_bytecode+= 1
                b_c[pos_bytecode] = stop
                print("b_c",b_c)
                vm_proc_to_learn(b_c)
                # b_c.clear()
                # print("b_c2",b_c)
                pos_bytecode = -1
            splitted_cmd_src = input_.split()
            for pos_to_write in range(len(splitted_cmd_src)):
                splitted_cmd[pos_to_write] = splitted_cmd_src[pos_to_write]
            main_cmd = splitted_cmd[0]
            par_cmd = splitted_cmd[1]
            # Ищем код в списке код-строку
            for idex_of_bytecode_is_bytecode in range(len(ops)):
                cmd_in_ops = ops[idex_of_bytecode_is_bytecode]
                if cmd_in_ops == main_cmd:
                    pos_bytecode += 1
                    # формируем числовой байт-код и если нужно значения параметра
                    b_c[pos_bytecode] = idex_of_bytecode_is_bytecode
                    if par_cmd != '':
                        pos_bytecode += 1
                        b_c[pos_bytecode] = par_cmd
                # Очищаем
                splitted_cmd[0] = ''
                splitted_cmd[1] = ''
            # pos_bytecode+= 1
            # if exit_flag:
            #    break
    # print("bye)")
X=[]
Y=[]
def vm_proc_to_learn(b_c:list):
    nn_params = create_nn_params()
    nn_params.with_bias = False
    nn_params.with_adap_lr = True
    nn_params.lr = 0.01
    nn_params.act_fu = TAN
    nn_params.alpha_sigmoid = 0.056
    nn_in_amount = 20
    nn_map = (nn_in_amount, 8, 2)
    initiate_layers(nn_params, nn_map, len(nn_map))

    ip=0
    sp=-1
    sp_str=-1
    sp_fl=-1
    steck=[0]*len_
    steck_fl=[0.0]*len_
    steck_str=['']*len_
    op=0
    op=b_c[ip]
    while True:
        if op==push_i:
            sp+=1
            ip+=1
            steck[sp]=int(b_c[ip]) # Из строкового параметра
        elif op == push_fl:
            sp_fl += 1
            ip += 1
            steck_fl[sp_fl] = float(b_c[ip])  # Из строкового параметра
        elif op==push_str:
            sp_str+= 1
            ip += 1
            steck_str[sp_str] = b_c[ip]
        #  вычисление векторов это еще добавление к тренировочным матрицам
        elif op==calc_vecs:
            ord_as_devided_val = 0.0
            float_x = [0] * nn_in_amount
            str_y = [0, 1]
            Y.append(str_y)
            str_x=steck_str[sp_str]
            sp_str-=1
            cn_char = 0
            for chr in str_x:
                ord_as_devided_val = ord(chr) / 255
                float_x[cn_char]= ord_as_devided_val
                cn_char+= 1
            X.append(float_x)

            print("in vm in calc_ve:",X,Y)
        elif op==stop:
            return

        elif op == fit_:
           X_new =[]
           x_new = [0] * nn_in_amount
           for i in range(len(X)):
               X_new.append(x_new)

           for row in range(len(X)):
               for elem in range(nn_in_amount):
                   X_new[row][elem] = X[row][elem]
           fit(None, nn_params, 10, X_new, Y, 100)
           # X_new.clear()
        elif op == recogn:
            float_x = [0] * nn_in_amount
            str_x = steck_str[sp_str]
            sp_str-= 1
            cn_char = 0
            for chr in str_x:
                ord_as_devided_val = ord(chr) / 255
                float_x[cn_char] = ord_as_devided_val
                cn_char+=1
            print("*In vm in recogn",answer_nn_direct(nn_params, float_x, 1))
        else:
            print("Unknown bytecode -> %d"%op)
            return
        ip+= 1
        op = b_c[ip]
       
if __name__ == '__main__':
    console()