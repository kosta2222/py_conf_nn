# nn_app.[py]
import math
import numpy as np
import struct as st
from nn_constants import max_in_nn, max_trainSet_rows, max_validSet_rows, max_rows_orOut, max_am_layer\
, max_am_epoch, max_am_objMse, max_stack_matrEl, max_stack_otherOp, bc_bufLen
from NN_params import NnParams   # импортруем параметры сети
from Nn_lay import nnLay   # импортируем слой
# import pdb
# pdb.set_trace()
np.random.seed(42)
# команды для operations
np.random.seed(42)
RELU = 1
RELU_DERIV = 2
SIGMOID = 3
SIGMOID_DERIV = 4
TRESHOLD_FUNC = 5
TRESHOLD_FUNC_DERIV = 6
LEAKY_RELU = 7
LEAKY_RELU_DERIV = 8
INIT_W_HE = 9
INIT_W_GLOROT = 10
DEBUG = 11
DEBUG_STR = 12
#----------------------дебаг, хелп функции--------------------------------------
def _0_(str_):
    print("Success ->", end = " ")
    print("function",str_)
#----------------------Основные параметры сети----------------------------------
# создать параметры сети
def create_nn_params():
    return NnParams()
nn_params = create_nn_params()
nn_params.inputNeurons = 2
nn_params.outputNeurons = 1
nn_params.lr = None
nn_map = (2, 3, 1)

#----------------------сериализации/десериализации------------------------------
# байт-коды для сериализации/десериализации-загрузка входов/выходов,загрузка элементов матрицы,сворачивание то есть создания ядра,остановка ВМ
push_i = 0
push_fl = 1
make_kernel = 2
stop = 3

b_c=[0] * bc_bufLen  # буффер для сериализации матричных элементов и входов/ds[jljd
def py_pack (op_i, val_i_or_fl):
    """
    Добавляет в b_c буффер байт-комманды и сериализованные матричные числа как байты
    :param op_i: байт-комманда
    :param val_i_or_fl: число для серелизации - матричный элемент или количество входов выходов
    :return: следующий индекс куда можно записать команду stop
    """
    ops_name = ['push_i', 'push_fl', 'make_kernel', 'stop']
    p = 0
    if op_i == push_fl:
        b_c[p] = st.pack('B', push_fl)
        p+=1
        for i in st.pack('>f', val_i_or_fl):
            b_c[p] = i.to_bytes(1, 'little')
            p+=1
    elif op_i == push_i:
        b_c[p] = st.pack('B', push_i))
        p+=1
        b_c[p] = st.pack('B', val_i_or_fl)
        p+=1
    elif op_i == make_kernel:
        b_c.append(st.pack('B', make_kernel))
        p+=1
    return p
def def dump_bc(f_name, p):
      b_c[p] = stop.to_bytes(1,"little")
      with open(f_name,'wb') as f:
        for i in b_c:
            f.write(i)


def make_kernel_f(list_:list, lay_pos, matrix_el_st:list,  ops_st:list,  sp_op):
     """
     Создает  ядро в векторе слоев
     :param list_: ссылка на вектор слоев
     :param lay_pos: позиция слоя (int)
     :param matrix_el_st: ссылка на стек матричных элементов
     :param ops_st: ссылка на стек входов/выходов
     :param sp_op: вершина стека входов/выходов(int)
     :return:
     """
     out = ops_st[sp_op]
     in_ = ops_st[sp_op - 1]
     list_[lay_pos].out = out
     list_[lay_pos].in_ = in_
     for  row in range(out):
        for elem in range(in_):
            list_[lay_pos].matrix[row][elem] = matrix_el_st[row * elem]   # десериализированная матрица
     _0_("make_kernel")


def vm_to_deserialize(list_:list, bin_buf:list):
     """
     Элемент виртуальной машины чтобы в вектор list_ матриц весов
     записать десериализированные из файла матрицы весов и смочь 
     пользоваться этим вектором для предсказания.
     :param list_: вектор матриц весов
     :param bin_buf: список байт - комманд из файла
     :return: 
     """
     ops_name =['push_i', 'push_fl', 'make_kernel', 'stop']
     matrix_el_st = [0] * max_stack_matrEl # стек для временного размещения элементов матриц из файла потом этот стек
                                           # сворачиваем в матрицу слоя после команды make_kernel
     ops_st = [0] * max_stack_otherOp      # стек для количества входов и выходов (это целые числа)
     ip = 0
     sp_ma = -1
     sp_op = -1
     op = -1
     arg = 0
     n_lay = 0
     op = bin_buf[ip]
     while (op != stop):
            # загружаем на стек количество входов и выходов ядра
            if  op == push_i:
                sp_op+=1
                ip+=1
                ops_st[sp_op] = bin_buf[ip]
                # break

            # загружаем на стек элементы матриц
            elif op == push_fl:
                arg = bin_buf[ip+1]

                # print("in vm op push_fl arg",st.unpack('>f',bytes(bin_buf[ip+1:4])))
                # print("len by",len(bin_buf[ip+1:4:1]),"p bin_buf",bin_buf[ip+1:4:1],"bin_buf",bin_buf,"ip=",ip)
                arg=len(bin_buf[:4:1])
                i_0=ip + 1
                i_1=ip+2
                i_2=ip+3
                i_3=ip+4
                arg=st.unpack('<f', bytes(list([i_0,i_1,i_2,i_3])))
                sp_ma+=1
                matrix_el_st[sp_ma] = arg[0]
                ip += 4
                # break

            # создаем одно ядро в массиве
            elif op == make_kernel:
                make_kernel_f(list_, n_lay, matrix_el_st, ops_st, sp_op)
                # переходим к следующему индексу ядра
                n_lay+=1
                # зачищаем стеки
                sp_op = -1
                sp_ma = -1
                # break

            # показываем на следующую инструкцию
            ip+=1
            op = bin_buf[ip]
     # также подсчитаем сколько у наc ядер
     nn_params.nlCount = n_lay
     # находим количество входов
     nn_params.inputNeurons=(nn_params.list_[0].in_)  #-1  # -1 зависит от биасов
     # находим количество выходов когда образовали сеть
     nn_params.outputNeurons=nn_params.list_[nn_params.nlCount-1].out
     _0_("vm")


def  deserializ( list_:list, f_name:str):
    bin_buf = [0] * bc_bufLen
    buf_str = b''
    with open(f_name, 'rb') as f:
        buf_str = f.read()
    j = 0
    for i in buf_str:
        bin_buf[j] = i
        j+=1
    # разборка байт-кода
    vm_to_deserialize(list_, bin_buf)

    _0_("vm_deserializ")

def copy_matrixAsStaticSquare_toRibon(src, dest, in_,out):
    for row in range(out):
       for elem in range(in_):
            dest[row * in_ + elem] = src[row][elem];
    _0_("copy_matrixAsStaticSquare_toRibon");

def compil_serializ(list_:nnLay, len_lst, f_name):
    in_=0
    out=0
    matrix=[0]*(max_in_nn * max_rows_orOut)
    for i in range(len_lst):
        in_=list_[i].in_
        out=list_[i].out
        py_pack(push_i,in_)
        py_pack(push_i,out)
        copy_matrixAsStaticSquare_toRibon(list_[i].matrix, matrix,in_,out)
        for j in range(in_ * out):
            py_pack(push_fl, matrix[j])
        py_pack(make_kernel, 0)
    dump_bc(f_name)

#----------------------------------------------------------------------


def calc_out_error(objLay:nnLay, targets:list):
    """
    Вычислить градиентную ошибку на выходном слое,записать этот параметр-вектор в обьект nnLay выходного слоя
    в параметр errors
    :param objLay: обьект слоя
    :param targets: вектор-ответы от учителя
    :return:
    """
    for row in range(objLay.out):
        nn_params.out_errors[row] = (objLay.hidden[row] - targets[row]) * operations(RELU_DERIV,objLay.cost_signals[row],0,0,0,"")

def calc_hid_error(objLay:nnLay, essential_gradients:list, entered_vals:list,i = 0):
    for elem in range(objLay.in_):
        for row in range(objLay.out):
            objLay.errors[elem]+=essential_gradients[row] * objLay.matrix[row][elem]  * operations(RELU_DERIV, entered_vals[elem], 0, 0, 0, "")
    # print("in calc_hid_error essential_gradients",essential_gradients)
    # print("in calc_hid_error entered_vals",entered_vals)
    # print("in calc_hid_error errors",objLay.errors)
    # print("in calc_hid_error matrix",objLay.matrix)

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

def get_min_square_err(out_nn:list,teacher_answ:list,n):
    sum=0
    for row in range(n):
        sum+=math.pow((out_nn[row] - teacher_answ[row]),2)
    return sum/n

def copy_vector(src:list,dest:list,n):
    for i in range(n):
        dest[i] = src[i]

def get_cost_signals(objLay:nnLay):
    return objLay.cost_signals

def get_hidden(objLay:nnLay):
    return objLay.hidden

def get_essential_gradients(objLay:nnLay):
    return objLay.errors

def calc_hid_zero_lay(zeroLay:nnLay,essential_gradients:list, i = 0):
    for elem in range(zeroLay.in_):
        for row in range(zeroLay.out):
            zeroLay.errors[elem]+=essential_gradients[row] * zeroLay.matrix[row][elem]
    print("in calc_hid_zero_lay",zeroLay.errors)

def upd_matrix(objLay:nnLay, entered_vals, i = 0):
    for row in range(objLay.out):
        for elem in range(objLay.in_):
            objLay.matrix[row][elem]-= nn_params.lr * objLay.errors[elem] * entered_vals[elem]
    print("in upd_matrix matrix state",objLay.matrix)

def feed_forwarding(ok:bool, debug:int):
    make_hidden(nn_params.list_[0],nn_params.inputs,debug)
    for i in range(1,nn_params.nlCount):
        make_hidden(nn_params.list_[i], get_hidden(nn_params.list_[i - 1]), debug)
    if ok:
        for i in range(nn_params.outputNeurons):
            print("%d item val %f"%(i + 1,nn_params.list_[nn_params.nlCount - 1].hidden[i]))
        return nn_params.list_[nn_params.nlCount - 1].hidden
    else:
         backpropagate()

# для теста, создать один слой
def create_one_lay():
    lay=nnLay()
    lay.in_=3
    lay.out=2
    lay.matrix=[[-1,1,4],[3,4,-7]]
    return lay

def train(in_:list,targ:list, debug):
    copy_vector(in_,nn_params.inputs,nn_params.inputNeurons)
    copy_vector(targ,nn_params.targets,nn_params.outputNeurons)
    # print("in train in_ vec",in_)
    # print("in train targ vec",targ)
    feed_forwarding(False, 1)

def answer_nn(in_:list, debug):
    copy_vector(in_,nn_params.inputs,nn_params.inputNeurons)
    # print("in answer_nn in_ vec",in_)
    out_nn=feed_forwarding(True, 1)
    return out_nn

# Получить вектор входов, сделать матричный продукт и матричный продукт пропустить через функцию активации,
# записать этот вектор в параметр слоя сети(hidden)
def make_hidden(objLay:nnLay, inputs, debug):
    tmp_v = 0
    val = 0
    for row in range(objLay.out):
        for elem in range(objLay.in_):
            # if elem==1:
            #     tmp_v+=objLay.matrix[row][1]
            # else:
                tmp_v+=objLay.matrix[row][elem] * inputs[elem]
        objLay.cost_signals[row] = tmp_v
        val = operations(RELU,tmp_v, 1, 0, 0, "")
        objLay.hidden[row] = val
        tmp_v = 0
        val = 0
    print("in make_hidden e",objLay.cost_signals)
    print("in make_hidden h",objLay.hidden)
    # print("in make_hidden matrix state",objLay.matrix)
"""
def backpropagate1():
    calc_out_error(nn_params.list_[nn_params.nlCount - 1], nn_params.targets)
    calc_hid_error(nn_params.list_[1], nn_params.out_errors, get_cost_signals(nn_params.list_[0]))
    calc_hid_zero_lay(nn_params.list_[0], get_essential_gradients(nn_params.list_[1]))
    upd_matrix(nn_params.list_[1],  get_cost_signals(nn_params.list_[1 - 1]))
    upd_matrix(nn_params.list_[0], nn_params.inputs)
"""
def backpropagate():
    calc_out_error(nn_params.list_[nn_params.nlCount - 1],nn_params.targets)
    for i in range(nn_params.nlCount - 1, 0, -1):
        if i == nn_params.nlCount - 1:
           calc_hid_error(nn_params.list_[i], nn_params.out_errors, get_cost_signals(nn_params.list_[i - 1]), i)
        else:
            calc_hid_error(nn_params.list_[i], get_essential_gradients(nn_params.list_[i + 1]), get_cost_signals(nn_params.list_[i - 1]), i)
    calc_hid_zero_lay(nn_params.list_[0], get_essential_gradients(nn_params.list_[1]), 0)
    for i in range(nn_params.nlCount - 1, 0, -1):
          upd_matrix(nn_params.list_[i],  get_cost_signals(nn_params.list_[i - 1]), i)

    upd_matrix(nn_params.list_[0], nn_params.inputs, 0)




# заполнить матрицу весов рандомными значениями по He, исходя из количесва входов и выходов,
# записать результат в вектор слоев(параметр matrix), здесь проблема матрица неправильно заполняется
def set_io(objLay:nnLay, inputs, outputs):
    objLay.in_=inputs
    objLay.out=outputs
    for row in range(outputs):
        for elem in range(inputs):
            objLay.matrix[row][elem] =operations(INIT_W_HE, inputs, 0, 0, 0, "")
    print("in set_io matrix", objLay.matrix)

def initiate_layers(nn_params:NnParams,network_map:tuple,size):
    """
    инициализировать вектор слоев используя функцию set_io исходя из кортежа, который должен описывать
    например количество входов, сколько нейронов(элементов) в следующем слое, сколько выводов
    :param network_map: кортеж карта слоев
    :param size: размер кортежа
    :return: None
    """
    in_ = 0
    out = 0
    nn_params.nlCount = size - 1
    nn_params.inputNeurons = network_map[0]
    nn_params.outputNeurons = network_map[nn_params.nlCount]
    set_io(nn_params.list_[0],network_map[0],network_map[1])
    for i in range(1, nn_params.nlCount ):# след. матр. д.б. (3,1) т.е. in(elems)=3 out(rows)=1
        in_ = network_map[i]
        out = network_map[i + 1]
        set_io(nn_params.list_[i], in_, out)


def learn(l_r, epochcs, train_set:list, target_set:list):
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
            train(X, Y, 1)
            mse = get_min_square_err(nn_params.list_[nn_params.nlCount - 1].hidden, Y, nn_params.outputNeurons)
            print("in learn mse",mse)
        if mse == 0:
            break
        iteration+=1
    cross_validation(train_set, target_set)
    compil_serializ(nn_params.list_,len(nn_map)-1,"wei_wei")

k = 0

def cross_validation(X_test: list, Y_test: list):
    print("***CV***")
    """
    Производит (кросс-валидацию) предсказание и сверка ответов
    по всему  Обучающий набор/тестовая выборка
    :param X_test: 2D Обучающий набор/тестовая выборка
    :param Y_test: 2D Набор ответов/тестовая выборка
    :return: аккуратность в процентах
    """
    scores=[0] * max_rows_orOut
    res = 0
    # out_nn=None
    res_acc = 1
    rows = len(X_test)
    for i in range(rows):
        x_test = X_test[i]
        y_test = Y_test[i]
        print("in cross val x_test",x_test)
        out_nn=answer_nn(x_test, 1)
        print("in cross val out_nn",out_nn)
        res=check_oneHotVecs(scores, out_nn, y_test, len(y_test))
    res_acc=calc_accur(scores,rows)
    print("Acсuracy:%f%s"%(res_acc,"%"))
    return res_acc
def check_oneHotVecs(scores:list, out_nn:list, y_test, len_)->int:
    tmp_elemOf_outNN_asHot = 0
    global k
    for col in range(len_):
        tmp_elemOf_outNN_asHot=out_nn[col]
        if (tmp_elemOf_outNN_asHot > 0 ) and (tmp_elemOf_outNN_asHot > 0.5 or tmp_elemOf_outNN_asHot == 1):
              tmp_elemOf_outNN_asHot = 1
        else:
            tmp_elemOf_outNN_asHot = 0
        if (tmp_elemOf_outNN_asHot == int(y_test[col])):
            scores[k] = 1
            k += 1
        else:
            break
def calc_accur(scores:list, rows)->float:
    accuracy=0
    sum=0
    for col in range(rows):
        sum+=scores[col]
    accuracy=sum / rows * 100
    return accuracy



import unittest as u
class TestLay(u.TestCase):
    def setUp(self) -> None:
        self.lay=create_one_lay()
    def test_7(self):
        X = [[1, 1], [1, 0], [0, 1], [0, 0]]
        Y = [[1], [1], [1], [0]]

        initiate_layers(nn_params, nn_map, len(nn_map))
        learn(0.07, 7, X, Y)
    # def test_8(self):
    #     X = [[1, 1], [1, 0], [0, 1], [0, 0]]
    #     out_nn = [[1], [1], [1], [0]]
    #     Y = [[1], [1], [1], [0]]
    #     cross_validation(X, Y)
    def test_9(self):
        deserializ(nn_params.list_, "wei")
        for i in nn_params.list_:
            print(i.matrix)

