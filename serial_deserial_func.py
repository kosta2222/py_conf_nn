from work_with_arr import copy_matrixAsStaticSquare_toRibon
from nn_constants import bc_bufLen, max_in_nn, max_rows_orOut, max_stack_matrEl, max_stack_otherOp,\
    push_i, push_fl, make_kernel, with_bias, stop
from Nn_lay import nnLay
import struct as st
from NN_params import NnParams
from util_func import _0_
#----------------------сериализации/десериализации------------------------------



p=0

def py_pack (b_c:list, op_i, val_i_or_fl):
    """
    Добавляет в b_c буффер байт-комманды и сериализованные матричные числа как байты
    :param op_i: байт-комманда
    :param val_i_or_fl: число для серелизации - матричный элемент или количество входов выходов
    :return: следующий индекс куда можно записать команду stop
    """
    global p
    ops_name = ['push_i', 'push_fl', 'make_kernel', 'with_bias', 'stop']
    print("in py_pack op",ops_name[op_i],"val_i_or_fl",val_i_or_fl)
    if op_i == push_fl:
        b_c[p] = st.pack('B', push_fl)
        p+=1
        for i in st.pack('<f', val_i_or_fl):
            b_c[p] = i.to_bytes(1, 'little')
            p+=1
    elif op_i == push_i:
        b_c[p] = st.pack('B', push_i)
        p+=1
        b_c[p] = st.pack('B', val_i_or_fl)
        p+=1
    elif op_i == make_kernel:
        b_c[p] = st.pack('B', make_kernel)
        p+=1
    elif op_i == with_bias:
        b_c[p] = st.pack('B', with_bias)
        p+=1



def  dump_bc(b_c, f_name):
  global p
  b_c[p] = stop.to_bytes(1,"little")
  p+=1
  with open(f_name,'wb') as f:
     for i in range(p):
         f.write(b_c[i])


def make_kernel_f(nn_params:NnParams, list_:list, lay_pos, matrix_el_st:list,  ops_st:list,  sp_op):
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


def vm_to_deserialize(nn_params:NnParams, list_:list, bin_buf:list):
    """
    Элемент виртуальной машины чтобы в вектор list_ матриц весов
    записать десериализированные из файла матрицы весов и смочь
    пользоваться этим вектором для предсказания.
    :param list_: вектор матриц весов
    :param bin_buf: список байт - комманд из файла
    :return:
    """
    print("in vm_to_deserialize")
    ops_name =['push_i', 'push_fl', 'make_kernel','with_bias', 'stop']
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
        # чтение операции с параметром
        print(ops_name[op])
        if  op == push_i:
            sp_op+=1
            ip+=1
            ops_st[sp_op] = bin_buf[ip]
        # загружаем на стек элементы матриц
        # чтение операции с параметром
        elif op == push_fl:
            i_0 = bin_buf[ip + 1]
            i_1 = bin_buf[ip + 2]
            i_2 = bin_buf[ip + 3]
            i_3 = bin_buf[ip + 4]
            arg=st.unpack('<f', bytes(list([i_0, i_1, i_2, i_3])))
            sp_ma+=1
            matrix_el_st[sp_ma] = arg[0]
            ip += 4
        # создаем одно ядро в массиве
        # пришла команда создать ядро
        elif op == make_kernel:
            make_kernel_f(nn_params, list_, n_lay, matrix_el_st, ops_st, sp_op)
            # переходим к следующему индексу ядра
            n_lay+=1
            # зачищаем стеки
            sp_op = -1
            sp_ma = -1
        # пришла команда узнать пользуемся ли биасами
        # надо извлечь параметр
        elif op == with_bias:
            is_with_bias = ops_st[sp_op]
            sp_op-=1
            if is_with_bias == 1:
                nn_params.with_bias = True
            elif is_with_bias == 0:
                nn_params.with_bias = False
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


def deserializ(nn_params:NnParams, list_:list, f_name:str):
    bin_buf = [0] * bc_bufLen
    buf_str = b''
    with open(f_name, 'rb') as f:
        buf_str = f.read()
    j = 0
    for i in buf_str:
        bin_buf[j] = i
        j+=1
    # разборка байт-кода
    vm_to_deserialize(nn_params, list_, bin_buf)
    _0_("vm_deserializ")


def compil_serializ(nn_params:NnParams, b_c:list, list_:nnLay, len_lst, f_name):
    in_=0
    out=0
    p=0
    with_bias_i = 0
    stub = 0
    matrix=[0]*(max_in_nn * max_rows_orOut)
    if nn_params.with_bias:
        with_bias_i = 0
    else:
        with_bias_i = 1
    py_pack(b_c, push_i, with_bias_i)
    py_pack(b_c, with_bias, stub)
    for i in range(len_lst):
        in_=list_[i].in_
        out=list_[i].out
        py_pack(b_c, push_i,in_)
        py_pack(b_c, push_i,out)
        copy_matrixAsStaticSquare_toRibon(list_[i].matrix, matrix, in_, out)
        for j in range(in_ * out):
            py_pack(b_c, push_fl, matrix[j])
        py_pack(b_c, make_kernel, stub)
    dump_bc(b_c, f_name)
#----------------------------------------------------------------------
