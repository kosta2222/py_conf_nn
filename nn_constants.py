# nn_constants.[py]
# Параметры статических массивов,количества слоев,количества эпох
max_in_nn = 4
max_trainSet_rows = 4
max_validSet_rows = 10
max_rows_orOut = 4
max_am_layer = 7
max_am_epoch = 25
max_am_objMse = max_am_epoch
max_stack_matrEl = 256
max_stack_otherOp = 4
bc_bufLen = 256 * 2
elems_of_img=10000
# команды для operations
RELU = 1
RELU_DERIV = 2
SIGMOID = 3
SIGMOID_DERIV = 4
SIGMOID_VER1 = 5
SIGMOID_VER1_MEASURE = 6
TRESHOLD_FUNC = 7
TRESHOLD_FUNC_DERIV = 8
LEAKY_RELU = 9
LEAKY_RELU_DERIV = 10
INIT_W_HE = 11
INIT_W_GLOROT_MY = 12
DEBUG = 13
DEBUG_STR = 14
INIT_W_HABR = 15
INIT_W_MY = 16
INIT_W_UNIFORM = 17
TAN =18
TAN_DERIV =19
NOP = 20
INIT_W_HE_MY = 21

# байт-коды для сериализации/десериализации-загрузка входов/выходов,загрузка элементов матрицы,сворачивание то есть создания ядра, есть ли биасы,остановка ВМ
push_i = 1
push_fl = 2
make_kernel = 3
with_bias = 4
determe_act_func = 5
determe_alpha_leaky_relu = 6
determe_alpha_sigmoid = 7
determe_alpha_and_beta_tan = 8
stop = 9
