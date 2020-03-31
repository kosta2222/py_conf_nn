#NN_params.[py]
from nn_constants import max_in_nn,max_trainSet_rows,max_validSet_rows,max_rows_orOut,max_am_layer,max_am_epoch,max_am_objMse,max_stack_matrEl,max_stack_otherOp,bc_bufLen
from Nn_lay import nnLay
# Параметры сети
class    NnParams:
    def __init__(self):
        self.list_=[]
        for i in range(max_am_layer):
            ob_lay=nnLay()
            self.list_.append(ob_lay);  # вектор слоев
        self.inputNeurons=0;  # количество
          #выходных
          #нейронов
        self.outputNeurons=0;  # количество
        #входных
        #нейронов
        self.nlCount=0;  # количество
        #слоев
        self.inputs=[0]*(max_in_nn);  # входа сети
        self.targets=[0]*(max_rows_orOut);  # ответы от учителя
        self.out_errors = [0] * (max_rows_orOut)  # вектор ошибок слоя
        self.lr=0;  # коэффициент
          #обучения
