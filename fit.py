from cross_val_eval import evaluate
from lear_func import train, initiate_layers, get_min_square_err, answer_nn_direct, answer_nn_direct_on_contrary,\
get_mean



def fit(b_c:list, nn_params, epochcs, train_set:list, target_set:list, accuracy_shureness:int):
    iteration: int = 0
    A = nn_params.lr
    out_nn:list=None
    x:list = None  # 1d вектор из матрицы обучения
    y:list = None  # 1d вектор из матрицы ответов от учителя
    alpha = 0.99
    beta = 1.01
    gama = 1.01
    hei_target_set = len(target_set)
    E_spec = 0
    while True:
        print("epocha:", iteration)
        for i in range(hei_target_set):
            x = train_set[i]
            y = target_set[i]
            print("in learn x",x)
            print("in learn y",y)
            train(nn_params, x, y, 1)
            out_nn = nn_params.list_[nn_params.nlCount - 1].hidden
            if nn_params.with_adap_lr:
                if iteration == 0:
                    E_spec_t_minus_1 = E_spec
                    A_t_minus_1 = A
                E_spec = get_mean(out_nn, y, len(y))
                delta_E_spec = E_spec - gama * E_spec_t_minus_1
                if delta_E_spec > 0:
                    A = alpha * A_t_minus_1
                else:
                    A = beta * A_t_minus_1
                    print("A", A)
                    A_t_minus_1 = A
                    E_spec_t_minus_1 = E_spec
            nn_params.lr = A
            print("in learn",out_nn)
            mse = get_min_square_err(out_nn, y, nn_params.outputNeurons)
            print("in learn mse",mse)
        # if mse == 0:
        #     break
        acc = evaluate(nn_params, train_set, target_set)
        if acc == accuracy_shureness:
            break
        iteration+=1
    print("***CV***")
    evaluate(nn_params, train_set, target_set)
    # compil_serializ(b_c, nn_params.list_,len(nn_map)-1,"wei_wei")



