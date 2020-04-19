from nn_constants import max_rows_orOut
from nn_app import answer_nn_direct
from NN_params import NnParams
k = 0
def cross_validation(nn_params:NnParams, X_test: list, Y_test: list):
    # print("***CV***")
    """
    Производит (кросс-валидацию) предсказание и сверка ответов
    по всему  Обучающий набор/тестовая выборка
    :param X_test: 2D Обучающий набор/тестовая выборка
    :param Y_test: 2D Набор ответов/тестовая выборка
    :return: аккуратность в процентах
    """
    # scores=[0] * max_rows_orOut
    scores = []
    res = 0
    out_nn=None
    res_acc = 1
    rows = len(X_test)
    wi_y_test = len(Y_test[0])
    n = 0
    answer = 0
    for i in range(rows):
        x_test = X_test[i]
        y_test = Y_test[i]
        # print("in cross val x_test",x_test)
        out_nn=answer_nn_direct(nn_params, x_test, 1)
        is_2vecs_eq = 0
        # print("in cross val out_nn",out_nn)
        is_2vecs_eq = check_2vecs_to_eq(out_nn, y_test, nn_params.outputNeurons)
        scores.append(is_2vecs_eq)
        # res=check_oneHotVecs(scores, out_nn, y_test, len(y_test))
        # for i in range(wi_y_test):
        #     n = out_nn[i]
        #     answer = y_test[i]
        #     if (n > 0.5):
        #         n = 1
        #         print("output vector[ %f ] " % 1, end=' ')
        #     else:
        #         n = 0
        #         print("output vector[ %f ] " % 0, end=' ');
        #     print("expected [ %f ]\n" % answer);
        #     if n == answer:
        #         scores.append(1)
        #     else:
        #         scores.append(0)


    res_acc = sum(scores) / rows * 100

    print("Acсuracy:%f%s"%(res_acc,"%"))
    return res_acc
def check_2vecs_to_eq(out_nn:list, y_test, len_)->int:
    res_check_is_2vecs_eq= 1

    for col in range(len_):
       elem_out_nn = out_nn[col]
       elem_y_test =y_test[col]
       if elem_out_nn > 0 and elem_out_nn > 0.5:
              elem_out_nn = 1
       else:
           elem_out_nn = 0
       if elem_out_nn != elem_y_test:
           res_check_is_2vecs_eq = 0
           break
       else:
           pass
    return res_check_is_2vecs_eq

# def calc_accur(scores:list, rows)->float:
#     accuracy=0
#     sum=0
#     for col in range(rows):
#         sum+=scores[col]
#     accuracy=sum / rows * 100
#     return accuracy
