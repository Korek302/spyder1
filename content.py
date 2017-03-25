# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial

def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''
    N = len(x)
    dm = [(y[i] - polynomial(x[i], w))**2 for i in range(N)]
    return (1/N) * np.sum(dm)


def design_matrix(x_train,M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    N = len(x_train)
    
    m = np.zeros((N, M+1))
    for i in range(0, N):
        for j in range(0, M+1):
            m[i][j] = x_train[i]**j
    
    return m

def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''
    dm = design_matrix(x_train, M)
    temp = dm.transpose() @ dm
    w = (np.linalg.inv(temp) @ dm.transpose()) @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return (w, err)


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''
    dm = design_matrix(x_train, M)
    temp = dm.transpose() @ dm
    temp2 = regularization_lambda * np.eye(np.shape(temp)[0])
    w = ((np.linalg.inv(temp + temp2)) @ dm.transpose()) @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return (w, err)


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''
    
    w = []
    train_err = np.inf
    val_err = np.inf
    for i in range(7):
        (t_v, t_e) = least_squares(x_train, y_train, i)
        v_err = mean_squared_error(x_val, y_val, t_v)
        if(v_err < val_err):
            w = t_v
            train_err = t_e
            val_err = v_err
    
    return (w, train_err, val_err)


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''
    
    w = []
    train_err = np.inf
    val_err = np.inf
    regularization_lambda = np.inf
    for i in lambda_values:
        (t_v, t_e) = regularized_least_squares(x_train, y_train, M, i)
        v_err = mean_squared_error(x_val, y_val, t_v)
        if(v_err < val_err):
            w = t_v
            train_err = t_e
            val_err = v_err
            regularization_lambda = i
    
    return (w, train_err, val_err, regularization_lambda)

