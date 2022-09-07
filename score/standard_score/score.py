from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np

# Logistic ----------------------------------------------------------------
def logistic_training(X, Y):
    lgt_model = LogisticRegression()
    lgt_model.fit(X, Y)
    return lgt_model
    pass


def logistic_test(lgt_model: LogisticRegression, X, Y):
    return lgt_model.score(X, Y)


def read_in_file(synthetic_data, original_data, synth_num=30000, origin_num=10000, interval=5000):
    df_synth = pd.read_csv(synthetic_data)
    df_origin = pd.read_csv(original_data)
    matrix_synth = df_synth.fillna(interval).values.astype('float32')
    matrix_origin = df_origin.fillna(interval).values.astype('float32')
    matrix_synth = matrix_synth[:synth_num]
    matrix_origin = matrix_origin[:origin_num]
    matrix_synth_X = matrix_synth[:, :-1]
    matrix_synth_Y = matrix_synth[:, -1]
    matrix_origin_X = matrix_origin[:, :-1]
    matrix_origin_Y = matrix_origin[:, -1]

    return matrix_synth_X, matrix_synth_Y, matrix_origin_X, matrix_origin_Y
    pass


def logistic_scoring(X_train, Y_train, X_test, Y_test):

    lgt_model = logistic_training(X_train, Y_train)
    Y_hat = lgt_model.predict(X_test)
    # return accuracy_score(Y_test, Y_hat), f1_score(Y_test, Y_hat)
    return accuracy_score(Y_test, Y_hat)


# Multilayer Perceptron ----------------------------------------------------------

def mlp_training(X, Y, hidden=(50,)):
    mlp_model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=1000)
    mlp_model.fit(X, Y)
    return mlp_model

def mlp_testing(mlp_model: MLPClassifier, X, Y):
    return mlp_model.score(X, Y)

def mlp_scoring(X_train, Y_train, X_test, Y_test, hidden=(50,)):
    mlp_model = mlp_training(X_train, Y_train, hidden)
    Y_hat = mlp_model.predict(X_test)
    return accuracy_score(Y_test, Y_hat)
    # return accuracy_score(Y_test, Y_hat), f1_score(Y_test, Y_hat)

# Adaboost -------------------------------------------------------------------------

def ada_training(X, Y, n_est=50):
    ada_model = AdaBoostClassifier(n_estimators=n_est)
    ada_model.fit(X, Y)
    return ada_model

def ada_testing(ada_model: AdaBoostClassifier, X, Y):
    return ada_model.score(X, Y)

def ada_scoring(X_train, Y_train, X_test, Y_test, n_est=50):
    ada_model = ada_training(X_train, Y_train, n_est)
    Y_hat = ada_model.predict(X_test)
    return accuracy_score(Y_test, Y_hat)
    # return accuracy_score(Y_test, Y_hat), f1_score(Y_test, Y_hat)

# Decision tree --------------------------------------------

def decision_tree_training(X, Y, depth=20):
    ds_model = DecisionTreeClassifier(max_depth=depth)
    ds_model.fit(X, Y)
    return ds_model

def decision_tree_testing(ds_model: DecisionTreeClassifier, X, Y):
    return ds_model.score(X, Y)

def decision_tree_scoring(X_train, Y_train, X_test, Y_test, depth=20):
    ds_model = decision_tree_training(X_train, Y_train, depth)
    Y_hat = ds_model.predict(X_test)
    return accuracy_score(Y_test, Y_hat)
    # return accuracy_score(Y_test, Y_hat), f1_score(Y_test, Y_hat)

def score_results(synthetic_data, original_data, synth_num=30000, origin_num=10000, n_est=50, hidden=50, depth_ds=20):
    X_train, Y_train, X_test, Y_test = read_in_file(synthetic_data, original_data, synth_num, origin_num)
    lgt_score = logistic_scoring(X_train, Y_train, X_test, Y_test)
    mlp_score = mlp_scoring(X_train, Y_train, X_test, Y_test, hidden)
    ada_score = ada_scoring(X_train, Y_train, X_test, Y_test, n_est)
    ds_score = decision_tree_scoring(X_train, Y_train, X_test, Y_test, depth_ds)
    return ada_score, ds_score, lgt_score, mlp_score


def data_devide_column(matrix_synth, matrix_origin, column):
    matrix_synth_X = np.delete(matrix_synth, column, 1)
    matrix_synth_Y = matrix_synth[:, column]
    matrix_origin_X = np.delete(matrix_origin, column, 1)
    matrix_origin_Y = matrix_origin[:, column]
    return matrix_synth_X, matrix_synth_Y, matrix_origin_X, matrix_origin_Y
    pass


def read_in_file_only(synthetic_data, original_data,synth_num=30000, origin_num=100000, interval=-1):
    df_synth = pd.read_csv(synthetic_data)
    df_origin = pd.read_csv(original_data)
    matrix_synth = df_synth.fillna(interval).values.astype('float32')
    matrix_origin = df_origin.fillna(interval).values.astype('float32')
    matrix_synth = matrix_synth[:synth_num]
    matrix_origin = matrix_origin[:origin_num]
    return matrix_synth, matrix_origin


def read_in_1file_only(data_path, taken_num, interval=-1):
    df_data = pd.read_csv(data_path)
    matrix_data = df_data.fillna(interval).values.astype('float32')
    indices = np.arange(matrix_data.shape[0])
    indices = np.random.choice(indices, taken_num, replace=False)
    matrix_data = matrix_data[indices]
    return matrix_data, df_data.columns


def score_results_column(matrix_origin, matrix_synth, n_est=50, hidden=50,
                         depth_ds=20):
    n_columns = matrix_synth.shape[1]
    list_lgt = []
    list_mlp = []
    list_ada = []
    list_ds = []
    for column in range(n_columns):
        print("Processing %d"%column)
        X_train, Y_train, X_test, Y_test = data_devide_column(matrix_synth, matrix_origin, column)
        unique_elements, counts_elements = np.unique(Y_train, return_counts=True)
        if unique_elements.shape[0] < 2:
            list_lgt.append(-1)
            list_mlp.append(-1)
            list_ada.append(-1)
            list_ds.append(-1)
            continue
        lgt_score = logistic_scoring(X_train, Y_train, X_test, Y_test)
        mlp_score = mlp_scoring(X_train, Y_train, X_test, Y_test, hidden)
        ada_score = ada_scoring(X_train, Y_train, X_test, Y_test, n_est)
        ds_score = decision_tree_scoring(X_train, Y_train, X_test, Y_test, depth_ds)
        list_lgt.append(lgt_score)
        list_mlp.append(mlp_score)
        list_ada.append(ada_score)
        list_ds.append(ds_score)
    return list_lgt, list_mlp, list_ada, list_ds
    pass
