import pandas as pd
import numpy as np
import sklearn

def nmi_standard(df_col1: pd.DataFrame, df_col2: pd.DataFrame):
    var1 = df_col1.values
    var2 = df_col2.values
    nmi_value = sklearn.metrics.normalized_mutual_info_score(var1, var2, average_method='max')
    return nmi_value

# def nmi_tab(df_col1: pd.DataFrame, df_col2: pd.DataFrame):
#     var1 = df_col1.values
#     var2 = df_col2.values
#     mi_value = sklearn.metrics.normalized_mutual_info_score(var1, var2)
#     norm_value =


def nmi_distance(df_data1: pd.DataFrame, df_data2: pd.DataFrame, meta):
    # binning this two data
    columns1 = df_data1.columns
    columns2 = df_data2.columns
    n = len(columns1)
    for i in range(n):
        if columns2[i] != columns1[i]:
            print("columns must be the same")
            exit(1)
    for column in columns1:
        meta_column = meta[column]
        if meta_column['type'] == 'enum' or meta_column['type'] == 'ordinal':
            continue
        max = meta_column['max']
        min = meta_column['min']
        n = int((max - min)/20)
        if n == 0:
            n = 1
        df_data1[column] = pd.cut(df_data1[column], n)
        df_data2[column] = pd.cut(df_data2[column], n)


    # calculate pairwise nmi
    sum_p_nmi = 0.0
    sum_square_p_nmi = 0.0
    distances = []
    squar_distances = []
    count = 0
    for i in range(len(columns1)):
        for j in range(i+1, len(columns1), 1):
            column1 = columns1[i]
            column2 = columns1[j]
            nmi_value_1 = nmi_standard(df_data1[column1], df_data1[column2])
            nmi_value_2 = nmi_standard(df_data2[column1], df_data2[column2])
            distance = abs(nmi_value_1 - nmi_value_2)
            distances.append(distance)
            squar_distances.append(distance**2)
            sum_p_nmi += distance
            sum_square_p_nmi += distance**2
    nmi_pairwise_value = sum_p_nmi/len(distances)
    square_p_nmi = sum_square_p_nmi/len(squar_distances)

    return nmi_pairwise_value, square_p_nmi, distances, squar_distances

def mutual_minority_counts(train_df, synth_df, pair):
    reversed_pair = [pair[1], pair[0]]

    pair_train_df = train_df[pair]

    # pair_synthesized_df = synth_df[pair]
    reversed_pair_train_df = train_df[reversed_pair]
    # reversed_pair_synthesized_df = synth_df[reversed_pair]

    train_total = 0
    synthesized_total = 0

    column_0_unique = pair_train_df[pair[0]].unique()
    count = 0
    count_synth = 0
    for value in column_0_unique:
        column_1_unique = pair_train_df[pair[1]].unique()
        frequency_0 = pair_train_df[pair_train_df[pair[0]] == value].value_counts()
        # frequency_0 = pair_train_df.query(pair[0] + '==' + str(value)).value_counts()
        frequent_total_0 = np.sum(frequency_0.values)

        for value_1 in column_1_unique:
            if ((value, value_1) in frequency_0.keys()):
                num_cat_1 = len(frequency_0.keys())
                prob = frequency_0[(value, value_1)] / frequent_total_0
                # threshold = 1/(10*num_cat_1)
                threshold = 1 / num_cat_1
                if (prob < threshold):
                    frequency_1 = reversed_pair_train_df[reversed_pair_train_df[pair[1]] == value_1].value_counts()
                    # frequency_1 = reversed_pair_train_df.query(pair[1] + '==' + str(value_1)).value_counts()
                    frequent_total_1 = np.sum(frequency_1.values)
                    prob_10 = frequency_1[(value_1, value)] / frequent_total_1
                    num_cat_0 = len(frequency_1.keys())
                    if (prob_10 < threshold):
                        train_imbalanced  = train_df[train_df[pair[0]] == value]
                        train_imbalanced = train_imbalanced[train_imbalanced[pair[1]] == value_1]
                        # train_imbalanced = train_df.query(
                        #     pair[0] + '==' + str(value) + " and " + pair[1] + '==' + str(value_1))
                        frequency_imbalanced = len(train_imbalanced)
                        synthesized_imbalanced = synth_df[synth_df[pair[0]] == value]
                        synthesized_imbalanced = synthesized_imbalanced[synthesized_imbalanced[pair[1]] == value_1]
                        # synthesized_imbalanced = synth_df.query(
                        #     pair[0] + '==' + str(value) + " and " + pair[1] + '==' + str(value_1))
                        synthesized_frequency_imbalanced = len(synthesized_imbalanced)
                        # print("Frequency Compare", frequency_imbalanced, synthesized_frequency_imbalanced)
                        count += 1
                        if synthesized_frequency_imbalanced > 0:
                            count_synth += 1
                        train_total += frequency_imbalanced
                        synthesized_total += synthesized_frequency_imbalanced
    return count, count_synth ,train_total, synthesized_total

def mutual_minority_counts2(train_df, synth_df1, synth_df2, pair):
    reversed_pair = [pair[1], pair[0]]

    pair_train_df = train_df[pair]

    # pair_synthesized_df = synth_df[pair]
    reversed_pair_train_df = train_df[reversed_pair]
    # reversed_pair_synthesized_df = synth_df[reversed_pair]

    train_total = 0
    synthesized_total = 0

    column_0_unique = pair_train_df[pair[0]].unique()
    count = 0
    count_synth1 = 0
    count_synth2 = 0
    for value in column_0_unique:
        column_1_unique = pair_train_df[pair[1]].unique()
        frequency_0 = pair_train_df[pair_train_df[pair[0]] == value].value_counts()
        # frequency_0 = pair_train_df.query(pair[0] + '==' + str(value)).value_counts()
        frequent_total_0 = np.sum(frequency_0.values)

        for value_1 in column_1_unique:
            if ((value, value_1) in frequency_0.keys()):
                num_cat_1 = len(frequency_0.keys())
                prob = frequency_0[(value, value_1)] / frequent_total_0
                # threshold = 1/(10*num_cat_1)
                threshold = 1 / num_cat_1
                if (prob < threshold):
                    frequency_1 = reversed_pair_train_df[reversed_pair_train_df[pair[1]] == value_1].value_counts()
                    # frequency_1 = reversed_pair_train_df.query(pair[1] + '==' + str(value_1)).value_counts()
                    frequent_total_1 = np.sum(frequency_1.values)
                    prob_10 = frequency_1[(value_1, value)] / frequent_total_1
                    num_cat_0 = len(frequency_1.keys())
                    if (prob_10 < threshold):
                        train_imbalanced  = train_df[train_df[pair[0]] == value]
                        train_imbalanced = train_imbalanced[train_imbalanced[pair[1]] == value_1]

                        synthesized_imbalanced1 = synth_df1[synth_df1[pair[0]] == value]
                        synthesized_imbalanced1 = synthesized_imbalanced1[synthesized_imbalanced1[pair[1]] == value_1]
                        synthesized_frequency_imbalanced1 = len(synthesized_imbalanced1)

                        synthesized_imbalanced2 = synth_df2[synth_df2[pair[0]] == value]
                        synthesized_imbalanced2 = synthesized_imbalanced2[synthesized_imbalanced2[pair[1]] == value_1]
                        synthesized_frequency_imbalanced2 = len(synthesized_imbalanced2)

                        count += 1
                        if synthesized_frequency_imbalanced1 > 0:
                            count_synth1 += 1
                        if synthesized_frequency_imbalanced2 > 0:
                            count_synth2 += 1
                        # train_total += frequency_imbalanced
                        # synthesized_total += synthesized_frequency_imbalanced
    return count, count_synth1, count_synth2


import random

def mutual_minority(train_data, synth_data):
    train_df = pd.read_csv(train_data)
    synth_df = pd.read_csv(synth_data)

    columns = train_df.columns
    total = 0
    total_synth= 0
    train_totalf = 0
    synth_totalf = 0
    n = len(columns)

    for i in range(50):

        column1 = random.randint(0, n-1)
        column2 = random.randint(0, n-1)
        while column2 == column1:
            column2 = random.randint(0, n-1)
        count, count_synth ,train_tt, synth_tt = mutual_minority_counts(train_df, synth_df, [columns[column1], columns[column2]])
        total += count
        total_synth += count_synth
        train_totalf += train_tt
        synth_totalf += synth_tt
        print("iter %d"%i)
    return total, total_synth ,train_totalf, synth_totalf



def mutual_minority2(train_data, synth_data1, synth_data2):
    train_df = pd.read_csv(train_data)
    synth_df1 = pd.read_csv(synth_data1)
    synth_df2 = pd.read_csv(synth_data2)

    columns = train_df.columns
    total = 0
    total_synth1= 0
    total_synth2= 0
    n = len(columns)

    for i in range(50):

        column1 = random.randint(0, n-1)
        column2 = random.randint(0, n-1)
        while column2 == column1:
            column2 = random.randint(0, n-1)
        count, cs1, cs2 = mutual_minority_counts2(train_df, synth_df1, synth_df2, [columns[column1], columns[column2]])
        total += count
        total_synth1 += cs1
        total_synth2 += cs2

        print("iter %d"%i)
    return total, total_synth1 , total_synth2

