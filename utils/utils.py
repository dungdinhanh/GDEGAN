from sdgym.data import load_dataset
import os
import pandas as pd
import json
from score.scripts import generate_gt_2_adult_fd
from score.scripts import test_scoring_adult_fd
from score.scripts import generate_ground_truth_1, generate_ground_truth_2, test_scoring
from score.scripts_m import scoring_inferface, generate_gt_1_adult_fd
import numpy as np

DATAFOLDER="data"

def get_columns_names(n):
    columns = []
    for i in range(n):
        columns.append("column_"+str(i))
    return columns

def get_columns_name_from_meta(meta):
    columns = []
    columns_meta = meta['columns']
    for column_meta in columns_meta:
        columns.append(column_meta['name'])
    return columns
    pass

def get_new_meta(meta):
    columns_meta = meta['columns']
    new_meta = {}
    for column_meta in columns_meta:
        name = column_meta['name']
        new_meta[name] = {}
        if column_meta['type'] == 'continuous':
            new_meta[name]['type'] = 'float'
            new_meta[name]['max'] = column_meta['max']
            new_meta[name]['min'] = column_meta['min']
        else:
            new_meta[name]['type'] = 'enum'
            new_meta[name]['count'] = column_meta['size']
    return new_meta


def check_float(column_data: np.ndarray):
    for element in column_data:
        value = float(element)
        if not value.is_integer():
            return True
    return False


def get_new_meta2(meta, train):
    """
    allow distinguish between continuous and integer values
    """
    columns_meta = meta['columns']
    new_meta = {}
    count = 0
    for column_meta in columns_meta:
        name = column_meta['name']
        new_meta[name] = {}
        if column_meta['type'] == 'continuous':
            data = train[:, count]
            if check_float(data):
                new_meta[name]['type'] = 'float'
            else:
                new_meta[name]['type'] = 'integer'
            new_meta[name]['max'] = column_meta['max']
            new_meta[name]['min'] = column_meta['min']
        else:
            new_meta[name]['type'] = 'enum'
            new_meta[name]['count'] = column_meta['size']
    return new_meta


def get_real_file_path(dataset):
    output_dir = os.path.join(DATAFOLDER, dataset)
    os.makedirs(output_dir, exist_ok=True)
    # configure = Configure.get_configure()
    # train, categoricals, ordinals = load_dataset(dataset)
    train, test, meta, categoricals, ordinals = load_dataset(dataset, benchmark=True)
    train = np.concatenate((train, test), axis=0)
    new_meta = get_new_meta2(meta, train)
    real_file = os.path.join(output_dir, "%s.csv"%dataset)
    columns = get_columns_name_from_meta(meta)
    data = pd.DataFrame(train, columns=columns)
    data.to_csv(real_file, index=False)

    meta_file = os.path.join(output_dir, "%s.json"%dataset)
    f = open(meta_file, "w")
    json.dump(new_meta, f, indent=4)
    f.close()
    return real_file, meta_file


def check_data_exist(dataset):
    output_dir = os.path.join(DATAFOLDER, dataset)
    real_file = os.path.join(output_dir, "%s.csv" % dataset)
    meta_file = os.path.join(output_dir, "%s.json" % dataset)
    if os.path.isfile(real_file) and os.path.isfile(meta_file):
        return real_file, meta_file
    return get_real_file_path(dataset)

from PIL import Image
import torchvision.utils as vutils
import torch

def save_image(images: np.ndarray, file_folder):
    n = images.shape[0]
    im_folder = os.path.join(file_folder, "images")
    os.makedirs(im_folder, exist_ok=True)
    for i in range(n):
        file_name = os.path.join(im_folder, "%d.png"%i)
        im = Image.fromarray((images[i].reshape([28, 28]) * 255).astype(np.uint8))
        im = im.convert("L")
        im.save(file_name)



def immerge_row_col(N):
    c = int(np.floor(np.sqrt(N)))
    for v in range(c, N):
        if N % v == 0:
            c = v
            break
    r = N / c
    return r, c


def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)
    @images: is in shape of N * H * W(* C=1 or 3)
    """
    row = int(row)
    col = int(col)
    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image
    return img

def save_batch(images: np.ndarray, file_name):
    images = torch.Tensor(images.reshape(-1, 1, 28, 28))
    vutils.save_image(images, file_name, normalize=True, scale_each=True, nrow=10)
    pass





def get_ground_truth_af(dataset):
    output_dir = os.path.join(DATAFOLDER, dataset)
    ground1 = os.path.join(output_dir, "%s_gt1.gt"%dataset)
    ground2 = os.path.join(output_dir, "%s_gt2.gt"%dataset)
    real, meta = check_data_exist(dataset)
    if not os.path.isfile(ground1):
        os.makedirs(output_dir, exist_ok=True)
        generate_gt_1_adult_fd.generate_ground_truth1(real, meta, ground1)
    if not os.path.isfile(ground2):
        os.makedirs(output_dir, exist_ok=True)
        generate_gt_2_adult_fd.generate_ground_truth2(real, meta, ground2)
    return ground1, ground2
    pass

def get_ground_truth(dataset):
    output_dir = os.path.join(DATAFOLDER, dataset)
    ground1 = os.path.join(output_dir, "%s_gt1.gt" % dataset)
    ground2 = os.path.join(output_dir, "%s_gt2.gt" % dataset)
    real, meta = check_data_exist(dataset)
    if not os.path.isfile(ground1):
        os.makedirs(output_dir, exist_ok=True)
        generate_ground_truth_1.generate_ground_truth1(real, meta, ground1)
    if not os.path.isfile(ground2):
        os.makedirs(output_dir, exist_ok=True)
        generate_ground_truth_2.generate_ground_truth2(real, meta, ground2)
    return ground1, ground2

def get_ground_truth_af_set(dataset, set):
    output_dir = os.path.join(DATAFOLDER, dataset)
    ground1 = os.path.join(output_dir, "%s_gt1_set%d.gt"%(dataset, set))
    real, meta = check_data_exist(dataset)
    total_minor = 0
    if set == 2:
        if not os.path.isfile(ground1):
            os.makedirs(output_dir, exist_ok=True)
            total_minor = generate_gt_1_adult_fd.generate_ground_truth1_set2(real, meta, ground1)
    else:
        return ground1, None
    return ground1, total_minor
import random

def generate_random_columns(real, gt_file):
    dtf = pd.read_csv(real)
    columns = dtf.columns
    n = len(columns)
    f = open(gt_file, "w")
    f.write("%s, %s\n"%("col1", "col2"))
    for i in range(100):
        selected = random.sample(range(n), 2)
        f.write("%d, %d\n"%(selected[0], selected[1]))
    f.close()

def get_ground_truth_mi_set(dataset):
    output_dir = os.path.join(DATAFOLDER, dataset)
    ground_mi = os.path.join(output_dir, "%s_gt_mi.csv"%(dataset))
    real, meta = check_data_exist(dataset)
    if not os.path.isfile(ground_mi):
        os.makedirs(output_dir, exist_ok=True)
        generate_random_columns(real, ground_mi)
    return ground_mi

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif as mi_reg
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def calculate_mi_2df(df1: pd.DataFrame, df2: pd.DataFrame):
    return calc_MI(df1.values, df2.values, 100)


def calculate_mi_score(dataset, synthetic_data_file):
    real, meta = check_data_exist(dataset)
    ground_mi = get_ground_truth_mi_set(dataset)
    ground = pd.read_csv(ground_mi)
    matrix = ground.values

    real_data = pd.read_csv(real)
    synth_data = pd.read_csv(synthetic_data_file)
    columns = real_data.columns
    n = matrix.shape[0]
    sum_mi1 = 0.0
    sum_mi2 = 0.0
    sum_joint = 0.0
    for i in range(n):
        col1 = columns[int(matrix[i][0])]
        col2 = columns[int(matrix[i][1])]
        mi1 = calculate_mi_2df(real_data[col1], synth_data[col1])
        mi2 = calculate_mi_2df(real_data[col2], synth_data[col2])
        newdf1 = pd.DataFrame(real_data[[col1, col2]].apply(tuple, axis=1))
        newdf2 = pd.DataFrame(synth_data[[col1, col2]].apply(tuple, axis=1))
        # newdf1 = pd.DataFrame(list(zip(real_data[[col1]].values, real_data[col2].values)))
        # newdf2 = pd.DataFrame(list(zip(synth_data[[col1]].values, synth_data[col2].values)))
        # newdf2 = pd.DataFrame(list(zip(synth_data[[col1, col2]])))
        mi_joint = calculate_mi_2df(newdf1, newdf2)
        # mi_joint = calculate_mi_2df(real_data[[col1, col2]].groupby([col1, col2]), synth_data[[col1, col2]].groupby([col1, col2]))
        sum_mi1 += mi1
        sum_mi2 += mi2
        sum_joint += mi_joint
    avg_mi1 = sum_mi1/n
    avg_mi2 = sum_mi2/n
    avg_joint = sum_joint/n
    return avg_mi1, avg_mi2, avg_joint


def calculate_comp_score(dataset, synthetic_data_file):
    gt1, gt2 = get_ground_truth_af(dataset)
    real, meta = check_data_exist(dataset)
    return test_scoring_adult_fd.calculate_scores(synthetic_data_file, gt1, gt2, meta)

def calculate_comp_score_colorado(dataset, synthetic_data_file):
    gt1, gt2 = get_ground_truth(dataset)
    real, meta = check_data_exist(dataset)
    return test_scoring.calculate_scores(synthetic_data_file, gt1, gt2, meta)

def normalize_values(synth: pd.DataFrame, meta: dict):
    for column in meta.keys():
        if meta[column]['type'] == 'integer' or meta[column]['type'] == 'enum':
            synth[column] = synth[column].round(0).astype(int)
            pass
        if meta[column]['type'] == 'float' or meta[column]['type'] == 'integer':
            col_max = meta[column]['max']
            col_min = meta[column]['min']
            dropping_1 = synth[synth[column] > col_max].index
            dropping_2 = synth[synth[column] < col_min].index
            synth.loc[dropping_1, column] = col_max
            synth.loc[dropping_2, column] = col_min
    return synth

def normalize_values_with_nan(synth: pd.DataFrame, meta: dict):
    for column in meta.keys():
        if meta[column]['type'] == 'integer' or meta[column]['type'] == 'enum':
            synth[column] = synth[column].round(0)
            pass
        if meta[column]['type'] == 'float' or meta[column]['type'] == 'integer':
            col_max = meta[column]['max']
            col_min = meta[column]['min']
            dropping_1 = synth[synth[column] > col_max].index
            dropping_2 = synth[synth[column] < col_min].index
            synth.loc[dropping_1, column] = col_max
            synth.loc[dropping_2, column] = col_min
            # this works for nan
    return synth




import csv
from statistics import mean, stdev

def calculate_outlier(statistic_dict:dict):
    for item in statistic_dict.keys():
        list_elements = statistic_dict[item]['elements']
        list_collapse = statistic_dict[item]['collapse']
        statistic_dict[item]['ratio_list'] = []
        mean_value = mean(list_elements)
        std_value = stdev(list_elements)
        if len(list_collapse) == 0:
            statistic_dict[item]['average_ratio'] = -1
            continue
        for value in list_collapse:
            distance = float(abs(value - mean_value))
            ratio = distance/std_value
            statistic_dict[item]['ratio_list'].append(ratio)
        # if len(statistic_dict[item]['ratio_list']) == 0:
        #     a=5
        statistic_dict[item]['average_ratio'] = mean(statistic_dict[item]['ratio_list'])
        print(statistic_dict[item]['average_ratio'])
    return statistic_dict

def calculate_score1_set(dataset, set, synthetic_data_file):
    gt1, total_minor = get_ground_truth_af_set(dataset, set)
    real, meta = check_data_exist(dataset)
    if set == 2:
        score1_set2_exist = scoring_inferface.calcScore1_set2_exist(synthetic_data_file, gt1, meta)
        return score1_set2_exist
    return None

def get_columns(dataframe: pd.DataFrame, meta: dict):
    columns = dataframe.columns
    categorical = []
    ordinal = []
    for idx, column in enumerate(columns):
        if meta[column]['type'] == 'enum':
            categorical.append(idx)
    return categorical, ordinal, columns




