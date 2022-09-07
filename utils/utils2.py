import os
import pandas as pd
import json
from score.scripts import generate_gt_2_adult_fd
from score.scripts import test_scoring_adult_fd
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





def check_data_exist(dataset):
    output_dir = os.path.join(DATAFOLDER, dataset)
    real_file = os.path.join(output_dir, "%s.csv" % dataset)
    meta_file = os.path.join(output_dir, "%s.json" % dataset)
    # if os.path.isfile(real_file) and os.path.isfile(meta_file):
    return real_file, meta_file


from PIL import Image


def save_image(images: np.ndarray, file_folder):
    n = images.shape[0]
    im_folder = os.path.join(file_folder, "images")
    os.makedirs(im_folder, exist_ok=True)
    for i in range(n):
        file_name = os.path.join(im_folder, "%d.png"%i)

        im = Image.fromarray((images[i].reshape([28, 28]) * 255).astype(np.uint8))
        im = im.convert("L")
        im.save(file_name)

# from sklearn.metrics import mutual_info_score
# import networkx as nx
#
# def calc_MI(x, y, bins):
#     c_xy = np.histogram2d(x, y, bins)[0]
#     mi = mutual_info_score(None, None, contingency=c_xy)
#     return mi
#
#
# def make_graph(name, data, degree, file_name):
#     if (os.path.exists(file_name)):
#         print("Graph of " + name + " is already!")
#         G = nx.read_gpickle(file_name)
#     else:
#         G = nx.Graph()
#         n = data.shape[1]
#         G.add_nodes_from(range(n))
#         added_vertices = []
#         nodes = list(G.nodes)
#         added_vertices.append(0)
#         while (1):
#             print("number of nodes", len(added_vertices))
#             if (len(added_vertices) == n):
#                 break
#             mutual_infos = dict()
#             adding_edges = []
#             for node in nodes:
#                 for source in added_vertices:
#                     mutual_infos[source] = dict()
#                     if (source != node and (not node in added_vertices)):
#                         mutual_infos[source][node] = calc_MI(data[:, source], data[:, node], bins=1000)
#                         if (len(adding_edges) < degree):
#                             adding_edges.append([source, node, mutual_infos[source][node]])
#                         else:
#                             for i in range(degree):
#                                 if (mutual_infos[source][node] > adding_edges[i][2]):
#                                     adding_edges[i] = [source, node, mutual_infos[source][node]]
#                                     break
#             print(adding_edges)
#             for adding_edge in adding_edges:
#                 G.add_edge(adding_edge[0], adding_edge[1])
#                 if (not adding_edge[0] in added_vertices):
#                     added_vertices.append(adding_edge[0])
#                 if (not adding_edge[1] in added_vertices):
#                     added_vertices.append(adding_edge[1])
#
#         print(G.edges, len(list(G.edges)))
#         # H, _ = nx.complete_to_chordal_graph(G)
#         # fh = open('graph/'+name+'.adjlist', "wb")
#         nx.write_gpickle(G, file_name)
#     return G
#
# def get_graph(dataset, degree=3):
#     output_dir  = os.path.join(DATAFOLDER, dataset)
#     graph_file = os.path.join(output_dir, "%s_d%d_graph.pickle"%(dataset, degree))
#     real_file, _ = get_real_file_path(dataset)
#     datadf = pd.read_csv(real_file)
#     data = datadf.values.astype('float32')
#     graph_G = make_graph(dataset, data, degree, graph_file)
#     return graph_G




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



def get_ground_truth_af_set(dataset, set):
    output_dir = os.path.join(DATAFOLDER, dataset)
    ground1 = os.path.join(output_dir, "%s_gt1_set%d.gt"%(dataset, set))
    real, meta = check_data_exist(dataset)
    if set == 1:
        if not os.path.isfile(ground1):
            os.makedirs(output_dir, exist_ok=True)
            generate_gt_1_adult_fd.generate_ground_truth1_set1(real, meta, ground1, 15)
    elif set == 2:
        if not os.path.isfile(ground1):
            os.makedirs(output_dir, exist_ok=True)
            generate_gt_1_adult_fd.generate_ground_truth1_set2(real, meta, ground1)
    elif set == 3:
        if not os.path.isfile(ground1):
            os.makedirs(output_dir, exist_ok=True)
            generate_gt_1_adult_fd.generate_ground_truth1_set3(real, meta, ground1)
    elif set == 4:
        if not os.path.isfile(ground1):
            os.makedirs(output_dir, exist_ok=True)
            generate_gt_1_adult_fd.generate_ground_truth1_set4(real, meta, ground1)
    return ground1


def calculate_comp_score(dataset, synthetic_data_file):
    gt1, gt2 = get_ground_truth_af(dataset)
    real, meta = check_data_exist(dataset)
    return test_scoring_adult_fd.calculate_scores(synthetic_data_file, gt1, gt2, meta)

def normalize_values(synth: pd.DataFrame, meta: dict):
    for column in meta.keys():
        if meta[column]['type'] == 'integer':
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
    gt1 = get_ground_truth_af_set(dataset, set)
    real, meta = check_data_exist(dataset)

    if set == 1:
        inData = csv.reader(open(synthetic_data_file, newline=''), dialect='excel')
        inGT1 = csv.reader(open(gt1, newline=''), dialect='excel')
        inDataSpecs = json.load(open(meta))
        score1_set1 = scoring_inferface.calcScore1_set1(inData, inGT1, inDataSpecs)
        inData = csv.reader(open(synthetic_data_file, newline=''), dialect='excel')
        inGT1 = csv.reader(open(gt1, newline=''), dialect='excel')
        score1_set1_exist = scoring_inferface.calcScore1_set1_exist(inData, inGT1, inDataSpecs)
        return score1_set1, score1_set1_exist
    if set == 2:
        inData = csv.reader(open(synthetic_data_file, newline=''), dialect='excel')
        inGT1 = csv.reader(open(gt1, newline=''), dialect='excel')
        inDataSpecs = json.load(open(meta))
        score1_set2 = scoring_inferface.calcScore1_set2(inData, inGT1, inDataSpecs)
        inData = csv.reader(open(synthetic_data_file, newline=''), dialect='excel')
        inGT1 = csv.reader(open(gt1, newline=''), dialect='excel')
        score1_set2_exist = scoring_inferface.calcScore1_set2_exist(inData, inGT1, inDataSpecs)
        return score1_set2, score1_set2_exist
    if set ==3:
        inData = csv.reader(open(synthetic_data_file, newline=''), dialect='excel')
        inGT1 = csv.reader(open(gt1, newline=''), dialect='excel')
        inDataSpecs = json.load(open(meta))
        # score1_set2 = scoring_inferface.calcScore1_set3(inData, inGT1, inDataSpecs)
        score1_set2 = None
        inData = csv.reader(open(synthetic_data_file, newline=''), dialect='excel')
        inGT1 = csv.reader(open(gt1, newline=''), dialect='excel')
        score1_set2_exist, stat_dict = scoring_inferface.calcScore1_set3_exist(inData, inGT1, inDataSpecs)
        stat_dict = calculate_outlier(stat_dict)
        return score1_set2, score1_set2_exist, stat_dict
    if set == 4:
        inData = csv.reader(open(synthetic_data_file, newline=''), dialect='excel')
        inGT1 = csv.reader(open(gt1, newline=''), dialect='excel')
        inDataSpecs = json.load(open(meta))
        score1_set2 = scoring_inferface.calcScore1_set4(inData, inGT1, inDataSpecs)
        inData = csv.reader(open(synthetic_data_file, newline=''), dialect='excel')
        inGT1 = csv.reader(open(gt1, newline=''), dialect='excel')
        score1_set2_exist = scoring_inferface.calcScore1_set4(inData, inGT1, inDataSpecs)
        return score1_set2, score1_set2_exist

    return None

def get_columns(dataframe: pd.DataFrame, meta: dict):
    columns = dataframe.columns
    categorical = []
    ordinal = []
    for idx, column in enumerate(columns):
        if meta[column]['type'] == 'enum':
            categorical.append(idx)
    return categorical, ordinal, columns




