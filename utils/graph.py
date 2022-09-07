from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
import networkx as nx
import os

from utils.utils import get_real_file_path, DATAFOLDER, check_data_exist
import pandas as pd
import numpy as np


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def make_graph(name, data, degree, file_name):
    if (os.path.exists(file_name)):
        print("Graph of " + name + " is already!")
        G = nx.read_gpickle(file_name)
    else:
        G = nx.Graph()
        n = data.shape[1]
        G.add_nodes_from(range(n))
        added_vertices = []
        nodes = list(G.nodes)
        added_vertices.append(0)
        while (1):
            print("number of nodes", len(added_vertices))
            if (len(added_vertices) == n):
                break
            mutual_infos = dict()
            adding_edges = []
            for node in nodes:
                for source in added_vertices:
                    mutual_infos[source] = dict()
                    if (source != node and (not node in added_vertices)):
                        mutual_infos[source][node] = calc_MI(data[:, source], data[:, node], bins=1000)
                        if (len(adding_edges) < degree):
                            adding_edges.append([source, node, mutual_infos[source][node]])
                        else:
                            for i in range(degree):
                                if (mutual_infos[source][node] > adding_edges[i][2]):
                                    adding_edges[i] = [source, node, mutual_infos[source][node]]
                                    break
            print(adding_edges)
            for adding_edge in adding_edges:
                G.add_edge(adding_edge[0], adding_edge[1])
                if (not adding_edge[0] in added_vertices):
                    added_vertices.append(adding_edge[0])
                if (not adding_edge[1] in added_vertices):
                    added_vertices.append(adding_edge[1])

        print(G.edges, len(list(G.edges)))
        # H, _ = nx.complete_to_chordal_graph(G)
        # fh = open('graph/'+name+'.adjlist', "wb")
        nx.write_gpickle(G, file_name)
    return G

def calculate_mi(data):
    # input m * n matrix with m rows and n columns
    # output A{n * n} matrix with A[i,j] = MI
    nodes = data.shape[1]
    mi_matrix = np.zeros([nodes, nodes], dtype=np.float32)
    for i in range(nodes):
        for j in range(i, nodes, 1):
            if i == j:
                continue
            mi_matrix[i, j] = normalized_mutual_info_score(data[:, i], data[:, j], average_method='max')
            mi_matrix[j, i] = mi_matrix[i, j]

    chosen_node = np.argmax(np.sum(mi_matrix, axis=1))
    return mi_matrix, chosen_node



def make_graph_hi(name, data, file_name):
    if (os.path.exists(file_name)):
        print("Graph of " + name + " is already!")
        G = nx.read_gpickle(file_name)
    else:
        # G = nx.Graph()
        n = data.shape[1]
        # G.add_nodes_from(range(n))
        # added_vertices = []
        # nodes = list(G.nodes)
        mi_matrix, chosen_node = calculate_mi(data)
        edges_list = np.dstack(np.unravel_index(np.argsort(-mi_matrix.ravel()), (n, n)))[0]
        flags = np.zeros(n, dtype=int)
        new_graph = np.zeros([n, n], dtype=int)
        n_edges = edges_list.shape[0]
        count = n
        for i in range(n_edges):
            edge = edges_list[i]
            source = edge[0]
            sink = edge[1]
            if flags[source] == 0:
                count -= 1
                flags[source] = 1
            if flags[sink] == 0:
                count -= 1
                flags[sink] = 1
            new_graph[source, sink] = 1
            new_graph[sink, source] = 1
            if count <= 0:
                break
        G = nx.from_numpy_array(new_graph)
        # H, _ = nx.complete_to_chordal_graph(G)
        # fh = open('graph/'+name+'.adjlist', "wb")
        nx.write_gpickle(G, file_name)
    return G

import json

def get_graph(dataset, degree=3, mode=0):
    output_dir = os.path.join(DATAFOLDER, dataset)

    real_file, meta = check_data_exist(dataset)
    datadf = pd.read_csv(real_file)
    # f = open(meta, "r")
    # meta_json = json.load(f)
    # f.close()
    # columns = datadf.columns
    # for column in columns:
    #     if meta_json[column]['type'] != 'enum':
    #         n = (float(meta_json[column]['max']) - float(meta_json[column]['min']))/100
    #         if n == 0:
    #             n = 1
    #         datadf[column] = pd.cut(datadf[column], n)
    data = datadf.values.astype('float32')
    if mode == 0:
        graph_file = os.path.join(output_dir, "%s_d%d_graph.pickle" % (dataset, degree))
        graph_G = make_graph(dataset, data, degree, graph_file)
    elif mode == 1:
        graph_file = os.path.join(output_dir, "%s_hi_graph.pickle" % (dataset))
        graph_G = make_graph_hi(dataset, data, graph_file)
    else:
        graph_G = None
        print("No graph the algorithm will be chosen based on random columns")
    return graph_G

def process_nan(datadf: pd.DataFrame):
    columns = datadf.columns
    new_datadf = pd.DataFrame()
    for column in columns:
        current = datadf[column]
        max_value = current.max()
        min_value = current.min()
        max_nan = current.max(skipna=False)
        if np.isnan(max_nan):
            nan_value = min_value - abs(max_value - min_value)
            current = current.fillna(nan_value)
        new_datadf[column] = current
    return new_datadf


def get_graph_nan(dataset, degree=3, mode=0):
    output_dir = os.path.join(DATAFOLDER, dataset)

    real_file, meta = check_data_exist(dataset)
    datadf = pd.read_csv(real_file)
    datadf = process_nan(datadf)
    # f = open(meta, "r")
    # meta_json = json.load(f)
    # f.close()
    # columns = datadf.columns
    # for column in columns:
    #     if meta_json[column]['type'] != 'enum':
    #         n = (float(meta_json[column]['max']) - float(meta_json[column]['min']))/100
    #         if n == 0:
    #             n = 1
    #         datadf[column] = pd.cut(datadf[column], n)
    data = datadf.values.astype('float32')
    if mode == 0:
        graph_file = os.path.join(output_dir, "%s_d%d_graph.pickle" % (dataset, degree))
        graph_G = make_graph(dataset, data, degree, graph_file)
    elif mode == 1:
        graph_file = os.path.join(output_dir, "%s_hi_graph.pickle" % (dataset))
        graph_G = make_graph_hi(dataset, data, graph_file)
    else:
        graph_G = None
        print("No graph the algorithm will be chosen based on random columns")
    return graph_G

