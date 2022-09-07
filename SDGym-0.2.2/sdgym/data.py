import json
import logging
import os
import urllib

import numpy as np
import sdgym.datasets as load31

from sdgym.constants import CATEGORICAL, ORDINAL

LOGGER = logging.getLogger(__name__)

BASE_URL = 'http://sdgym.s3.amazonaws.com/datasets/'
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _dump_json(meta, path):
    with open(path, "w") as json_file:
        return json.dump(meta, json_file, indent=4)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(local_path):
        os.makedirs(DATA_PATH, exist_ok=True)
        url = BASE_URL + filename

        LOGGER.info('Downloading file %s to %s', url, local_path)
        urllib.request.urlretrieve(url, local_path)

    return loader(local_path)


def _save_file(meta, filename, saver):
    local_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(local_path):
        os.makedirs(DATA_PATH, exist_ok=True)
    return saver(meta, local_path)

def _load_filedata(name, loader):
    local_path = os.path.join(DATA_PATH, name + ".npz")
    if not os.path.exists(local_path):
        os.makedirs(DATA_PATH, exist_ok=True)
        load_dataset31(name)

    return loader(local_path, allow_pickle=True)


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def load_dataset(name, benchmark=False):
    LOGGER.info('Loading dataset %s', name)
    data = _load_filedata(name, np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data['train']
    if benchmark:
        return train, data['test'], meta, categorical_columns, ordinal_columns

    return train, categorical_columns, ordinal_columns


def _get_columns31(real_data, table_metadata):
    model_columns = []
    categorical_columns = []

    fields_meta = table_metadata['fields']

    for column in real_data.columns:
        field_meta = fields_meta[column]
        field_type = field_meta['type']
        if field_type == 'id':
            continue

        index = len(model_columns)
        if field_type == 'categorical' or field_type == 'boolean':
            categorical_columns.append(index)

        model_columns.append(column)

    return model_columns, categorical_columns

import rdt

def get_i2s(encoded: {}, cat_names: []):
    i2s_dict = {}
    for column in encoded.keys():
        if column not in cat_names:
            continue
        tf = encoded[column]
        if type(tf) is rdt.transformers.BooleanTransformer:
            i2s = ['False', 'True']
            i2s_dict[column] = i2s
            continue
        encode_dict = tf.values_to_categories
        n = len(encode_dict)
        i2s = []
        for i in range(n):
            i2s.append(encode_dict[i])
        i2s_dict[column] = i2s
    return i2s_dict


def update_i2s(encoded_dict: {}, meta):
    n = len(meta['columns'])
    for i in range(n):
        name_column = meta['columns'][i]['name']
        if name_column in encoded_dict:
            # For boolean case default will be False and True. But for alarm case it should be FALSE and TRUE
            if encoded_dict[name_column][0] == 'False' and encoded_dict[name_column][1] == 'True':
                if meta['columns'][i]['i2s'][0].isupper():
                    encoded_dict[name_column][0] = 'FALSE'
                if meta['columns'][i]['i2s'][1].isupper():
                    encoded_dict[name_column][1] = 'TRUE'
            meta['columns'][i]['i2s'] = encoded_dict[name_column]
    return meta

def categorical_names(columns, categoricals):
    cat_names = []
    for cat in categoricals:
        cat_names.append(columns[cat])
    return cat_names


def preprocess_data(real_data, table_metadata):
    columns, categoricals = _get_columns31(real_data, table_metadata)
    real_data = real_data[columns]
    cat_names = categorical_names(columns, categoricals)

    ht = rdt.HyperTransformer(dtype_transformers={
        'O': 'label_encoding',
    })
    ht.fit(real_data.iloc[:, categoricals])
    # model_data = ht.transform(real_data)
    model_data = ht.transform(real_data)[columns]

    encoded_info = ht._transformers
    get_i2s_info = get_i2s(encoded_info, cat_names)
    supported = set(model_data.select_dtypes(('number', 'bool')).columns)
    unsupported = set(model_data.columns) - supported
    if unsupported:
        unsupported_dtypes = model_data[unsupported].dtypes.unique().tolist()
        print("Unspported preprocess")
        exit(1)

    nulls = model_data.isnull().any()
    if nulls.any():
        unsupported_columns = nulls[nulls].index.tolist()
        print("Unsuppored null")
        exit(2)
        # raise UnsupportedDataset(f'Null values found in columns {unsupported_columns}')
    return model_data, get_i2s_info

import copy
def load_dataset31(name):
    # LOGGER.info('Loading dataset %s', name)
    meta31 = load31.load_dataset(name)
    real_data = load31.load_tables(meta31)[name]
    real_data, i2s_dict = preprocess_data(real_data, meta31.get_table_meta(meta31.get_tables()[0]))
    _meta = _load_file(name + '.json', _load_json)
    _meta = update_i2s(i2s_dict, _meta)
    _save_file(_meta, name + '.json', _dump_json)

    data = real_data.values

    np.random.shuffle(data)

    n = data.shape[0]
    if name in ['alarm', 'asia', 'child', 'insurance', 'grid', 'gridr', 'ring']:
        n1 = int(n/2)
    elif name in ['adult']:
        n1 = 22561
    elif name in ['census']:
        n1 = 199523
    elif name in ['covtype']:
        n1 = 481012
    elif name in ['intrusion']:
        n1 = 394021
    elif name in ['mnist12']:
        n1 = 60000
    elif name in ['mnist28']:
        n1 = 60000
    elif name in ['credit']:
        n1 = 264000
    elif name in ['news']:
        n1 = 31000
    else:
        n1 = int (n * 2/3)
        # print("not available dataset")
        # exit(1)

    # # debug
    # xyz_data = _load_file(name + ".npz", np.load)
    # xyz_train = xyz_data['train']
    # xyz_meta = _load_file(name + '.json', _load_json)
    # #---------------

    train = data[0:n1]
    test = data[n1:]
    data_path = os.path.join(DATA_PATH, name+'.npz')
    np.savez(data_path, train=train, test=test)

    # meta = _load_file(name + '.json', _load_json)
    #
    # categorical_columns, ordinal_columns = _get_columns(meta)
    #
    # train = data['train']

    # return train, test, meta, categorical_columns, ordinal_columns



