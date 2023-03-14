import os, math

import pickle
import numpy as np
import pandas as pd
import geopy.distance
from scipy.sparse import linalg
import scipy.sparse as sp
import tqdm

class DataLoader(object):
    def __init__(self, xs, dateTime, ys, mn, batch_size):
        """
        :param xs: (N, 8, D, L)
        :param dateTime: (N, L)
        :param ys: (N, D, L)
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.dateTime = dateTime
        self.ys = ys
        self.mn = mn

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, dateTime, ys, mn = self.xs[permutation], self.dateTime[permutation], self.ys[permutation], self.mn[
            permutation]
        self.xs = xs
        self.dateTime = dateTime
        self.ys = ys
        self.mn = mn

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                dateTime_i = self.dateTime[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                mn_i = self.mn[start_ind: end_ind, ...]
                yield (x_i, dateTime_i, y_i, mn_i)
                self.current_ind += 1

        return _wrapper()

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename, adjtype):
    """
    Load directed adjacency matrix (predefined)
    :param pkl_filename:
    :param adjtype:
    :return:
    """
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj, adj_mx


def load_dataset(stat_file, batch_size):
    """

    :param traffic_df_filename:
    :param stat_file:
    :param batch_size:
    :return:
        - df: (N_all, D), the full dataframe including "dateTime" ass the first column
        - data: a dict including several componentss
    """
    data = {}
    # df = pd.read_hdf(traffic_df_filename)
    for cat in ['train', 'val', 'test']:
        file_save_path = stat_file[:-4] + '_' + cat + '.npz'  # e.g., 'x.npz' -> 'x_train.npz'
        cat_data = np.load(file_save_path)
        # x, dateTime, y: (N, 8, L, D), (N, L), (N, L, D)
        data['x_' + cat] = cat_data['x']
        data['dateTime_' + cat] = cat_data['dateTime']
        data['y_' + cat] = cat_data['y']
        data['missing_nodes_' + cat] = cat_data['missing_nodes']
        data['max_speed'] = cat_data['max_speed']

    # The data is scaled by "max_speed"
    data['train_loader'] = DataLoader(data['x_train'], data['dateTime_train'], data['y_train'], data['missing_nodes_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['dateTime_val'], data['y_val'], data['missing_nodes_val'], batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['dateTime_test'], data['y_test'], data['missing_nodes_test'], batch_size)
    return data

## newly added method from generating the adjacency matrix: directed graph
def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    Compute the directed adjacency matrix

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for index, row in tqdm(distance_df.iterrows(), total=distance_df.shape[0]):

        if row["from"] not in sensor_ids or row["to"] not in sensor_ids:
            continue
        dist_mx[sensor_id_to_ind[row["from"]], sensor_id_to_ind[row["to"]]] = row["cost"]
    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    return sensor_ids, sensor_id_to_ind, adj_mx
