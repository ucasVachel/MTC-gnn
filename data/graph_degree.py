import pickle
import os, math
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopy.distance


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


def calculate_degree(adj_mx):
    """计算邻接矩阵的度"""
    adj_mx = np.where(adj_mx > 0, 1, 0)
    degree = np.sum(adj_mx, axis=1) - 1  # 计算每个节点的度
    return degree


def get_dist_matrix(sensor_locs):
    """
    Compute the absolute spatial distance matrix

    :param sensor_locs: with header and index, [index, sensor_id, longitude, latitude]
    :return:
    """
    sensor_ids = sensor_locs[1:, 1]  # remove header and index (METR-LA)
    # sensor_ids = sensor_locs[:, 0]  # remove header and index (METR-LA)# no need to remove in PEMS
    sensor_id_to_ind = {}
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind.update({sensor_id: i})
    for id1 in sensor_ids:
        # coords_1 = sensor_locs[sensor_locs[:, 0] == id1][0][1:]  # PEMS
        coords_1 = sensor_locs[sensor_locs[:, 1] == id1][0][2:]  # METR-LA
        for id2 in sensor_ids:
            if math.isinf(dist_mx[sensor_id_to_ind[id1], sensor_id_to_ind[id2]]):
                # coords_2 = sensor_locs[sensor_locs[:, 0] == id2][0][1:]  # PEMS
                coords_2 = sensor_locs[sensor_locs[:, 1] == id2][0][2:]  # METR-LA
                # 计算两经纬度点间的距离
                dist = round(geopy.distance.distance(coords_1, coords_2).km, 2)
                dist_mx[sensor_id_to_ind[id1], sensor_id_to_ind[id2]] = dist
                dist_mx[sensor_id_to_ind[id2], sensor_id_to_ind[id1]] = dist
            else:
                continue
    return sensor_ids, sensor_id_to_ind, dist_mx


def generate_stat_features_files2(traffic_df_filename, dist_filename, output_dir, masking, L, S, missing_nodes, name):
    """
            To generate the statistic features from raw datasets and save them into "npz" files
        :param traffic_df_filename:
        :param dist_file: distance matrix file
        :param output_dir: the path to save generated datasets
        :param masking: default True
        :param L: the recent sample numbers
        :param S: missing_nodes
        """
    # read h5 file (pems-bay.h5)
    df = pd.read_hdf(traffic_df_filename)
    # df = df.iloc[:1000,:]
    # load data with missing value
    sensor_locs = np.genfromtxt(dist_filename, delimiter=',')
    sensor_ids, sensor_id_to_ind, dist_mx = get_dist_matrix(sensor_locs)
    x_offsets = np.sort(np.arange(1, 13, 1))
    # Predict the next one hour
    # x: (N, 8, L, D)
    # dateTime: (N, L)
    # y: (N, L, D)
    prepare_dataset(
        output_dir,
        df,
        x_offsets,
        masking,
        dist_mx,
        L,
        S,
        missing_nodes,
        name)


def prepare_dataset(output_dir, df, x_offsets, masking, dists, L, S, missing_nodes, name):
    num_samples, num_nodes = df.shape
    data = df.values  # (num_samples, num_nodes)
    # 将值范围外的值进行归缩
    speed_tensor = data.clip(0, 100)  # (N, D)
    max_speed = speed_tensor.max().max()
    # 为了让数字不失去相对意义,我们需要进行量纲化 (最大值化)
    speed_tensor = speed_tensor / max_speed  # (N, D)

    # 同时找到日期索引
    date_array = df.index.values  # (N)

    # 所有节点
    all_nodes = np.arange(num_nodes)

    x, dateTime, y, MN, sn = [], [], [], [], []
    for t in range(-1, len(speed_tensor) - L):
        # 选取缺失节点 每个节点只能被选择一次
        selected_numbers = missing_nodes
        sorted_selected = np.sort(selected_numbers)
        # m = np.ones((12, 207)) META-LA
        m = np.ones((12, num_nodes))

        for i in sorted_selected:
            m[:, i] = 0

        x_t = speed_tensor[t + x_offsets, ...]
        x_t[:, sorted_selected] = 0
        dateTime_t = date_array[t + x_offsets]

        y_t = speed_tensor[t + x_offsets, :][:, sorted_selected]
        x.append(x_t)
        dateTime.append(dateTime_t)
        y.append(y_t)
        MN.append(m)
        sn.append(sorted_selected)
    # 按顺序堆叠数组
    speed_sequences = np.stack(x, axis=0)  # (N, L, D)
    dateTime = np.stack(dateTime, axis=0)  # (N, L)
    speed_labels = np.stack(y, axis=0)  # (N, L, M)
    missing_nodes = np.stack(sn, axis=0)  # (M)
    Mask = np.stack(MN, axis=0)  # (N, M)

    win_size = speed_sequences.shape[1]

    # using zero-one mask to randomly set elements to zeros
    if masking:
        print('Split Speed/label finished. Start to generate Mask, Delta_t, Last_observed_X ...')

        # temporal information -> to consider extracting the statistic feature from longer history data (caan probablement improve the performance)
        interval = 5  # 5 minutes
        s = np.zeros_like(speed_sequences)  # time stamps in (N, L, D)
        for i in range(s.shape[1]):
            s[:, i, :] = interval * i

        # time intervals ; spatial distance ;

        Delta_t = np.zeros_like(
            speed_sequences)  # time intervals, if all previous measures are missing, Delta_t[i, j, k] = 0, X_last_obsv[i, j ,k] = 0
        Delta_s = np.zeros_like(
            speed_sequences)  # spatial distance, if all variables are missing, Delta_s[i, j, k] = 0, X_closest_obsv[i, j ,k] = 0
        X_last_obsv = np.copy(speed_sequences)
        X_closest_obsv = np.copy(speed_sequences)
        X_mean_t = np.zeros_like(speed_sequences)
        X_mean_s = np.zeros_like(speed_sequences)

        for i in range(1, s.shape[1]):
            Delta_t[:, i, :] = s[:, i, :] - s[:, i - 1, :]  # calculate the exact minuites

        # 返回三维数组索引
        missing_index = np.where(Mask == 0)  # (array1, array2, array3), length of each array: number of missing values

        # X_mean_t, temporal mean for each segment
        start = time.time()
        nbr_all = speed_sequences.shape[0] * speed_sequences.shape[2]
        nbr_finished = 0
        current_ratio = 0
        for i in range(speed_sequences.shape[0]):  # N samples
            for d in range(speed_sequences.shape[2]):
                nbr_finished += 1
                finished_ratio = nbr_finished // (0.01 * nbr_all)
                if finished_ratio != current_ratio:
                    print("{}% of X_mean_t are calculated ! Accumulated time cost: {}s" \
                          .format(nbr_finished // (0.01 * nbr_all), time.time() - start))
                    current_ratio = finished_ratio
                temp_neighbor = speed_sequences[i, :, d]  # (L)
                # 找到当前样本中 某个传感器的非0值
                nonzero_index = np.nonzero(temp_neighbor)  # return x arrays, for each we have xx elements
                if len(nonzero_index[0]) == 0:
                    continue
                else:
                    nonzero_temp_neighbor = temp_neighbor[nonzero_index]
                    avg = np.mean(nonzero_temp_neighbor, keepdims=True)
                    X_mean_t[i, :, d] = np.tile(avg, X_mean_t.shape[1])
        print("total time cost {}".format(time.time() - start))
        # save X_mean_t into ".npz" file

        # spatial information
        dists_one_all_array = []
        sorted_node_ids_array = []
        for d in range(speed_sequences.shape[2]):
            # 因为知道经纬度 去寻找每个点和其他点的距离
            # 而且从近到远排序
            dists_one_all = dists[d]  # the distance array between node k and all other nodes
            dists_one_all = list(enumerate(dists_one_all))  # [(idx, dist)]
            dists_one_all = sorted(dists_one_all, key=lambda x: x[1])  # by default ascending order
            sorted_node_ids = [x[0] for x in dists_one_all[:S]]  # only take S nearest nodes

            dists_one_all_array.append(dists_one_all)
            sorted_node_ids_array.append(sorted_node_ids)

        nbr_missing_all = missing_index[0].shape[0]
        nbr_finished = 0
        current_ratio = 0
        start = time.time()
        for idx in range(missing_index[0].shape[0]):  # number of missing values
            nbr_finished += 1
            finished_ratio = nbr_finished // (0.01 * nbr_missing_all)
            if finished_ratio != current_ratio:
                end = time.time()
                print("{}% of the statistic features are calculated ! Accumulated time cost: {}s".format(
                    nbr_finished // (0.01 * nbr_missing_all), end - start))
                current_ratio = finished_ratio

            # index in (N, L, D)
            i = missing_index[0][idx]
            j = missing_index[1][idx]
            k = missing_index[2][idx]

            speeds = speed_sequences[i, j]
            # Delta_t, X_last_obsv
            if j != 0 and j != L - 1:  # if the missing value is in the middle of the sequence
                Delta_t[i, j + 1, k] = Delta_t[i, j + 1, k] + Delta_t[i, j, k]
            if j != 0:
                X_last_obsv[i, j, k] = X_last_obsv[
                    i, j - 1, k]  # last observation, can be zero, problem when handling long-range missing values

            # Delta_s, X_closest_obsv
            # 最近点的距离 最近点的观测速度
            dists_one_all = dists_one_all_array[k]  # [(idx, dist)]
            for triple in dists_one_all:
                idx = triple[0]
                dist = triple[1]
                if speeds[idx] != 0:
                    Delta_s[i, j, k] = dist
                    X_closest_obsv[i, j, k] = speeds[idx]
                    break
                else:
                    continue

            # X_mean_s
            sorted_node_ids = sorted_node_ids_array[k]
            spatial_neighbor = speeds[sorted_node_ids]  # S measures
            nonzero_index = np.nonzero(spatial_neighbor)  # return x arrays, for each we have xx elements
            if len(nonzero_index[0]) == 0:
                continue
            else:
                nonzero_spatial_neighbor = spatial_neighbor[nonzero_index]
                X_mean_s[i, j, k] = np.mean(nonzero_spatial_neighbor)

    print('Generate Mask, Last/Closest_observed_X, X_mean_t/s, Delta_t/s finished.')

    if masking:
        # output_dir: "/random_missing/rand"
        np.savez_compressed(
            output_dir + "MissClass_{}.npz".format(name),
            speed_sequences=speed_sequences,
            missing_nodes=missing_nodes,
            Mask=Mask,
            X_last_obsv=X_last_obsv,
            X_closest_obsv=X_closest_obsv,
            X_mean_t=X_mean_t,
            X_mean_s=X_mean_s,
            Delta_t=Delta_t,
            Delta_s=Delta_s,
            dateTime=dateTime,
            speed_labels=speed_labels,
            max_speed=max_speed
        )
    else:
        np.savez_compressed(
            output_dir + "MissClass_{}.npz".format(name),
            speed_sequences=speed_sequences,
            dateTime=dateTime,  # (N, L, D), (N, L), (N, L, D)
            speed_labels=speed_labels,
            max_speed=max_speed
        )


def generate_train_val_test(stat_file, masking,
                            train_val_test_split=[0.7, 0.1, 0.2]):
    start = time.time()
    stat_data = np.load(stat_file)
    if masking:
        # read the stat features from files
        speed_sequences = np.expand_dims(stat_data['speed_sequences'], axis=1)
        Mask = np.expand_dims(stat_data['Mask'], axis=1)
        X_last_obsv = np.expand_dims(stat_data['X_last_obsv'], axis=1)
        X_mean_t = np.expand_dims(stat_data['X_mean_t'], axis=1)
        Delta_t = np.expand_dims(stat_data['Delta_t'], axis=1)
        X_closest_obsv = np.expand_dims(stat_data['X_closest_obsv'], axis=1)
        X_mean_s = np.expand_dims(stat_data['X_mean_s'], axis=1)
        Delta_s = np.expand_dims(stat_data['Delta_s'], axis=1)
        dataset_agger = np.concatenate(
            (speed_sequences, Mask, X_last_obsv, X_mean_t, Delta_t, X_closest_obsv, X_mean_s, Delta_s),
            axis=1)  # (N, 8, L, D)
        x = dataset_agger  # (N, 8, L, D)
    else:
        x = stat_data['speed_sequences']  # (N, L, D)

    dateTime = stat_data['dateTime']
    y = stat_data['speed_labels']
    max_speed = stat_data['max_speed']
    missing_nodes = stat_data['missing_nodes']

    print("x shape: ", x.shape, "dateTime shape: ", dateTime.shape, ", y shape: ", y.shape, ", missing_nodes shape: ",
          missing_nodes.shape)

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_train = round(num_samples * train_val_test_split[0])
    num_test = round(num_samples * train_val_test_split[2])
    num_val = num_samples - num_test - num_train

    x_train, dateTime_train, y_train, missing_nodes_train = x[:num_train], dateTime[:num_train], y[
                                                                                                 :num_train], missing_nodes[
                                                                                                              :num_train]
    x_val, dateTime_val, y_val, missing_nodes_val = (
        x[num_train: num_train + num_val],
        dateTime[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
        missing_nodes[num_train: num_train + num_val]
    )
    x_test, dateTime_test, y_test, missing_nodes_test = x[-num_test:], dateTime[-num_test:], y[
                                                                                             -num_test:], missing_nodes[
                                                                                                          -num_test:]

    for cat in ["train", "val", "test"]:
        _x, _dateTime, _y, _missing_nodes = locals()["x_" + cat], locals()["dateTime_" + cat], locals()["y_" + cat], \
                                            locals()["missing_nodes_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        # x: (N, 8, L, D)
        # dateTime: (N, L)
        # y: (N, L, D)
        file_save_path = stat_file[:-4] + '_' + cat + '.npz'  # e.g., 'x.npz' -> 'x_train.npz'
        np.savez_compressed(
            file=file_save_path,
            x=_x,
            dateTime=_dateTime,
            y=_y,
            missing_nodes=_missing_nodes,
            max_speed=max_speed
        )
    print("The data splitting is finised in {}s with splitting ratio: {}".format(time.time() - start,
                                                                                 str(train_val_test_split)))
    return


if __name__ == "__main__":

    root_path = "../Datasets/"
    datasets = ["PEMS-BAY/", "METR-LA/"]
    dataset = datasets[1]
    data_path = root_path + dataset  # "PEMS-BAY"

    traffic_df_filename = data_path + dataset[:-1].lower() + '.h5'  # raw_hdf file
    dist_filename = data_path + "graph_sensor_locations.csv"

    output_dir = data_path + "class_missing/class"
    masking = True
    L = 12
    S = 5

    adjdata = root_path + dataset + 'adj_mx.pkl'
    _, _, adj_mx = load_pickle(adjdata)
    degree = calculate_degree(adj_mx)
    nodes = np.arange(adj_mx.shape[0])

    conut = np.bincount(degree)
    total = np.sum(conut)
    avg = int(total / 5)
    line1 = 0
    line2 = 0
    line3 = 0
    line4 = 0
    for index, i in enumerate(conut):
        if line1 < avg:
            line1 += i
            x1 = index
        if line2 < avg * 2:
            line2 += i
            x2 = index
        if line3 < avg * 3:
            line3 += i
            x3 = index
        if line4 < avg * 4:
            line4 += i
            x4 = index

    very_low = nodes[np.where(degree < x1)]
    idx = np.random.choice(nodes[np.where(degree == x1)], size=avg - len(very_low), replace=False)
    very_low = np.concatenate((very_low, idx))

    low = np.concatenate(
        (nodes[np.array(np.where((degree > x1) & (degree < x2)))][0], np.setdiff1d(nodes[np.where(degree == x1)], idx)))
    idx = np.random.choice(nodes[np.where(degree == x2)], size=avg - len(low), replace=False)
    low = np.concatenate((low, idx))

    mid = np.concatenate(
        (nodes[np.array(np.where((degree > x2) & (degree < x3)))][0], np.setdiff1d(nodes[np.where(degree == x2)], idx)))
    idx = np.random.choice(nodes[np.where(degree == x3)], size=avg - len(mid), replace=False)
    mid = np.concatenate((mid, idx))

    high = np.concatenate(
        (nodes[np.array(np.where((degree > x3) & (degree < x4)))][0], np.setdiff1d(nodes[np.where(degree == x3)], idx)))
    idx = np.random.choice(nodes[np.where(degree == x4)], size=avg - len(high), replace=False)
    high = np.concatenate((high, idx))

    very_high = np.concatenate(
        (nodes[np.array(np.where(degree > x4))][0], np.setdiff1d(nodes[np.where(degree == x4)], idx)))

    name = ['very_low', 'low', 'mid', 'high', 'very_high']
    i = 0
    for missing_nodes in [very_low, low, mid, high, very_high]:
        generate_stat_features_files2(traffic_df_filename, dist_filename, output_dir, masking, L, S, missing_nodes,
                                      name[i])
        print("{} dataset finish the {} class !".format(dataset, name[i]))
        i += 1

    # 画统计图
    '''
    conut = np.bincount(degree)
    total = np.sum(conut)
    avg = total / 5
    line1 = 0
    line2 = 0
    line3 = 0
    line4 = 0
    for index, i in enumerate(conut):
        if line1 < avg:
            line1 += i
            x1 = index
        if line2 < avg*2:
            line2 += i
            x2 = index
        if line3 < avg*3:
            line3 += i
            x3 = index
        if line4 < avg*4:
            line4 += i
            x4 = index

    conut2 = np.bincount(degree2)
    total2 = np.sum(conut2)
    avg2 = total2 / 5
    line11 = 0
    line22 = 0
    line33 = 0
    line44 = 0
    for index, i in enumerate(conut2):
        if line11 < avg2:
            line11 += i
            x11 = index
        if line22 < avg2 * 2:
            line22 += i
            x22 = index
        if line33 < avg2 * 3:
            line33 += i
            x33 = index
        if line44 < avg2 * 4:
            line44 += i
            x44 = index

    
    
    G = nx.Graph()
    # 添加节点
    G.add_nodes_from(range(adj_mx.shape[0]))

    # 添加边
    for i in range(adj_mx.shape[0]):
        for j in range(i, adj_mx.shape[1]):
            if adj_mx[i][j] == 1:
                G.add_edge(i, j)

    # 绘制图形
    nx.draw(G, with_labels=True)
    plt.show()
    



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.bar(range(len(conut)), conut, color='#5352ed')
    ax1.set_xticks(range(len(conut)))
    ax1.set_title('PEMS-BAY')
    ax1.set_ylabel('Quantity')
    ax1.set_xlabel('Degree of nodes')
    ax1.axvline(x=x1, linestyle='--', color='#eccc68', linewidth=5)
    ax1.axvline(x=x2, linestyle='--', color='#eccc68', linewidth=5)
    ax1.axvline(x=x3, linestyle='--', color='#eccc68', linewidth=5)
    ax1.axvline(x=x4, linestyle='--', color='#eccc68', linewidth=5)


    ax2.bar(range(len(conut2)), conut2, color='#ff6348')
    ax2.set_xticks(range(len(conut2)))
    ax2.set_title('METR-LA')
    ax2.set_ylabel('Quantity')
    ax2.set_xlabel('Degree of nodes')
    ax2.axvline(x=x11, linestyle='--', color='#eccc68', linewidth=5)
    ax2.axvline(x=x22, linestyle='--', color='#eccc68', linewidth=5)
    ax2.axvline(x=x33, linestyle='--', color='#eccc68', linewidth=5)
    ax2.axvline(x=x44, linestyle='--', color='#eccc68', linewidth=5)

    plt.show()
    fig.savefig('degree.jpg', dpi=800)
    '''
