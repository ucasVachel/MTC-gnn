import os, json
import torch
import numpy as np
from models.slmgnn import slmgnn
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchmetrics
from prettytable import PrettyTable

import argparse
import time
from data.slmgnn_utils import load_dataset, load_adj
from data.generate_dated_data_statistics import generate_train_val_test
from utils.metrics import MAE, MSE, RMSE, MAPE
from utils.tools import EarlyStopping, adjust_learning_rate


def print_table(val_loss_mae, val_loss_rmse, val_loss_mape, test_loss_mae, test_loss_rmse, test_loss_mape):
    # 创建一个表格对象
    table = PrettyTable()

    # 设置表格的列标题
    table.field_names = ['Metric', 'Val-MAE', 'Val-RMSE', 'Val-MAPE', '  ', 'Test-MAE', 'Test-RMSE', 'Test-MAPE']

    # 添加数据
    table.add_row(['Value', round(val_loss_mae, 4), round(val_loss_rmse, 4), round(val_loss_mape, 4), '  ', round(test_loss_mae, 4), round(test_loss_rmse, 4), round(test_loss_mape, 4)])

    # 打印表格
    print(table)

class Exp_slmgnn(object):
    def __init__(self, config):
        self.config = config
        self.data_config = config['Data']
        self.model_config = config['Model']
        self.training_config = config['Training']

        # data config.
        self.root_path = self.data_config['root_path']
        self.data_path = self.data_config['data_path']
        self.dataset_name = self.data_config['dataset_name']
        self.data_split = json.loads(self.data_config['data_split'])  # load list
        self.dist_path = self.data_config['dist_path']
        self.adjdata = self.data_config['adjdata']
        self.adjtype = self.data_config['adjtype']
        self.missing_ratio = float(self.data_config['missing_ratio'])
        self.mask_option = self.data_config['mask_option']
        self.missing_level = self.data_config['missing_level']

        # model config
        self.model_name = self.model_config['model_name']
        self.layers = int(self.model_config['layers'])
        self.add_supports = True

        # training config
        self.use_gpu = json.loads(self.training_config['use_gpu'].lower())
        self.gpu = int(self.training_config['gpu'])
        self.save_path = self.training_config['save_path']
        self.learning_rate = float(self.training_config['learning_rate'])
        self.lr_type = self.training_config['lr_type']
        self.patience = int(self.training_config['patience'])
        self.use_amp = json.loads(self.training_config['use_amp'].lower())
        self.batch_size = int(self.training_config['batch_size'])
        self.train_epochs = int(self.training_config['train_epochs'])

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # result save
        testing_info = "model_{}_missR_{:.2f}_{}".format(
            self.model_name,
            self.missing_ratio * 100,
            self.dataset_name
        )
        self.save_path = self.save_path + testing_info + '/'

    def _build_model(self):
        sensor_ids, sensor_id_to_ind, adj_mx, _ = load_adj(self.adjdata, self.adjtype)
        if self.add_supports == True:
            supports = [torch.tensor(i).to(self.device) for i in adj_mx]
        else:
            supports = None
        model = slmgnn(
            n=adj_mx[0].shape[0],
            imputation=True,
            layers=self.layers,
            supports=supports,
            device=self.device
        )
        return model

    def _acquire_device(self):
        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
            device = torch.device('cuda:{}'.format(self.gpu))
            print('Use GPU: cuda:{}'.format(self.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return model_optim


    def vali(self, vali_loader):
        self.model.eval()
        totle_moss_mse = []
        total_loss_mae = []
        total_loss_rmse = []
        total_loss_mape = []

        for i, (batch_x, batch_dateTime, batch_y, missing_nodes) in enumerate(vali_loader.get_iterator()):
            batch_x = torch.Tensor(batch_x).to(self.device)  # (B, L, D)
            batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, W)
            batch_y = torch.transpose(batch_y, 1, 2) # (B, W, L)
            missing_nodes = torch.Tensor(missing_nodes).long().to(self.device)  # (B, M)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, missing_nodes)  # (B, L, D) -> (B, L, W)
            else:
                outputs = self.model(batch_x, missing_nodes)

            outputs = torch.mul(outputs, torch.Tensor(self.max_speed)).cpu().detach().numpy()  # (B, L, D)
            batch_y = torch.mul(batch_y, torch.Tensor(self.max_speed)).cpu().detach().numpy()

            loss_mse = MSE(outputs, batch_y)
            loss_mae = MAE(outputs, batch_y)
            loss_rmse = RMSE(outputs, batch_y)
            loss_mape = MAPE(outputs, batch_y)

            totle_moss_mse.append(loss_mse)
            total_loss_mae.append(loss_mae)
            total_loss_rmse.append(loss_rmse)
            total_loss_mape.append(loss_mape)

        total_loss_mse = np.average(totle_moss_mse)
        total_loss_mae = np.average(total_loss_mae)
        total_loss_rmse = np.average(total_loss_rmse)
        total_loss_mape = np.average(total_loss_mape)
        self.model.train()
        # return total_loss
        return total_loss_mse, total_loss_rmse, total_loss_mae, total_loss_mape

    def train(self):
        # full_dataset: (N, D)
        if self.mask_option == "random":
            stat_file = os.path.join(self.root_path, "random_missing",
                                     "randMissRatio_{:.2f}%.npz".format(self.missing_ratio * 100))
        else:
            stat_file = os.path.join(self.root_path, "class_missing",
                                     "classMissClass_{}.npz".format(self.missing_level))
        '''
        generate_train_val_test(stat_file,
                                train_val_test_split=self.data_split)
        '''

        # read the pre-processed & splitted dataset from file
        self.dataloader = load_dataset(stat_file, self.batch_size)
        train_loader = self.dataloader['train_loader']
        vali_loader = self.dataloader['val_loader']
        test_loader = self.dataloader['test_loader']
        self.max_speed = self.dataloader['max_speed']
        print("self.max_speed is {}".format(self.max_speed))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        time_now = time.time()

        train_steps = train_loader.size
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        model_optim = self._select_optimizer()

        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # 打乱数据顺序
            train_loader.shuffle()
            for i, (batch_x, batch_dateTime, batch_y, missing_nodes) in enumerate(train_loader.get_iterator()):

                iter_count += 1

                model_optim.zero_grad()
                batch_x = torch.Tensor(batch_x).to(self.device)  # (B, L, D)
                batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, M)
                batch_y = torch.transpose(batch_y, 1, 2)
                missing_nodes = torch.Tensor(missing_nodes).long().to(self.device)  # (B, M)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, missing_nodes)  # (B, L, M)
                else:
                    outputs = self.model(batch_x, missing_nodes)

                outputs = torch.mul(outputs, torch.Tensor(self.max_speed).to(self.device))  # (B, L, D)

                batch_y = torch.mul(batch_y, torch.Tensor(self.max_speed).to(self.device))

                loss = F.mse_loss(outputs, batch_y)  # [N, L, M]
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * (train_steps // self.batch_size) - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            val_loss_mse, val_loss_rmse, val_loss_mae, val_loss_mape = self.vali(vali_loader)
            test_loss_mse, test_loss_rmse, test_loss_mae, test_loss_mape = self.vali(test_loader)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss mae: {2:.7f} ".format(
                    epoch + 1, train_steps, train_loss))
            print_table(val_loss_mae, val_loss_rmse, val_loss_mape, test_loss_mae, test_loss_rmse, test_loss_mape)

            early_stopping(val_loss_mse, self.model, self.save_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.learning_rate, self.lr_type)

        best_model_path = self.save_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def test(self):
        test_loader = self.dataloader['test_loader']
        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_dateTime, batch_y, missing_nodes) in enumerate(test_loader.get_iterator()):
            batch_x = torch.Tensor(batch_x).to(self.device)  # (B, L, D)
            batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, M)
            batch_y = torch.transpose(batch_y, 1, 2)
            missing_nodes = torch.Tensor(missing_nodes).long().to(self.device)  # (B, M)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x,missing_nodes)  # (B, L, D), -> [N, L, W]
            else:
                outputs = self.model(batch_x, missing_nodes)

            outputs = torch.mul(outputs, torch.Tensor(self.max_speed).to(self.device))  # (B, L, D)

            batch_y = torch.mul(batch_y, torch.Tensor(self.max_speed).to(self.device))

            preds.append(outputs.cpu().detach().numpy())
            trues.append(batch_y.cpu().detach().numpy())

        # print('test shape 1:', preds.shape, trues.shape)
        outputs = np.concatenate(preds, axis=0)  # [B, L, D] -> [N, L, D]
        batch_y = np.concatenate(trues, axis=0)  # [B, L, D] -> [N, L, D]
        print('test shape:', outputs.shape, batch_y.shape)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        loss_mae = MAE(outputs, batch_y)
        loss_rmse = RMSE(outputs, batch_y)
        loss_mape = MAPE(outputs, batch_y)

        print('[Average value] mae:{}, rmse:{}, mape:{}'.format(loss_mae, loss_rmse, loss_mape))

        np.save(self.save_path + 'metrics.npy', np.array([loss_mae, loss_rmse, loss_mape]))
        np.save(self.save_path + 'pred.npy', preds)
        np.save(self.save_path + 'true.npy', trues)

        return
