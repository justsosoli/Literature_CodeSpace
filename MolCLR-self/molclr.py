import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.nt_xent import NTXentLoss

def _save_config_file(model_checkpoints_folder):
    """
    这个函数用于在指定的模型检查点文件夹（model_checkpoints_folder）下保存当前的配置文件 config.yaml。
    具体步骤如下：
    1. 检查 model_checkpoints_folder 路径是否存在；
    2. 如果不存在，则先创建该文件夹；
    3. 然后将项目根目录下的 config.yaml 文件复制到新建的 model_checkpoints_folder 下，
       保存名称同为 config.yaml。
    这样做的目的是每次训练或保存模型时，把本次运行的配置文件一同备份到对应的结果文件夹里，
    方便后续复现或分析不同实验结果时查阅参数设置。
    """
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class MolCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join('ckpt', dir_name)

        self.writer = SummaryWriter(log_dir = log_dir)

        self.dataset = dataset

        self.nt_xent_criterion = NTXentLoss(device=self.device, batch_size = config['batch_size'], **config['loss'])

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print('Running on:', device)
        return device

    def _step(self, model, xis, xjs, n_iter):
        ris, zis = model(xis)
        rjs, zjs = model(xjs)

        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loader()
        if self.config['model_type'] == 'gcn':
            from models.gcn_model import GCN
            model = GCN(**self.config['model']).to(self.device)
            model = self._load_pre_trained_weights(model)
        
        else:
            raise ValueError(f"Invalid model type: {self.config['model_type']}")
        print(model)

        optimizer = torch.optim.Adam(
            model.parameters(),
            float(self.config['init_lr']),
            weight_decay = float(self.config['weight_decay'])
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max = self.config['epochs'] - self.config['warm_up'],
            eta_min = 0,
            last_epoch = -1
        )

        # model_checkpoint_path = './ckpt/checkpoints'
        model_checkpoint_path = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoint_path)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            print(len(train_loader))

            for bn, (xis, xjs) in enumerate(train_loader):
                optimizer.zero_grad()
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                loss.backward()
                optimizer.step()
                n_iter += 1

        if epoch_counter % self.config['eval_every_n_epochs'] == 0:
            valid_loss = self._validate(model, valid_loader)
            print(epoch_counter, valid_loss,"(validation)")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(model_checkpoint_path, 'best_model.pth'))

            self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
            valid_n_iter += 1

        if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
            torch.save(model.state_dict(), os.path.join(model_checkpoint_path, 'model_{}.pth'.format(str(epoch_counter))))

        if epoch_counter >= self.config['warm_up']:
            scheduler.step()

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'best_model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
        return model

    def _validate(self, model, valid_loader):
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            counter = 0
            for xis, xjs in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        
        model.train()
        return valid_loss


def main():
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
    
    if config['aug'] == 'node':
        from datasets.dataset import MoleculeDatasetWrapper
    else:
        raise ValueError('Not defined molecule augmentation!')

    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    molclr = MolCLR(dataset, config)
    molclr.train()

if __name__ == '__main__':
    main()