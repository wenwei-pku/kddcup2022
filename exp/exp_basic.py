import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, config):
        self.config = config
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.config['use_gpu']:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.config.gpu) if not self.config.use_multi_gpu else self.config.devices
            device = torch.device('cuda:{}'.format(self.config['cuda_name']))
            print('Use GPU: cuda:{}'.format(self.config['cuda_name']))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
