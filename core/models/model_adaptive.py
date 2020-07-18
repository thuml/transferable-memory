import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import convlstm, adaptive_model
from torch.nn.parameter import Parameter


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            "convlstm" : convlstm.RNN,
            'adaptive_model' : adaptive_model.RNN
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
            self.network_teacher1 = convlstm.RNN(self.num_layers, self.num_hidden, configs).to(configs.device)
            self.network_teacher2 = convlstm.RNN(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.softmax = nn.Softmax(dim=0)


        self.MSE_criterion = nn.MSELoss()
        self.alpha = 0.9

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path1, checkpoint_path2):
        print('load model:', checkpoint_path1, checkpoint_path2)
        stats1 = torch.load(checkpoint_path1)
        stats2 = torch.load(checkpoint_path2)
        self.network_teacher1.load_state_dict(stats1['net_param'])
        self.network_teacher2.load_state_dict(stats2['net_param'])
        self.network_teacher1.eval()
        self.network_teacher2.eval()


    def train(self, frames, mask, iter):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        with torch.no_grad():
            next_frames_, cell_list_source = self.network_teacher1(frames_tensor, mask_tensor)
            next_frames_, cell_list_target = self.network_teacher2(frames_tensor, mask_tensor)
        next_frames, l1_loss = self.network(frames_tensor, mask_tensor, cell_list_source, cell_list_target, True, iter)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        with torch.no_grad():
            next_frames = self.network(frames_tensor, mask_tensor, None, None, False)
        return next_frames.detach().cpu().numpy()