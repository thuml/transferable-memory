__author__ = 'yunbo'

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            #nn.LayerNorm([num_hidden * 4, width, width])
        )



    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.num_hidden, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next










