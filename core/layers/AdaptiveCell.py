__author__ = 'yunbo'

import torch
import torch.nn as nn

class AdaptiveCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, distiller_number):
        super(AdaptiveCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.distiller_number = distiller_number
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden * 6, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 6, width, width])
        )
        self.distiller1 = nn.Sequential(nn.Conv2d(num_hidden, num_hidden,
                                   kernel_size=1, stride=1, padding=0, bias=False),
                                nn.LayerNorm([num_hidden * 1, width, width])
                                )

        self.distiller2 = nn.Sequential(nn.Conv2d(num_hidden, num_hidden,
                            kernel_size=1, stride=1, padding=0, bias=False),
                                nn.LayerNorm([num_hidden * 1, width, width])
                                )



    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g, cc_s1, cc_s2 = torch.split(combined_conv, self.num_hidden, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next_ = f * c + i * g
        h_next = o * torch.tanh(c_next_)
        c_s1 = self.distiller1(c_next_)
        c_s2 = self.distiller2(c_next_)
        c_real_s1 = torch.sigmoid(cc_s1) * c_s1 + (1 - torch.sigmoid(cc_s1)) * c_next_
        c_real_s2 = torch.sigmoid(cc_s2) * c_s2 + (1 - torch.sigmoid(cc_s2)) * c_next_
        c_next =  c_real_s2 + c_next_ + c_real_s1

        return h_next, c_next, c_s1, c_s2