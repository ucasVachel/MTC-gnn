import torch.utils.data as utils
import torch.nn.functional as F
import torch
from models.gnn import gcn_gwnet
import torch.nn as nn


class slmgnn(nn.Module):
    def __init__(self, n, imputation, layers, supports, device):
        super(slmgnn, self).__init__()

        self.n = n
        self.zl = nn.Linear(2 * n, n)
        self.rl = nn.Linear(2 * n, n)
        self.hl = nn.Linear(2 * n, n)
        self.gcn_bool = True
        self.addaptadj = True
        self.supports = supports

        self.imputation = imputation

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.layers = layers

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if self.gcn_bool and self.addaptadj:
            if supports is None:
                self.supports = []
            self.nodevec1 = nn.Parameter(torch.randn(n, 10).to(device), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(10, n).to(device), requires_grad=True).to(device)
            self.supports_len += 1

        self.skip_convs = nn.Conv2d(in_channels=4,
                                    out_channels=32,
                                    kernel_size=(1, 1))

        self.InGRU_convs = nn.Conv2d(in_channels=32,
                                     out_channels=1,
                                     kernel_size=(1, 1))

        self.ReGRU_convs = nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=(1, 1))

        for i in range(layers):
            # conv2d 是二维卷积
            # dilated causal conv.
            new_dilation = 1
            self.filter_convs.append(nn.Conv2d(in_channels=32,
                                               out_channels=32,
                                               kernel_size=(1, 2), dilation=new_dilation))

            # conv1d 是一维卷积
            self.gate_convs.append(nn.Conv2d(in_channels=32,
                                             out_channels=32,
                                             kernel_size=(1, 2), dilation=new_dilation))

            new_dilation *= 2

            self.gconv.append(gcn_gwnet(32, 32, 0.3, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=64,
                                    out_channels=1,
                                    kernel_size=(1, 1),
                                    bias=True)

    def step(self, x, h):
        # test
        x = torch.sigmoid(x)
        xh = torch.cat((x, h), 1)
        z = torch.sigmoid(self.zl(xh))
        r = torch.sigmoid(self.rl(xh))
        combined_r = torch.cat((x, r * h), 1)
        h_tilde = torch.tanh(self.hl(combined_r))
        h = (1 - z) * h + z * h_tilde
        return h

    def forward(self, input, missing_nodes):

        # (batch_size, channels, seq_len, num_nodes)
        batch_size = input.size(0)
        step_size = input.size(2)
        spatial_size = input.size(3)

        M = torch.squeeze(input[:, 1, :, :])

        input = input[:, [0, 3, 5, 6], :, :]
        # (batch_size, channels, seq_len, num_nodes)
        input = input.transpose(2, 3)
        # (batch_size, channels, num_nodes, seq_len)

        input = self.skip_convs(input)
        # (batch_size, 32, num_nodes, seq_len)

        # X = M * X + (1 - M) * delta_x + (1 - M) * x_last_obsv

        skip = 0
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        for i in range(self.layers):
            # 补一维
            input = nn.functional.pad(input, (1, 0, 0, 0))
            # (batch_size, 32, num_nodes, seq_len + 1)
            residual = input
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            try:
                skip = skip[:, :, :, -x.size(3):]
            except:
                skip = 0

            gnn = self.gconv[i](x, new_supports)

            skip = x + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)

            X = torch.squeeze(self.InGRU_convs(x))

            h = self.initHidden(batch_size, spatial_size)

            for i in range(step_size):
                x = X[:, :, i] * M[:, i, :] + h * (1 - M[:, i, :])
                h = self.step(x, h)
                if i != 0:
                    outputs = torch.cat((outputs, h.unsqueeze(1)), dim=1)
                else:
                    outputs = h.unsqueeze(1)

            outputs = self.ReGRU_convs(torch.unsqueeze(outputs, dim=1)).transpose(2, 3)

            input = outputs + residual[:, :, :, -filter.size(3):] + gnn

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [N, L, D, 1]
        x = x[:, 0, :, :].contiguous()  # [N, L, D]
        x = x.transpose(1, 2)
        # (batch_size, num_nodes, seq_len)
        outputs = torch.transpose(x, 1, 2)
        # (batch_size, seq_len, num_nodes)
        outputs = torch.gather(outputs, dim=1, index=missing_nodes.unsqueeze(2).expand(-1, -1, step_size))
        return outputs

    def initHidden(self, batch_size, spatial_size):
        Hidden_State = torch.zeros((batch_size, spatial_size)).to('cuda:0')
        return Hidden_State
