import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear


torch.backends.cudnn.benchmark = True


def sigmoid_out(input: Tensor) -> Tensor:
    return torch.sigmoid(0.1 * input)


class SigmoidOut(torch.nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return sigmoid_out(input)


class NN(torch.nn.Module):
    def __init__(self, device: str, dim: int, B: Tensor):
        super().__init__()
        self.dim = dim
        self.B = B.T.to(device)
        self._fourier_w = 2.0 * math.pi * self.B
        self.scale = 10
        self.act = torch.nn.Softplus(beta=self.scale)
        self.actout = SigmoidOut()
        self.nl1 = 2
        h_size = 256
        fourier_dim = 2 * self.B.shape[1]

        self.encoder = torch.nn.ModuleList()
        self.encoder_norm = LayerNorm(h_size)
        self.encoder.append(Linear(fourier_dim, h_size))
        for _ in range(3 * self.nl1):
            self.encoder.append(Linear(h_size, h_size))
        self.encoder.append(Linear(h_size, h_size))

        self.gate = torch.nn.ModuleList()
        for _ in range(self.nl1):
            self.gate.append(Linear(1, 1))

        self.pe_gate = torch.nn.ModuleList()
        self.pe_gate.append(Linear(h_size, h_size))
        self.pe_gate.append(Linear(h_size, h_size))

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            stdv = np.sqrt(2.0 / (m.weight.size(0) + m.weight.size(1)))
            torch.nn.init.trunc_normal_(
                m.weight, mean=0.0, std=stdv, a=-2.0 * stdv, b=2.0 * stdv
            )
            m.bias.data.fill_(0.0)

        for gate in self.gate:
            gate.weight.data.fill_(0.0)
            gate.bias.data.fill_(0.0)

    def input_mapping(self, x: Tensor) -> Tensor:
        x_proj = x @ self._fourier_w
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def lip_norm(self, w: Tensor) -> Tensor:
        absrowsum = torch.sqrt(torch.sum(w**2, dim=1)).detach()
        scale = 1 + 1e-5 - self.act(1 - 1 / absrowsum)
        return w * scale.unsqueeze(1)

    def out(self, coords: Tensor):
        coords = coords.clone().detach().requires_grad_(True)
        size = coords.shape[0]

        x0 = coords[:, : self.dim]
        x1 = coords[:, self.dim :]
        x = torch.vstack((x0, x1))
        x = self.input_mapping(x)

        w = self.lip_norm(self.pe_gate[0].weight)
        b = self.pe_gate[0].bias
        u = torch.sin(x @ w.T + b)

        w = self.lip_norm(self.pe_gate[1].weight)
        b = self.pe_gate[1].bias
        v = torch.sin(x @ w.T + b)

        for ii in range(self.nl1):
            x_tmp = x

            w = self.lip_norm(self.encoder[3 * ii + 1].weight)
            b = self.encoder[3 * ii + 1].bias
            y = x @ w.T + b
            x = u * torch.sin(y) + v * (1 - torch.sin(y))

            w = self.lip_norm(self.encoder[3 * ii + 2].weight)
            b = self.encoder[3 * ii + 2].bias
            y = x @ w.T + b
            x = u * torch.sin(y) + v * (1 - torch.sin(y))

            w = self.lip_norm(self.encoder[3 * ii + 3].weight)
            b = self.encoder[3 * ii + 3].bias
            y = x @ w.T + b

            weight = torch.sigmoid(0.1 * self.gate[ii].weight)
            x = (1 - weight) * x_tmp + weight * torch.sin(y)

        w = self.lip_norm(self.encoder[-1].weight)
        b = self.encoder[-1].bias
        y = x @ w.T + b
        y = self.encoder_norm(y)

        x0 = y[:size, ...]
        x1 = y[size:, ...]
        x = torch.sqrt((x0 - x1) ** 2 + 1e-6)
        x = x.view(x.shape[0], -1, 16)
        x = (torch.logsumexp(10 * x, dim=2) - np.log(16)) / 10
        x = 0.2 * torch.sum(x, dim=1, keepdim=True)
        return x, w, coords

    def forward(self, coords: Tensor):
        coords = coords.clone().detach().requires_grad_(True)
        output, _w, coords = self.out(coords)
        return output, coords
