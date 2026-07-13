import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import Linear


torch.backends.cudnn.benchmark = True


class NN(torch.nn.Module):
    """Paper-faithful factorized NTField network.

    The network predicts the bounded factor ``tau(q0, q1)``.  The arrival
    time ``T = log(tau)^2 * ||q0 - q1||`` and all of its derivatives are
    formed in :mod:`model_function_metric`, where the C-space dimension is
    known.  Keeping that factorization outside this module makes it harder to
    accidentally use gradients of tau as gradients of arrival time.
    """

    ARCHITECTURE_VERSION = "pinn_factorized_t_v2"

    def __init__(self, device: str, dim: int, B: Tensor):
        super().__init__()
        self.dim = int(dim)
        self.B = B.T.to(device)
        self._fourier_w = 2.0 * math.pi * self.B
        self.hidden_dim = 128
        fourier_dim = 2 * self.B.shape[1]

        # Eq. 8 and Fig. 3(a): one 256->128 sine layer followed by
        # three 128->128 sine layers.
        self.encoder = torch.nn.ModuleList([Linear(fourier_dim, self.hidden_dim)])
        self.encoder.extend(Linear(self.hidden_dim, self.hidden_dim) for _ in range(3))

        # Fig. 3(b): squared-difference symmetric feature, then three
        # Softplus layers and a bounded scalar tau generator.
        self.generator = torch.nn.ModuleList(
            Linear(self.hidden_dim, self.hidden_dim) for _ in range(3)
        )
        self.output = Linear(self.hidden_dim, 1)
        self.softplus = torch.nn.Softplus(beta=10.0)

    @staticmethod
    def init_weights(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            stdv = np.sqrt(2.0 / (module.weight.size(0) + module.weight.size(1)))
            torch.nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=stdv,
                a=-2.0 * stdv,
                b=2.0 * stdv,
            )
            module.bias.data.zero_()

    def input_mapping(self, x: Tensor) -> Tensor:
        x_proj = x @ self._fourier_w
        return torch.cat((torch.cos(x_proj), torch.sin(x_proj)), dim=-1)

    def _encode(self, q: Tensor) -> Tensor:
        x = self.input_mapping(q)
        for layer in self.encoder:
            x = torch.sin(layer(x))
        return x

    def out(self, coords: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        coords = coords.clone().detach().requires_grad_(True)
        size = coords.shape[0]
        if coords.ndim != 2 or coords.shape[1] != 2 * self.dim:
            raise ValueError(
                f"Expected pair coordinates [N,{2 * self.dim}], got {tuple(coords.shape)}"
            )

        encoded = self._encode(torch.vstack((coords[:, : self.dim], coords[:, self.dim :])))
        enc0 = encoded[:size]
        enc1 = encoded[size:]
        symmetric_feature = torch.square(enc0 - enc1)

        x = symmetric_feature
        for layer in self.generator:
            x = self.softplus(layer(x))
        tau = torch.sigmoid(self.output(x))
        return tau, self.output.weight, coords

    def forward(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        output, _w, mapped_coords = self.out(coords)
        return output, mapped_coords
