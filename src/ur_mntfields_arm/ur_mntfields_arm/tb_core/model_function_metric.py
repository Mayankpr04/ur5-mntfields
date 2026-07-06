import torch
from torch import Tensor


torch.backends.cudnn.benchmark = True


class Function:
    def __init__(self, path: str, device: str, network, dim: int):
        self.path = path
        self.device = device
        self.network = network
        self.dim = dim
        self.total_train_loss = []
        self.total_val_loss = []
        self.alpha = 1.025
        limit = 0.5
        self.margin = limit / 15.0
        self.offset = self.margin / 10.0
        self.td_loss_weight = 1.0e-3
        self.speed_loss_weight = 1.0e-2
        self.direct_speed_loss_weight = 0.0
        self.normal_loss_weight = 1.0e-3
        self.normal_cos_loss_weight = 0.0
        self.near_obstacle_loss_weight = 0.0
        self.low_speed_threshold = 0.20
        self.low_speed_pred_max = 0.35
        self.low_speed_penalty_weight = 0.0
        self.effective_speed_floor = 0.05

    def gradient(self, y: Tensor, x: Tensor, create_graph=True) -> Tensor:
        grad_y = torch.ones_like(y)
        grad_x = torch.autograd.grad(
            y,
            x,
            grad_y,
            only_inputs=True,
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        return grad_x

    def TravelTimes(self, Xp):
        tau, _w, _coords = self.network.out(Xp)
        return tau[:, 0]

    def Speed(self, Xp):
        tau, _w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        dt0 = dtau[:, : self.dim]
        s = torch.einsum("ij,ij->i", dt0, dt0)
        return 1 / torch.sqrt(s)

    def Gradient(self, Xp):
        tau, _w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        y0 = -dtau[:, : self.dim]
        s0 = torch.norm(y0, dim=1).view(-1, 1)
        y0 = 1 / (s0**2) * y0
        y1 = -dtau[:, self.dim :]
        s1 = torch.norm(y1, dim=1).view(-1, 1)
        y1 = 1 / (s1**2) * y1
        return torch.cat((y0, y1), dim=1)

    def Loss(self, points, Yobs, normal, beta, gamma, epoch):
        n = Yobs.shape[0]
        tau, _w, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)
        target_speed = torch.clamp(Yobs, min=0.0, max=1.0)
        yobs_safe = torch.clamp(
            target_speed,
            min=float(getattr(self, "effective_speed_floor", 0.05)),
            max=1.0,
        )

        dt0 = dtau[:, : self.dim]
        dt1 = dtau[:, self.dim :]
        s0 = torch.einsum("ij,ij->i", dt0, dt0)
        s1 = torch.einsum("ij,ij->i", dt1, dt1)

        td_weight = float(getattr(self, "td_loss_weight", 1.0e-3))
        with torch.no_grad():
            length0 = (0.03) / (yobs_safe[:, 0]).unsqueeze(1)
            dir0 = length0 * (dt0 * yobs_safe[:, 0].unsqueeze(1) ** 2)
            Xpnew0 = Xp.clone().detach()
            Xpnew0[:, : self.dim] = Xpnew0[:, : self.dim] - dir0
            taunew0, _w, _Xpnew0 = self.network.out(Xpnew0)
            taunew1 = length0
        tau_loss0 = td_weight * ((tau - taunew0 - taunew1) ** 2).squeeze()

        with torch.no_grad():
            length1 = (0.03) / (yobs_safe[:, 1]).unsqueeze(1)
            dir1 = length1 * (dt1 * yobs_safe[:, 1].unsqueeze(1) ** 2)
            Xpnew1 = Xp.clone().detach()
            Xpnew1[:, self.dim :] = Xpnew1[:, self.dim :] - dir1
            taunew0, _w, _Xpnew1 = self.network.out(Xpnew1)
            taunew1 = length1
        tau_loss1 = td_weight * ((tau - taunew0 - taunew1) ** 2).squeeze()

        where_d0 = tau[:, 0] > length0.squeeze()
        where_d1 = tau[:, 0] > length1.squeeze()
        tau_loss0[~where_d0] = 0
        tau_loss1[~where_d1] = 0
        tau_loss = tau_loss0 + tau_loss1

        ypred0 = torch.sqrt(s0 + 1e-8)
        ypred1 = torch.sqrt(s1 + 1e-8)
        yobs0 = yobs_safe[:, 0]
        yobs1 = yobs_safe[:, 1]

        l0 = yobs0 * ypred0
        l1 = yobs1 * ypred1
        l02 = torch.sqrt(l0)
        l12 = torch.sqrt(l1)
        loss_weight = float(getattr(self, "speed_loss_weight", 1.0e-2))
        near_weight = float(getattr(self, "near_obstacle_loss_weight", 0.0))
        risk0 = 1.0 + near_weight * (1.0 - target_speed[:, 0]) ** 2
        risk1 = 1.0 + near_weight * (1.0 - target_speed[:, 1]) ** 2
        loss0 = loss_weight * risk0 * (l02 - 1) ** 2
        loss1 = loss_weight * risk1 * (l12 - 1) ** 2

        pred_speed0 = torch.rsqrt(s0 + 1e-8)
        pred_speed1 = torch.rsqrt(s1 + 1e-8)
        direct_speed_weight = float(getattr(self, "direct_speed_loss_weight", 0.0))
        if direct_speed_weight > 0.0:
            loss0 = loss0 + direct_speed_weight * risk0 * (pred_speed0 - target_speed[:, 0]) ** 2
            loss1 = loss1 + direct_speed_weight * risk1 * (pred_speed1 - target_speed[:, 1]) ** 2
        low_threshold = float(getattr(self, "low_speed_threshold", 0.20))
        low_pred_max = float(getattr(self, "low_speed_pred_max", 0.35))
        low_penalty = float(getattr(self, "low_speed_penalty_weight", 0.0))
        if low_penalty > 0.0 and low_threshold > 0.0:
            low0 = (target_speed[:, 0] <= low_threshold).float()
            low1 = (target_speed[:, 1] <= low_threshold).float()
            loss0 = loss0 + low_penalty * low0 * torch.relu(pred_speed0 - low_pred_max) ** 2
            loss1 = loss1 + low_penalty * low1 * torch.relu(pred_speed1 - low_pred_max) ** 2
        t = tau[:, 0]
        diff = loss0 + loss1

        normal_weight = float(getattr(self, "normal_loss_weight", 1.0e-3))
        normal0 = normal[:, : self.dim]
        normal1 = normal[:, self.dim :]
        n_risk0 = (1.0 + near_weight * (1.0 - target_speed[:, 0]) ** 2).unsqueeze(1)
        n_risk1 = (1.0 + near_weight * (1.0 - target_speed[:, 1]) ** 2).unsqueeze(1)
        n_loss0 = n_risk0 * (1.001 - yobs_safe[:, 0].unsqueeze(1)) * (
            yobs_safe[:, 0].unsqueeze(1) * dt0 + normal0
        ) ** 2
        n_loss1 = n_risk1 * (1.001 - yobs_safe[:, 1].unsqueeze(1)) * (
            yobs_safe[:, 1].unsqueeze(1) * dt1 + normal1
        ) ** 2
        n_loss = normal_weight * (
            torch.sum(n_loss0, dim=1) + torch.sum(n_loss1, dim=1)
        )
        normal_cos_weight = float(getattr(self, "normal_cos_loss_weight", 0.0))
        if normal_cos_weight > 0.0:
            dt0_dir = -dt0 / torch.clamp(torch.linalg.norm(dt0, dim=1, keepdim=True), min=1.0e-8)
            dt1_dir = -dt1 / torch.clamp(torch.linalg.norm(dt1, dim=1, keepdim=True), min=1.0e-8)
            cos0 = torch.sum(dt0_dir * normal0, dim=1)
            cos1 = torch.sum(dt1_dir * normal1, dim=1)
            near0 = (1.0 - target_speed[:, 0]) ** 2
            near1 = (1.0 - target_speed[:, 1]) ** 2
            n_loss = n_loss + normal_cos_weight * (
                near0 * (1.0 - cos0) ** 2 + near1 * (1.0 - cos1) ** 2
            )
        exp_arg = torch.clamp(-0.15 * t, min=-20.0, max=20.0)
        w = torch.exp(exp_arg)
        lossn = torch.sum((diff + n_loss + tau_loss) * w) / n
        loss = beta * lossn
        return loss, lossn, diff
