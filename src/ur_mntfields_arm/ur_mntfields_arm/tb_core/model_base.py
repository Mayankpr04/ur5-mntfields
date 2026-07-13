import copy
import math
import warnings

import torch

from ur_mntfields_arm.tb_core import model_function_metric as model_function
from ur_mntfields_arm.tb_core import model_network_metric as model_network


class Model:
    def __init__(
        self,
        folder: str,
        dim: int,
        B_scale: float,
        device: str = "cuda:0",
        dim_cells: int = 128,
        init_network: bool = True,
        eval: bool = False,
        lr: float = 1e-4,
        td_loss_weight: float = 1.0e-3,
        speed_loss_weight: float = 1.0e-2,
        log_speed_loss_weight: float = 0.0,
        direct_speed_loss_weight: float = 0.0,
        normal_loss_weight: float = 1.0e-3,
        normal_cos_loss_weight: float = 0.0,
        near_obstacle_loss_weight: float = 0.0,
        low_speed_threshold: float = 0.20,
        low_speed_pred_max: float = 0.35,
        low_speed_penalty_weight: float = 0.0,
        effective_speed_floor: float = 0.05,
    ):
        self.folder = folder
        self.dim = dim
        self.B_scale = float(B_scale)
        if device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                f"[Model] Requested device '{device}' but CUDA is not available. Falling back to CPU.",
                RuntimeWarning,
            )
            device = "cpu"

        self.Params = {
            "Device": device,
            "Pytorch Amp": False,
            "Network": {"Normalisation": "OffsetMinMax"},
            "Training": {
                "Number of sample points": 2e5,
                "Batch Size": 2000,
                "Validation Percentage": 10,
                "Number of Epochs": 20000,
                "Resampling Bounds": (0.1, 0.9),
                "Print Every Epoch": 1,
                "Save Every Epoch": 50,
                "Learning Rate": lr,
                "Random Distance Sampling": True,
                "Use Scheduler": False,
            },
        }
        self.device = device
        self.total_train_loss = []
        self.total_val_loss = []
        self.epoch = 0
        self.frame_idx = 0
        self.network = None
        self.function = None
        self.optimizer = None
        self.scheduler = None
        self.loss_config = {
            "td_loss_weight": float(td_loss_weight),
            "speed_loss_weight": float(speed_loss_weight),
            "log_speed_loss_weight": float(log_speed_loss_weight),
            "direct_speed_loss_weight": float(direct_speed_loss_weight),
            "normal_loss_weight": float(normal_loss_weight),
            "normal_cos_loss_weight": float(normal_cos_loss_weight),
            "near_obstacle_loss_weight": float(near_obstacle_loss_weight),
            "low_speed_threshold": float(low_speed_threshold),
            "low_speed_pred_max": float(low_speed_pred_max),
            "low_speed_penalty_weight": float(low_speed_penalty_weight),
            "effective_speed_floor": float(effective_speed_floor),
        }
        self.B = None
        self.last_loss = 1.0
        self.prev_state_queue = []
        self.prev_loss_queue = []
        self.prev_optimizer_queue = []
        self.timer = []
        if init_network:
            self.init_network()

    def init_network(self):
        self.B = self.B_scale * torch.normal(0, 1, size=(128, self.dim))
        torch.nn.init.trunc_normal_(
            self.B,
            mean=0.0,
            std=self.B_scale,
            a=-2.0 * self.B_scale,
            b=2.0 * self.B_scale,
        )
        self.network = model_network.NN(self.Params["Device"], self.dim, self.B)
        self.network.apply(self.network.init_weights)
        self.network.float()
        self.network.to(self.Params["Device"])
        self.function = model_function.Function(
            self.folder, self.Params["Device"], self.network, self.dim
        )
        self._apply_loss_config()
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.Params["Training"]["Learning Rate"],
            weight_decay=0.1,
        )
        if self.Params["Training"]["Use Scheduler"]:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer
            )

    def _apply_loss_config(self):
        if self.function is None:
            return
        for key, value in self.loss_config.items():
            setattr(self.function, key, float(value))

    def delete_network(self):
        self.network = None
        self.function = None
        self.optimizer = None

    def _batch_loss(self, data: torch.Tensor, beta: float):
        points = data[:, : 2 * self.dim].float()
        speed = torch.clamp(data[:, 2 * self.dim : 2 * self.dim + 2].float(), min=0.0, max=1.0)
        normal = data[:, 2 * self.dim + 2 :].float()
        return self.function.Loss(points, speed, normal, beta, gamma=0.001, epoch=self.epoch)

    def train_batch(self, batch_data: torch.Tensor, accumulation_steps: int = 1):
        if batch_data is None or len(batch_data) == 0:
            return None
        if self.optimizer is None or self.network is None or self.function is None:
            return None

        data = batch_data.to(self.device)
        # Keep the optimizer objective stationary.  Scaling each update by the
        # inverse of the previous raw loss made one large early loss suppress
        # every subsequent gradient (the direct-speed auxiliary can be very
        # large while the randomly initialized time gradient is near zero).
        # AdamW already normalizes update magnitudes; the paper's online
        # training also optimizes Eq. 12 directly without this feedback term.
        beta = 1.0

        self.network.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        total_rows = int(len(data))
        chunks = max(1, min(int(accumulation_steps), total_rows))
        base = total_rows // chunks
        remainder = total_rows % chunks
        start = 0
        weighted_loss_n = 0.0
        for chunk_idx in range(chunks):
            chunk_size = base + (1 if chunk_idx < remainder else 0)
            if chunk_size <= 0:
                continue
            end = start + chunk_size
            loss_value, loss_n, _wv = self._batch_loss(data[start:end], beta)
            # This is equivalent to one batch loss while keeping each
            # second-derivative graph bounded by the CUDA microbatch size.
            (loss_value * (float(chunk_size) / float(total_rows))).backward()
            weighted_loss_n += float(loss_n.detach().item()) * float(chunk_size)
            start = end
        self.optimizer.step()
        self.epoch += 1
        self.last_loss = weighted_loss_n / float(total_rows)
        return self.last_loss

    def train_core(self, epoch, frame_data=None, is_one_frame=True):
        if frame_data is None or len(frame_data) == 0:
            return None, None
        if self.optimizer is None or self.network is None:
            return None, None

        beta = 1.0
        prev_diff = 1.0
        current_diff = 1.0
        total_train_loss = 0.0
        total_diff = 0.0

        cur_data = frame_data.to(self.device)
        n = cur_data.shape[0]
        batch_size = 5000
        max_batches = 6

        for e in range(epoch):
            total_train_loss = 0.0
            total_diff = 0.0

            current_state = {
                k: v.detach().clone() for k, v in self.network.state_dict().items()
            }
            current_optimizer = copy.deepcopy(self.optimizer.state_dict())
            self.prev_state_queue.append(current_state)
            self.prev_optimizer_queue.append(current_optimizer)
            self.prev_loss_queue.append(current_diff)
            if len(self.prev_state_queue) > 5:
                self.prev_state_queue.pop(0)
                self.prev_optimizer_queue.pop(0)
                self.prev_loss_queue.pop(0)

            self.optimizer.param_groups[0]["lr"] = self.Params["Training"][
                "Learning Rate"
            ]
            prev_diff = current_diff
            iter_count = 0

            while True:
                total_train_loss = 0.0
                total_diff = 0.0
                perm = torch.randperm(n, device=self.device)
                shuffled = cur_data[perm]
                n_batches = min(max_batches, math.ceil(n / batch_size))
                for i in range(n_batches):
                    data = shuffled[i * batch_size : (i + 1) * batch_size]
                    points = data[:, : 2 * self.dim].float()
                    speed = data[:, 2 * self.dim : 2 * self.dim + 2].float()
                    normal = data[:, 2 * self.dim + 2 :].float()

                    speed = torch.clamp(speed, min=0.0, max=1.0)

                    gamma = 0.001
                    loss_value, loss_n, _wv = self.function.Loss(
                        points, speed, normal, beta, gamma, epoch
                    )
                    loss_value.backward()
                    if self.optimizer is None:
                        return None, None
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_train_loss += loss_value.detach().item()
                    total_diff += loss_n.detach().item()

                total_train_loss /= n_batches
                total_diff /= n_batches
                if not math.isfinite(total_diff):
                    return None, None
                current_diff = total_diff
                diff_ratio = current_diff / (prev_diff + 1e-12)
                if (0 < diff_ratio < 1.2) or e < 10:
                    break

                iter_count += 1
                if iter_count > 20:
                    break
                with torch.no_grad():
                    best_idx = min(
                        range(len(self.prev_loss_queue)),
                        key=lambda idx: self.prev_loss_queue[idx],
                    )
                    self.network.load_state_dict(
                        self.prev_state_queue[best_idx], strict=True
                    )
                    self.optimizer.load_state_dict(self.prev_optimizer_queue[best_idx])

            if total_diff < 0.001:
                break
            beta = 1.0 / (total_diff + 1e-12)
            if self.scheduler is not None:
                self.scheduler.step(total_train_loss)

        self.last_loss = total_diff
        return total_diff, None
