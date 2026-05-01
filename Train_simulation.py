import torch
import torch.nn as nn
import pandas as pd
from client_net import ClientNetwork
from new_dataset import data_preparing
from transmitter_simulation import Transmitter


class CAT(nn.Module):
    def __init__(
        self,
        seq_len,
        dataset_name,
        batch_size,
        test_size,
        target,
        d_latent,
        h,
        dropout,
        cap_in_dim,
        lr,
        chartevents_path="./CHARTEVENTS.csv",  # بهبود: مسیر قابل تنظیم
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # تعداد feature ها بر اساس dataset
        self.N = 4 if dataset_name == "metavision" else 5

        # client network
        self.network = ClientNetwork(
            self.N, d_latent, h, dropout, seq_len, cap_in_dim, lr
        ).to(self.device)

        # خواندن dataset
        df = pd.read_csv(chartevents_path)

        # آماده‌سازی data loader ها
        self.data = data_preparing(
            df, dataset_name, seq_len, test_size, target, batch_size
        )

        # ماژول ارتباط با server
        self.transmitter = Transmitter(cap_in_dim, self.device)

        # loss functions
        self.L1Loss  = nn.L1Loss()
        self.MSELoss = nn.MSELoss()

    # ─────────────────────────────────────────────────────────────────────────
    # محاسبه MSE و R² روی یک loader
    # ─────────────────────────────────────────────────────────────────────────
    def _evaluate_loader(self, loader):
        """
        بهبود نسبت به کد قبلی:
        - از torch.no_grad() استفاده میشه (کد قبلی نداشت)
        - محاسبه R² درست‌تر شد
        """
        sse = 0.0
        sum_y = 0.0
        sum_y2 = 0.0
        n = 0

        with torch.no_grad():
            for x, l, mask in loader:
                x    = x.to(self.device)
                mask = mask.to(self.device)
                y    = l.to(self.device).view(-1).float()

                # forward pass بدون محاسبه gradient
                v    = self.network(x, mask, train=False)
                yhat = self.transmitter.send_data(v, l.to(self.device), status="test")
                yhat = yhat.view(-1).float()

                diff   = yhat - y
                sse   += (diff * diff).sum().item()
                sum_y += y.sum().item()
                sum_y2+= (y * y).sum().item()
                n     += y.numel()

        if n == 0:
            return float("nan"), float("nan")

        mse = sse / n
        sst = sum_y2 - (sum_y ** 2) / n
        r2  = 1.0 - (sse / sst) if sst > 1e-12 else float("nan")

        return mse, r2

    # ─────────────────────────────────────────────────────────────────────────
    # یک epoch آموزش
    # ─────────────────────────────────────────────────────────────────────────
    def train_one_epoch(self):
        self.network.train()
        self.transmitter.model.train()

        for x, l, mask in self.data.train_loader:
            x    = x.to(self.device)
            mask = mask.to(self.device)

            # forward pass در client
            v, loss_client = self.network(x, mask, train=True)

            # ارسال به server و دریافت gradient
            grad = self.transmitter.send_data(v, l.to(self.device), status="train")

            # بروزرسانی وزن‌های client
            self.network.train_one_batch(loss_client, v, grad.clone())

    # ─────────────────────────────────────────────────────────────────────────
    # ارزیابی یک epoch
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_one_epoch(self):
        self.network.eval()
        self.transmitter.model.eval()

        train_mse, train_r2 = self._evaluate_loader(self.data.train_loader)
        test_mse,  test_r2  = self._evaluate_loader(self.data.test_loader)

        # بهبود: scheduler بر اساس test_mse بروز میشه
        self.network.scheduler.step(test_mse)

        self.network.train()
        self.transmitter.model.train()

        return train_mse, train_r2, test_mse, test_r2

    # ─────────────────────────────────────────────────────────────────────────
    # حلقه اصلی آموزش
    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, epochs):
        history = {
            "loss_train": [],
            "loss_test" : [],
            "r2_train"  : [],
            "r2_test"   : [],
        }

        for epoch in range(epochs):
            self.train_one_epoch()
            train_mse, train_r2, test_mse, test_r2 = self.evaluate_one_epoch()

            print(
                f"[epoch {epoch+1}/{epochs}] "
                f"train_mse={train_mse:.6f}  train_r2={train_r2:.4f} | "
                f"test_mse={test_mse:.6f}  test_r2={test_r2:.4f}"
            )

            history["loss_train"].append(train_mse)
            history["loss_test" ].append(test_mse)
            history["r2_train"  ].append(train_r2)
            history["r2_test"   ].append(test_r2)

        return history

    # ─────────────────────────────────────────────────────────────────────────
    # انتقال دانش از یک CAT به CAT دیگر
    # ─────────────────────────────────────────────────────────────────────────
    def get_knowledge(self, source_cat):
        """
        بهبود نسبت به کد قبلی:
        - torch.no_grad() اضافه شد (کد قبلی نداشت و gradient بیهوده ساخته میشد)
        - پیام خروجی واضح‌تر شد
        """
        source_aes = source_cat.network.multi_autoEncoder.auto_encoders

        with torch.no_grad():
            for i in range(self.N):
                losses = []

                # ارزیابی هر autoencoder از source روی feature i
                for ae in source_aes:
                    loss = self._compute_ae_loss(ae, i)
                    losses.append(loss)

                best_idx = torch.argmin(torch.stack(losses)).item()
                best_loss = losses[best_idx].item()

                print(
                    f"feature {i} → "
                    f"بهترین autoencoder: {best_idx}  "
                    f"(loss={best_loss:.6f})"
                )

                # کپی وزن‌های بهترین autoencoder
                src_weights = source_aes[best_idx].state_dict()
                self.network.multi_autoEncoder.auto_encoders[i].load_state_dict(
                    src_weights
                )

    def _compute_ae_loss(self, auto_encoder, feature_idx):
        """
        محاسبه reconstruction loss یک autoencoder روی یک feature خاص.
        بهبود: auto_encoder به device منتقل میشه قبل از inference.
        """
        auto_encoder = auto_encoder.to(self.device)
        total_loss   = 0.0
        total_n      = 0

        for x, _, mask in self.data.train_loader:
            x    = x.to(self.device)
            mask = mask.to(self.device)
            b, seq_len, _ = x.shape

            inp = x[:, :, feature_idx].reshape(-1, 1)  # (b*seq_len, 1)

            _, rec = auto_encoder(inp)

            rec        = rec.reshape(b, seq_len)
            inp_2d     = inp.reshape(b, seq_len)
            masked_inp = inp_2d * mask
            masked_rec = rec    * mask

            total_loss += self.L1Loss(masked_inp, masked_rec).item() * b
            total_n    += b

        return torch.tensor(total_loss / total_n)