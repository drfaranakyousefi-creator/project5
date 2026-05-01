import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ─── Squash ───────────────────────────────────────────────────────────────────
def squash(x, dim=-1):
    """
    تابع فعال‌سازی capsule net.
    بردار رو به بازه [0,1) نگاشت می‌کنه بدون اینکه جهتش عوض بشه.
    این تابع در کد اصلی درست بود — فقط حفظش کردیم.
    """
    norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
    norm    = torch.sqrt(norm_sq + 1e-9)
    scale   = norm_sq / (1.0 + norm_sq) / norm
    return scale * x


# ─── Primary Capsules ─────────────────────────────────────────────────────────
class PrimaryCapsules(nn.Module):
    """
    ورودی رو reshape می‌کنه به کپسول‌های اولیه.
    در کد اصلی این بخش درست بود — فقط squash اضافه شد.
    """
    def __init__(self, n_caps: int, input_dim: int) -> None:
        super().__init__()
        assert input_dim % n_caps == 0, \
            f"input_dim={input_dim} باید بر n_caps={n_caps} بخش‌پذیر باشد"
        self.n_caps = n_caps
        self.caps_dim = input_dim // n_caps

    def forward(self, x):
        # x: (batch, input_dim)
        b = x.shape[0]
        x = x.reshape(b, self.n_caps, self.caps_dim)
        # squash روی هر کپسول اعمال می‌شه — در کد اصلی این نبود!
        return squash(x, dim=-1)


# ─── Secondary Capsules ───────────────────────────────────────────────────────
class SecondaryCapsules(nn.Module):
    """
    باگ اصلی کد قبلی: شکل W اشتباه بود.

    قبلی (اشتباه):
        self.W = nn.Parameter(torch.randn(n_caps_in, caps_input_dim, caps_out_dim))
        → n_caps_out نادیده گرفته شده بود!

    درست:
        self.W = nn.Parameter(torch.randn(n_caps_in, n_caps_out, caps_input_dim, caps_out_dim))
        → هر کپسول ورودی یک ماتریس تبدیل جداگانه برای هر کپسول خروجی دارد.
    """
    def __init__(
        self,
        n_caps_in: int,
        n_caps_out: int,
        caps_input_dim: int,
        caps_out_dim: int,
        n_routing: int = 3,
    ) -> None:
        super().__init__()
        # شکل درست: (n_caps_in, n_caps_out, caps_input_dim, caps_out_dim)
        self.W = nn.Parameter(
            torch.randn(n_caps_in, n_caps_out, caps_input_dim, caps_out_dim) * 0.01
        )
        self.n_caps_out = n_caps_out
        self.n_routing  = n_routing

    def forward(self, x):
        # x: (batch, n_caps_in, caps_input_dim)
        b, n_caps_in, _ = x.shape

        # u_hat: پیش‌بینی هر کپسول ورودی از هر کپسول خروجی
        # x → (b, n_caps_in, 1, caps_input_dim, 1)
        x_exp = x.unsqueeze(2).unsqueeze(-1)
        # W → (1, n_caps_in, n_caps_out, caps_input_dim, caps_out_dim)
        W_exp = self.W.unsqueeze(0)
        # u_hat: (b, n_caps_in, n_caps_out, caps_out_dim)
        u_hat = torch.matmul(W_exp.transpose(-2, -1), x_exp).squeeze(-1)

        # Dynamic Routing
        # b_ij: لاگیت‌های اولیه routing (همه صفر)
        b_ij = torch.zeros(b, n_caps_in, self.n_caps_out, device=x.device)

        for r in range(self.n_routing):
            # c_ij: ضرایب coupling از softmax
            c_ij = F.softmax(b_ij, dim=2)  # (b, n_caps_in, n_caps_out)

            # s_j: جمع وزن‌دار پیش‌بینی‌ها
            # c_ij → (b, n_caps_in, n_caps_out, 1)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)  # (b, n_caps_out, caps_out_dim)

            # v_j: squash
            v_j = squash(s_j, dim=-1)  # (b, n_caps_out, caps_out_dim)

            # بروزرسانی b_ij بر اساس agreement (فقط برای iteration‌های میانی)
            if r < self.n_routing - 1:
                # agreement: dot product بین u_hat و v_j
                # u_hat: (b, n_caps_in, n_caps_out, caps_out_dim)
                # v_j:   (b, n_caps_out, caps_out_dim) → (b, 1, n_caps_out, caps_out_dim)
                agreement = (u_hat * v_j.unsqueeze(1)).sum(dim=-1)  # (b, n_caps_in, n_caps_out)
                b_ij = b_ij + agreement

        return v_j  # (b, n_caps_out, caps_out_dim)


# ─── Prediction Network ───────────────────────────────────────────────────────
class prediction_net(nn.Module):
    """
    بهبودها نسبت به کد اصلی:
    1. شکل W در SecondaryCapsules اصلاح شد (باگ اصلی)
    2. squash روی primary capsules هم اعمال می‌شه
    3. gradient clip اضافه شد
    4. Layer Normalization قبل از capsules اضافه شد
    """
    def __init__(
        self,
        d_in: int,
        n_input_caps: int,
        n_output_caps: int,
        in_caps_dim: int,
        out_caps_dim: int,
        n_routing: int = 3,
        lr: float = 0.001,
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # لایه‌های projection
        self.layer1 = nn.Linear(d_in, 32)
        self.layer2 = nn.Linear(32, n_input_caps * in_caps_dim)
        self.norm   = nn.LayerNorm(n_input_caps * in_caps_dim)
        self.act    = nn.GELU()

        # capsule layers
        self.primary_caps   = PrimaryCapsules(n_input_caps, n_input_caps * in_caps_dim)
        self.secondary_caps = SecondaryCapsules(
            n_input_caps, n_output_caps, in_caps_dim, out_caps_dim, n_routing
        )

        # خروجی نهایی
        self.final = nn.Linear(n_output_caps * out_caps_dim, 1)

        self.loss_fn   = nn.MSELoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.to(self.device)

    def prediction(self, x):
        # x: (batch, d_in)
        x = self.act(self.layer1(x))
        x = self.norm(self.act(self.layer2(x)))
        x = self.primary_caps(x)    # (b, n_caps_in, in_caps_dim)  + squash
        x = self.secondary_caps(x)  # (b, n_caps_out, out_caps_dim) + routing
        x = x.reshape(x.shape[0], -1)
        return self.final(x)        # (b, 1)

    def forward(self, combined_embedded, label=None, status='test'):
        inp = torch.tensor(combined_embedded, dtype=torch.float, device=self.device)
        inp.requires_grad_(True)

        if status == 'train':
            lbl = torch.tensor(label, dtype=torch.float, device=self.device).view(-1, 1)
            self.optimizer.zero_grad()
            out  = self.prediction(inp)
            loss = self.loss_fn(out, lbl)
            loss.backward()
            # gradient برای بازگشت به client
            grad = inp.grad.detach().cpu().tolist()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            return {'grad': grad}
        else:
            with torch.no_grad():
                out = self.prediction(inp)
            return {'prediction': out.cpu().tolist()}