import torch
import torch.nn as nn
import torch.optim as optim
import math


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model باید بر h بخش‌پذیر باشد"
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.shape[-1]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask_ = mask.to(device=scores.device)
            if mask_.dtype != torch.bool:
                mask_ = (mask_ != 0)
            scores = scores.masked_fill(
                ~mask_.unsqueeze(1).unsqueeze(2), -1e9
            )
        scores = scores.softmax(dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ value, scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q).view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
        key   = self.w_k(k).view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


# ─── AutoEncoder بهینه‌شده ─────────────────────────────────────────────────
class encoder(nn.Module):
    """
    بهبود: اضافه شدن لایه میانی اضافه (1→8→d_latent) برای یادگیری
    representation غنی‌تر از هر feature.
    """
    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, d_latent),
        )

    def forward(self, x):
        return self.net(x)


class decoder(nn.Module):
    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


class Auto_encoder(nn.Module):
    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.encoder = encoder(d_latent)
        self.decoder = decoder(d_latent)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class Multi_auto_encoder(nn.Module):
    def __init__(self, d_latent: int, N: int) -> None:
        super().__init__()
        self.auto_encoders = nn.ModuleList([Auto_encoder(d_latent) for _ in range(N)])

    def forward(self, x):
        b, seq_len, N = x.shape
        out_enc, out_dec = [], []
        for i, ae in enumerate(self.auto_encoders):
            z, rec = ae(x[:, :, i].reshape(-1, 1))
            out_enc.append(z.reshape(b, seq_len, -1))
            out_dec.append(rec.reshape(b, seq_len, -1))
        return torch.cat(out_enc, dim=-1), torch.cat(out_dec, dim=-1)


# ─── Compressor بهینه‌شده با Attention Pooling ────────────────────────────
class AttentionPooling(nn.Module):
    """
    بهبود مهم: به جای جمع ساده، از attention pooling استفاده می‌کنیم.
    این روش یاد می‌گیرد کدام time-step‌ها مهم‌ترند.
    همچنین GRU برای capture temporal context اضافه شد.
    """
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        # GRU برای capture کردن الگوهای زمانی
        self.gru = nn.GRU(d_in, d_in, batch_first=True, bidirectional=False)
        # attention scorer: هر time-step یک score می‌گیرد
        self.score = nn.Linear(d_in, 1)
        # projection به فضای خروجی
        self.proj = nn.Linear(d_in, d_out)
        # decoder برای بازسازی (reconstruction loss)
        self.decod = nn.Linear(d_out, d_in)
        self.d_out = d_out

    def forward(self, x, mask):
        # x: (b, seq_len, d_in) | mask: (b, seq_len)
        # گذر از GRU برای enriched context
        gru_out, _ = self.gru(x)           # (b, seq_len, d_in)

        # محاسبه attention scores و mask کردن padding
        scores = self.score(gru_out).squeeze(-1)  # (b, seq_len)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)   # (b, seq_len)

        # weighted sum: خلاصه کل sequence
        v_raw = (gru_out * weights.unsqueeze(-1)).sum(dim=1)  # (b, d_in)
        v = self.proj(v_raw)                                   # (b, d_out)

        # decoder برای reconstruction loss
        rec = self.decod(v).unsqueeze(1).expand_as(gru_out)   # (b, seq_len, d_in)
        return v, rec


# ─── ClientNetwork نهایی ──────────────────────────────────────────────────
class ClientNetwork(nn.Module):
    def __init__(self, N, d_latent, h, dropout, seq_len, cap_in_dim, lr):
        super().__init__()
        d_model = N * d_latent

        self.multi_autoEncoder = Multi_auto_encoder(d_latent, N)
        self.PE = PositionalEncoding(d_model, seq_len, dropout)
        self.attention = MultiHeadAttentionBlock(d_model, h, dropout)
        self.residual = ResidualConnection(d_model, dropout)
        self.norm = LayerNormalization(d_model)

        # جایگزین compressor قدیمی با attention pooling
        self.pooler = AttentionPooling(d_model, cap_in_dim)

        self.loss_fn = nn.L1Loss()
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        # scheduler برای کاهش lr در صورت توقف پیشرفت
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, factor=0.5
        )

    def forward(self, x, mask, train=True):
        mask = mask.to(x.device)

        # encode هر feature جداگانه
        out_enc, out_dec1 = self.multi_autoEncoder(x)

        # positional encoding
        out_enc = self.PE(out_enc)

        # self-attention با residual
        out_att = self.residual(
            out_enc,
            lambda z: self.attention(z, z, z, mask)
        )
        out_att = self.norm(out_att)

        # pooling آگاه به زمان
        v, out_dec2 = self.pooler(out_att, mask)

        if train:
            loss1 = self.loss_fn(x, out_dec1)
            # مقایسه با out_att (نه out_dec1) برای reconstruction loss درست
            loss2 = self.loss_fn(out_att, out_dec2)
            return v, loss1 + loss2
        else:
            return v

    def train_one_batch(self, loss_client, v, grad_back):
        """
        اصلاح باگ مهم: در کد قبلی optimizer.zero_grad() دو بار صدا زده
        می‌شد که gradient های loss_client از دست می‌رفت.
        حالا فقط یک بار در ابتدا zero می‌کنیم.
        """
        self.optimizer.zero_grad()
        # محاسبه gradient های client loss
        loss_client.backward(retain_graph=True)
        # اضافه کردن gradient از سمت server
        grad_back = grad_back.to(v.device)
        v.backward(grad_back)
        # clip برای جلوگیری از exploding gradient
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()