import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Sinusoidal Positional Embedding (for timestep)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]  # (B, dim/2)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # (B, dim)


# Residual block with timestep embedding
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h += self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        h = self.block2(h)
        return h + self.residual_conv(x)


# Down / Up sampling
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.block(x, t)
        return self.pool(x), x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.block = ResBlock(in_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # skip connection
        return self.block(x, t)


# Full UNet
class ContextUnet(nn.Module):
    def __init__(self, time_emb_dim=256, in_channels=1, base_channels=64):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Encoder
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1 = Down(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = Down(base_channels * 2, base_channels * 4, time_emb_dim)

        # Bottleneck
        self.bot1 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.bot2 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Decoder
        self.up1 = Up(base_channels * 8, base_channels * 2, time_emb_dim)
        self.up2 = Up(base_channels * 4, base_channels, time_emb_dim)

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, 1)
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        x = self.init_conv(x)
        x1_pooled, x1 = self.down1(x, t_emb)
        x2_pooled, x2 = self.down2(x1_pooled, t_emb)

        x_bot = self.bot1(x2_pooled, t_emb)
        x_bot = self.bot2(x_bot, t_emb)

        x = self.up1(x_bot, x2, t_emb)
        x = self.up2(x, x1, t_emb)

        return self.final_conv(x)
