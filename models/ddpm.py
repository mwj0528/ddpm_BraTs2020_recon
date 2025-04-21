import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DDPM(nn.Module):
    def __init__(self, unet, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__()
        self.model = unet
        self.device = device
        self.T = timesteps

        # 1. 베타 스케줄
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # ᾱ_t

    def noise_images(self, x0, t):
        """
        x0 (B, C, H, W): 원본 이미지
        t  (B): timestep
        return: noisy image x_t, added noise ε
        """
        B, C, H, W = x0.shape
        noise = torch.randn_like(x0).to(self.device)

        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise

    def forward(self, x0):
        """
        학습 시 forward 함수
        x0: (B, 1, H, W)
        return: loss
        """
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device).long()
        x_t, noise = self.noise_images(x0, t)

        pred_noise = self.model(x_t, t)

        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, img_size, sample_steps=None):
        """
        Sampling: noise에서 T1 이미지 생성
        img_size: (C, H, W)
        """
        self.eval()
        C, H, W = img_size
        x = torch.randn(1, C, H, W).to(self.device)
        sample_steps = sample_steps or self.T

        for t in reversed(range(sample_steps)):
            t_batch = torch.tensor([t], device=self.device)

            beta = self.betas[t]
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]

            noise_pred = self.model(x, t_batch)
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = 0

            x = (
                1 / torch.sqrt(alpha)
                * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * noise_pred)
                + torch.sqrt(beta) * z
            )

        return x

    def train_step(self, x0, optimizer):
        """
        한 step 학습 (forward + backward + optimizer step)
        :param x0: 원본 이미지 배치 (B, C, H, W)
        :param optimizer: 옵티마이저
        :return: loss (torch.Tensor)
        """
        self.train()
        optimizer.zero_grad()
        loss = self.forward(x0)
        loss.backward()
        optimizer.step()
        return loss
