import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu",
                 img_channels=1, img_size=28):
        self.model = model
        self.img_channels = img_channels
        self.img_size = img_size
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1.0 - alpha_bars).sqrt()

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        xt = sqrt_ab * x0 + sqrt_1m_ab * noise
        return xt, noise

    def training_loss(self, x0):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=self.device)
        xt, noise = self.forward_diffusion(x0, t)
        pred_noise = self.model(xt, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, n, return_all=False):
        self.model.eval()
        x = torch.randn(n, self.img_channels, self.img_size, self.img_size,
                        device=self.device)
        steps = []

        for t in reversed(range(self.T)):
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t_batch)

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]

            coeff1 = 1.0 / alpha.sqrt()
            coeff2 = (1.0 - alpha) / (1.0 - alpha_bar).sqrt()
            x = coeff1 * (x - coeff2 * pred_noise)

            if t > 0:
                noise = torch.randn_like(x)
                x = x + beta.sqrt() * noise

            if return_all and (t % 100 == 0 or t == self.T - 1 or t == 0):
                steps.append(x.clone())

        self.model.train()
        x = torch.clamp(x, -1.0, 1.0)
        if return_all:
            return x, steps
        return x
