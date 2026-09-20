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

    @torch.no_grad()
    def sample_ddim(self, n, num_steps=50, eta=0.0, return_all=False):
        """DDIM sampling with the same trained eps model.

        num_steps is the number of function evaluations (NFE); eta=0 gives
        deterministic sampling. The last transition goes to alpha_bar = 1.
        """
        self.model.eval()
        x = torch.randn(n, self.img_channels, self.img_size, self.img_size,
                        device=self.device)
        seq = torch.linspace(self.T - 1, 0, num_steps).round().long().tolist()
        steps = [x.clone()] if return_all else None
        capture_every = max(1, num_steps // 10)

        for i, t_cur in enumerate(seq):
            t_prev = seq[i + 1] if i + 1 < len(seq) else -1
            t_batch = torch.full((n,), t_cur, device=self.device, dtype=torch.long)
            eps = self.model(x, t_batch)

            ab_t = self.alpha_bars[t_cur]
            if t_prev >= 0:
                ab_prev = self.alpha_bars[t_prev]
            else:
                ab_prev = torch.ones((), device=self.device)

            x0 = ((x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1.0, 1.0)
            sigma = eta * ((1.0 - ab_prev) / (1.0 - ab_t)).sqrt() \
                * (1.0 - ab_t / ab_prev).sqrt()
            x = ab_prev.sqrt() * x0 + (1.0 - ab_prev - sigma ** 2).sqrt() * eps
            if eta > 0:
                x = x + sigma * torch.randn_like(x)

            if return_all and (i + 1) % capture_every == 0:
                steps.append(x.clone())

        self.model.train()
        x = torch.clamp(x, -1.0, 1.0)
        if return_all:
            return x, steps
        return x
