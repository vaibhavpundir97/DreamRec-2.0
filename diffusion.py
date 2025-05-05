import math
import torch
import torch.nn.functional as F


def extract(a, t, x_shape):
    """Index tensor a at timesteps t and reshape to x_shape."""
    batch = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch, *((1,) * (len(x_shape) - 1))).to(t.device)


# —— Beta schedules (same as teacher) —— #
def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s)/(1+s)*math.pi*0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2*timesteps+1, timesteps)
    return 1 - torch.exp(- beta_min/timesteps - x*0.5*(beta_max-beta_min)/(timesteps*timesteps))


# —— Original diffusion class for teacher rollout —— #
class Teacher:
    def __init__(self, timesteps, beta_start, beta_end, beta_sche, w):
        self.timesteps = timesteps
        self.w = w
        if beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps)
        elif beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError("Unknown beta_sche")

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_omacp = torch.sqrt(1 - self.alphas_cumprod)

        self.posterior_mean_coef1 = self.betas * \
            torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * \
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return extract(self.sqrt_acp, t, x0.shape) * x0 + extract(self.sqrt_omacp, t, x0.shape) * noise

    @torch.no_grad()
    def p_sample(self, model_fwd, model_unfwd, x, h, t, t_index):
        # one step posterior sample
        x_start = (1 + self.w) * model_fwd(x, h, t) - \
            self.w * model_unfwd(x, t)
        mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        if t_index == 0:
            return mean
        var = extract(self.posterior_variance, t, x.shape)
        return mean + torch.sqrt(var) * torch.randn_like(x)


# —— Simple noise sampler for student inference —— #
class Student:
    def __init__(self, timesteps, beta_start, beta_end, sche):
        self.timesteps = timesteps
        if sche == 'linear':
            self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif sche == 'exp':
            self.betas = exp_beta_schedule(timesteps)
        elif sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError("Unknown sche")
        self.alphas = 1 - self.betas
        self.acp = torch.cumprod(self.alphas, dim=0)
        self.sqrt_acp = torch.sqrt(self.acp)
        self.sqrt_omacp = torch.sqrt(1 - self.acp)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return extract(self.sqrt_acp, t, x0.shape) * x0 + extract(self.sqrt_omacp, t, x0.shape) * noise

    def sample(self, model_fw, model_uncond, h, device, steps):
        """
        Few-step inference by sampling 'steps' noise levels once,
        sorting them in descending order, and iteratively denoising.
        """
        B = h.size(0)
        # initialize from pure noise
        z = torch.randn_like(h).to(device)
        # sample random noise levels excluding 0, then add 0
        rand_levels = torch.randint(
            1, self.timesteps, (steps - 1,), device=device, dtype=torch.long)
        zero_level = torch.zeros(1, device=device, dtype=torch.long)
        t_vals = torch.cat([rand_levels, zero_level], dim=0)
        # sort in descending order
        t_vals, _ = torch.sort(t_vals, descending=True)

        # iterative denoising
        for i, t in enumerate(t_vals):
            tb = t.expand(B)
            # predict clean embedding at this noise level
            x0_pred = model_fw(z, h, tb)
            # final step: return prediction
            if i == steps - 1:
                return x0_pred
            # compute implicit noise
            sqrt1 = extract(self.sqrt_acp, tb, z.shape)
            sqrt2 = extract(self.sqrt_omacp, tb, z.shape)
            eps = (z - sqrt1 * x0_pred) / sqrt2
            # next noise level
            next_t = t_vals[i + 1]
            tn = next_t.expand(B)
            # re-noise to next level
            z = extract(self.sqrt_acp, tn, z.shape) * x0_pred + \
                extract(self.sqrt_omacp, tn, z.shape) * eps
        return x0_pred


class Consistency:
    """
    Provides q_sample and one-step sampling for consistency models.
    """

    def __init__(self, timesteps, w, beta_start, beta_end, beta_sche='exp'):
        self.timesteps = timesteps
        self.w = w
        if beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            self.betas = exp_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        a_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        b_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return a_t * x_start + b_t * noise

    def sample(self, model_fw, model_fw_uncond, h, device, steps):
        # one-step sampling: start from pure Gaussian noise at highest level
        x = torch.randn_like(h).to(device)
        t = torch.full((h.shape[0],), self.timesteps-1,
                       device=device, dtype=torch.long)
        cond = model_fw(x, h, t)
        uncond = model_fw_uncond(x, t)
        return (1 + self.w) * cond - self.w * uncond
