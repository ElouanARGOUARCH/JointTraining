import torch
from torch import nn
import math


class ConditionalDensityEstimationTarget(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, theta):
        raise NotImplementedError

    def log_prob(self, x, theta):
        raise NotImplementedError

class DeformedCircles(ConditionalDensityEstimationTarget):
    def __init__(self):
        super().__init__()

    def sample(self, theta, means=torch.tensor([1., 2.]), weights=torch.tensor([.5, .5]), noise=0.125):
        angle = torch.rand(theta.shape[0]) * 2 * math.pi
        cat = torch.distributions.Categorical(weights).sample([theta.shape[0]])
        x, y = means[cat] * torch.cos(angle) + torch.randn_like(angle) * noise, means[cat] * torch.sin(
            angle) + torch.randn_like(angle) * noise
        return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)*theta

    def log_prob(self, samples,theta, means=torch.tensor([1., 2.]), weights=torch.tensor([.5, .5]), noise=0.125):
        r = torch.norm(samples/theta, dim=-1).unsqueeze(-1)
        cat = torch.distributions.Categorical(weights)
        mvn = torch.distributions.MultivariateNormal(means.unsqueeze(-1),
                                                     torch.eye(1).unsqueeze(0).repeat(2, 1, 1) * noise)
        mixt = torch.distributions.MixtureSameFamily(cat, mvn)
        return mixt.log_prob(r)
