import torch
from torch import nn
from tqdm import tqdm

class IMH(nn.Module):
    def __init__(self, target_log_density,d,proposal_distribution=None,number_chains=1):
        super().__init__()
        self.target_log_density = target_log_density
        self.d = d

        if proposal_distribution is None:
            self.proposal_distribution = torch.distributions.MultivariateNormal(torch.zeros(self.d), torch.eye(self.d))
        else:
            self.proposal_distribution = proposal_distribution
        self.number_chains = number_chains

    def independant_metropolis_step(self,x, x_prime):
        target_density_ratio = self.target_log_density(x_prime) - self.target_log_density(x)
        proposal_density_ratio = self.proposal_distribution.log_prob(x) - self.proposal_distribution.log_prob(x_prime)
        acceptance_log_prob = target_density_ratio + proposal_density_ratio
        mask = ((torch.rand(x_prime.shape[0]) < torch.exp(acceptance_log_prob)) * 1.).unsqueeze(-1)
        x = (mask) * x_prime + (1 - (mask)) * x
        return x,mask

    def sample(self, number_steps):
        x = self.proposal_distribution.sample([self.number_chains])
        pbar = tqdm(range(number_steps))
        for t in pbar:
            proposed_x = self.proposal_distribution.sample([self.number_chains])
            x,mask = self.independant_metropolis_step(x,proposed_x)
            pbar.set_postfix_str('acceptance = ' + str(torch.mean(mask * 1.)))
        return x