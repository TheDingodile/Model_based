import pyro
import torch
from pyro.infer import NUTS
from pyro.infer import MCMC

def model(tt):
    at = pyro.sample("at", pyro.distributions.Normal(12, 10))
    tu = pyro.sample("tu", pyro.distributions.HalfCauchy(10))
    with pyro.plate("data", len(tt)):
        pyro.sample("tt", pyro.distributions.Normal(at, tu), obs=tt)

# Input data for model - must be a PyTorch tensor!
tt_obs = torch.tensor([13,17,16,32,12,13,28,12,14,18,36,16,16,31])
# Run inference in Pyro
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500, num_chains=1)
mcmc.run(tt_obs)
# Show summary of inference results
mcmc.summary()