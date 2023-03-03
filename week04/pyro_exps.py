import pyro

def model(tt):
    at = pyro.sample("at", pyro.distributions.Normal(12, 10))
    tu = pyro.sample("tu", pyro.distributions.HalfCauchy(10))
    with pyro.plate("data", len(tt)):
        pyro.sample("tt", pyro.distributions.Normal(at, tu), obs=tt)