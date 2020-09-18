# Normalizing Flows

# Description
This repo is the PyTorch implementation of [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770).

# Motivation
The performance of variational inference is decided by how close the approximate posterior q(z|x) is against the true posterior p(z|x). The previous black box methods assume the distribution of q(z) and search for q(z|x) that maximize ELBO, equivalent to minimizing the KL divergence of q(z|x) and p(z|x), in asymptotic manner.\ 

The chosen class of approximate posteriors have prominent impact on the performance. For example, if the chosen class of posteriors are not faithful enough, it can result in poor prediction. Therefore, it’s important to make sure the class of posterior distributions is wide enough to cover the true posterior.\

The SGVB approach assumes the distribution of the latent variable Z, e.g., a Gaussian distribution. And it optimizes the parameters of (μ, σ2) through SGVB process. However, we cannot faithfully guarantee N(μ, σ2) be flexible enough to recover the true posterior.\

This paper proposes to use layers of Gaussian variables from z1 to zk to encode the input variable x. Each layer performs transformation from one distribution to another distribution, which are assembled into normalizing flows. Hopefully, normalizing flows can generate better posterior approximations than other approaches.

# Network structure
The whole network consists of an inference model and a generative model. The inference model is to encode the input vector x into latent variable z0 and further into zk, while the generative model is to reconstruct the random variables. The network can be illustrated with following diagram:\
![Image of Normalizing Flows Network](/image/normal-flows-network.png)

# Dependecies
The code in this repo was implemented and tested with PyTorch 1.5.1 and Python 3.8.3
