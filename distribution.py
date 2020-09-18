import torch
import numpy as np

class Distribution():
    def __init__(self):
        self.w1 = lambda z: torch.sin(2 * np.pi * z[:, 0] / 4)
        self.w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
        self.w3 = lambda z: 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)

    def log_prob(self):
        raise NotImplementedError('log_prob method not implemented for Distribution')

class DualMoon(Distribution):
    def __init__(self):
        super().__init__()
        self.log_pdf = lambda z: -0.5 * (((torch.norm(z, dim=1) - 2) / 0.4) ** 2) + torch.log(torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2))
    
    def log_prob(self, z):
        return self.log_pdf(z)

class Sinunoid(Distribution):
    def __init__(self):
        super().__init__()
        self.log_pdf = lambda z: -0.5 * torch.pow((z[:, 1] - self.w1(z)) / 0.4, 2)
    
    def log_prob(self, z):
        return self.log_pdf(z)

class SinunoidHole(Distribution):
    def __init__(self):
        super().__init__()
        self.log_pdf = lambda z: torch.log(torch.exp(-0.5 * ((z[:, 1] - self.w1(z)) / 0.35) ** 2) + torch.exp(-0.5 * (( z[:, 1] - self.w1(z) + self.w2(z)) / 0.35) ** 2))
    
    def log_prob(self, z):
        return self.log_pdf(z)

class SinunoidForked(Distribution):
    def __init__(self):
        super().__init__()
        self.log_pdf = lambda z: torch.log(torch.exp(-0.5 * ((z[:, 1] - self.w1(z)) / 0.4) ** 2) + torch.exp(-0.5 * ((z[:, 1] - self.w1(z) + self.w3(z)) / 0.35) ** 2))
    
    def log_prob(self, z):
        return self.log_pdf(z)