import torch
#torch.manual_seed(123)
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal
import math

class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        assert x.shape[-1] % self.pool_size == 0, 'Maxout dimension error: The last dimension of {} should be dividable by {}'.format(x.shape[-1], self.pool_size)
        m, _ = x.view(*x.shape[:-1], x.shape[-1] // self.pool_size, self.pool_size).max(-1)
        return m

class InferenceNet(nn.Module):
    def __init__(self, x_dim, hidden_dim, window_size, z_dim):
        super().__init__()

        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.z_dim = z_dim

        self.hidden = nn.Linear(x_dim, hidden_dim)
        self.maxout = Maxout(self.window_size)
        self.miu_z = nn.Linear(hidden_dim // window_size, z_dim)
        self.log_var_z = nn.Linear(hidden_dim // window_size, z_dim)
    
    def forward(self, x):
        # h shape: [batch_size, hidden_dim]
        h = self.hidden(x)
        # h_max shape: [batch_size, hidden_dim//window_size]
        h_max = self.maxout(h)
        # miu and log_var shape: [batch_size, z_dim]
        miu = self.miu_z(h_max)
        log_var = self.log_var_z(h_max)
        sigma = torch.exp(0.5 * log_var)
        z_dist = Normal(miu, sigma)

        # generating z0 through re-parameterization
        z0 = miu + torch.randn(1) * sigma
        # log_q_z0 shape: [batch_size, z_dim]
        log_q_z0 = z_dist.log_prob(z0)

        return z0, log_q_z0

class NormalizingFlows(nn.Module):
    def __init__(self, z_dim, flow_num, flow_type='planar'):
        super().__init__()
        assert flow_num > 0 and isinstance(flow_num, int), 'Argument flow_num should be positive integer'
        self.flow_num = flow_num

        assert flow_type == 'planar' or flow_type == 'radial', 'Flow type can only be either planar or radial'
        if flow_type == 'planar':
            self.flows = nn.ModuleList([PlanarFlow(z_dim) for _ in range(flow_num)])
        else:
            raise NotImplementedError('Radial flow not implemented yet')
    
    def forward(self, z0, log_q_z0):
        z = z0
        log_q_z = log_q_z0
        for i in range(self.flow_num):
            z, log_q_z = self.flows[i](z, log_q_z)
        
        return z, log_q_z

class BaseFlow(nn.Module):
    def __init__(self):
        super().__init__()

    def adjust_scale(self):
        pass

    def forward(self, z, log_q_z):
        pass


class PlanarFlow(BaseFlow):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim

        self.weight = Parameter(torch.Tensor(1, z_dim))
        self.bias = Parameter(torch.Tensor(1))
        self.scale = Parameter(torch.Tensor(1, z_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.scale)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def reset_param(self):
        self.weight = Parameter(torch.Tensor([[3.0, 0.]]))
        self.scale = Parameter(torch.Tensor([[2.0, 0.]]))
        self.bias = Parameter(torch.Tensor([0.]))

    def adjust_scale(self):
        v_w = torch.squeeze(self.weight, 0)
        v_u = torch.squeeze(self.scale, 0)
        s = torch.dot(v_w, v_u)
        m_s = -1 + torch.log(1 + torch.exp(s))
        return self.scale + (m_s - s) * self.weight / torch.norm(self.weight)

    def forward(self, z, log_q_z):
        
        # z shape: [batch_size, z_dim]
        # log_q_z shape: [batch_size, z_dim]
        # h shape: [batch_size, 1]
        h = F.linear(z, self.weight, self.bias)
        # u_hat shape: [1, z_dim]
        u_hat = self.adjust_scale()
        z_n = z + torch.mm(torch.tanh(h), u_hat)

        # derivative of tanh(x) is 1 - tanh(x)**2
        # psi shape: [batch_size, z_dim]
        psi = torch.mm((1 - torch.tanh(h) ** 2), self.weight)

        # log_det shape: [batch_size, 1]
        log_det = torch.log(torch.abs(1 + torch.mm(psi, torch.t(u_hat))))
        return z_n, log_q_z - log_det

class GenerativeNet(nn.Module):
    def __init__(self, z_dim, hidden_dim, window_size, x_dim):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.x_dim = x_dim

        self.hidden = nn.Linear(z_dim, hidden_dim)
        self.maxout = Maxout(window_size)
        self.miu_x = nn.Linear(hidden_dim // window_size, x_dim)
        self.log_var_x = nn.Linear(hidden_dim // window_size, x_dim)
    
    def forward(self, z):
        # h shape: [batch_size, hidden_dim]
        h = self.hidden(z)
        # h_max shape: [batch_size, hidden_dim//window_size]
        h_max = self.maxout(h)
        # miu, log_var shape: [batch_size, x_dim]
        miu = self.miu_x(h_max)
        log_var = self.log_var_x(h_max)
        sigma = torch.exp(0.5 * log_var)
        x_dist = Normal(miu, sigma)

        # generating x_hat through re-parameterization
        x_hat = miu + torch.randn(1) * sigma
        # log_p_x_z shape: [batch_size, x_dim] 
        log_p_x_z = x_dist.log_prob(x_hat)
        return x_hat, log_p_x_z

class VaeNormalizingFlow(nn.Module):
    def __init__(self, x_dim, hidden_dim, window_size, z_dim, flow_num, flow_type):
        super().__init__()
        self.z_prior = MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))

        self.inference_net = InferenceNet(x_dim, hidden_dim, window_size, z_dim)
        self.normalizing_flows = NormalizingFlows(z_dim, flow_num, flow_type)
        self.generative_net = GenerativeNet(z_dim, hidden_dim, window_size, x_dim)
    
    def forward(self, x):
        z0, log_q_z0 = self.inference_net(x)
        z, log_q_z = self.normalizing_flows(z0, log_q_z0)
        x_hat, log_likelihood = self.generative_net(z)
        return z, log_q_z, log_likelihood
    
    def free_energy(self, z, log_q_z, log_likelihood, beta):
        # z_prior log prob shape: [batch_size]
        log_p_x_z = log_likelihood + self.z_prior.log_prob(z).unsqueeze(1)
        return (log_q_z - beta * log_p_x_z).mean()
    