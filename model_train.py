import os
import torch
import hamiltorch

from config import Config
import norm_flows
import distribution as dist
import matplotlib.pyplot as plt

conf = Config()

n_samples = conf.batch_size
step_size = 0.3
num_steps_per_sample = 10
hamiltorch.set_random_seed(131)
params_init = torch.zeros(2)

def sample(d):
    params_hmc = hamiltorch.sample(log_prob_func=d.log_prob, params_init=params_init, num_samples=n_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample)
    return params_hmc

def inverse_temperature(epoch):
    return min(1, 0.01 + epoch/conf.epoch_num)

def main():
    d = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

    vae_model = norm_flows.VaeNormalizingFlow(conf.x_dim, conf.hidden_dim, conf.window_size, conf.z_dim, conf.flow_num, conf.flow_type)
    opt = torch.optim.Adam(vae_model.parameters(), lr=1e-3, amsgrad=True)

    losses = []
    for epoch in range(conf.epoch_num):
        samples = [d.sample().unsqueeze(0) for _ in range(conf.batch_size)]
        data = torch.cat(samples, dim=0)
        
        z, log_q_z, log_likelihood = vae_model(data)
        beta = inverse_temperature(epoch)
        loss = vae_model.free_energy(z, log_q_z, log_likelihood, beta)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss)
        print('epoch: %d   loss: %.4f' % (epoch, loss))
        print('log_q(z): %.4f  log_likelihood: %.4f' % (log_q_z.mean(), log_likelihood.mean()))
    
    plot_loss_moment(losses)

def plot_loss_moment(losses):
    _, ax = plt.subplots(figsize=(16, 9), dpi=80)
    ax.plot(losses, 'blue', label='train', linewidth=1)
    ax.set_title('Loss change in training')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.legend(loc='upper right')
    plt.savefig(os.path.join('./output/', 'loss_vae.png'))

if __name__ == "__main__":
    main()