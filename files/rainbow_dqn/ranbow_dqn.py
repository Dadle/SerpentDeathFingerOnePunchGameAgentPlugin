import math
import torch
from torch import nn
from torch.nn import functional as F



class DQN(nn.Module):
    def __init__(self, args, action_space):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space

        self.conv1 = nn.Conv3d(args.history_length, 32, kernel_size=(9,9,3), stride=5, padding=(2,2,0))
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc_h_v = NoisyLinear(29400, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(29400, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        #x = x.reshape(1,3,139,139,4)
        print("x shape:", x.shape)
        print("0", x.shape)
        x = F.relu(self.conv1(x))
        print('1', x.shape)
        x = x.view(-1,32,35,35)
        print("2", x.shape)
        x = F.relu(self.conv2(x))
        print("3", x.shape)
        x = F.relu(self.conv3(x))
        print("4", x.shape)
        x = x.view(-1, 29400)  #
        print("view", x.shape)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        print("value", x.shape)
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        print("advantage", x.shape)
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        print("q1", x.shape)
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        print("Q:",q.shape)
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
