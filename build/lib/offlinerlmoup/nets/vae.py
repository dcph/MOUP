import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_action: Union[int, float],
        device: str = "cpu"
    ) -> None:
        super(VAE, self).__init__()
        self.e1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(input_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, output_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        self.to(device=self.device)


    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.relu(self.e1(torch.cat([obs, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)

        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(obs, z)

        return u, mean, std

    def decode(self, obs: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:

        if z is None:
            z = torch.randn((obs.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([obs, z], 1)))
        a = F.relu(self.d2(a))
        # return self.max_action * torch.tanh(self.d3(a))
        return torch.tanh(self.d3(a))

    def decode_multiple(self, obs: torch.Tensor, z: Optional[torch.Tensor] = None, num_decode=10) -> torch.Tensor:
        if z is None:
            z = torch.randn((obs.shape[0], num_decode, self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([obs.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a)), self.d3(a)


class DVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_action: Union[int, float],
        device: str = "cpu",
        fult: bool = True
    ) -> None:
        super(VAE, self).__init__()
        if(fult):
            self.e1 = nn.Linear(input_dim, hidden_dim)
        else:
            self.e1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(input_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, output_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.fult = fult
        self.device = torch.device(device)

        self.to(device=self.device)


    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z = F.relu(self.e1(torch.cat([obs, action], 1)))
        if(self.fult):
            z = F.relu(self.e1(obs))
        else:
            z = F.relu(self.e1(torch.cat([obs, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)

        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(obs, z)

        return u, mean, std

    def decode(self, obs: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:

        if z is None:
            z = torch.randn((obs.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([obs, z], 1)))
        a = F.relu(self.d2(a))
        # return self.max_action * torch.tanh(self.d3(a))
        return torch.tanh(self.d3(a))

    def decode_multiple(self, obs: torch.Tensor, z: Optional[torch.Tensor] = None, num_decode=10) -> torch.Tensor:
        if z is None:
            z = torch.randn((obs.shape[0], num_decode, self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([obs.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a)), self.d3(a)