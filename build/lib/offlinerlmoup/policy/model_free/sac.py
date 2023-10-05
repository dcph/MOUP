import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from typing import Dict, Union, Tuple
from offlinerlmoup.policy import BasePolicy
from torch.nn import functional as F
from offlinerlmoup.dynamics import BaseDynamics

deta = 10000
class SACPolicy(BasePolicy):

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        dynamics: BaseDynamics,
        behavior_policy: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        behavior_policy_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        num_samples_mmd_match: int = 4
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.behavior_policy = behavior_policy

        self.actor_optim = actor_optim
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim
        self.behavior_policy_optim = behavior_policy_optim

        self._tau = tau
        self._gamma = gamma
        self.num_samples_mmd_match = num_samples_mmd_match
        self.dynamics = dynamics

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def actforward_multy(
        self,
        obs: torch.Tensor,
        nums: int,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        res = []
        for i in range(nums):
            squashed_action, raw_action = dist.rsample()
            res.append(raw_action)
        return torch.stack(res)

    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actforward(obs, deterministic)
        return action.cpu().numpy()

    def mmd_loss_laplacian(self, samples1, samples2, sigma=20):
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1) 
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def learn(self, real_batch, fake_batch, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        # global deta
        # deta += 1
        # real_obs = real_batch["observations"]
        # real_act = real_batch["actions"]
        # real_next_obs = real_batch["next_observations"]
        # fake_obs = fake_batch["observations"]
        # fake_act = fake_batch["actions"]
        # fake_next_obs = fake_batch["next_observations"]
        # real_obs_act = torch.cat((real_obs, real_act), axis=-1)
        # fack_obs_act = torch.cat((fake_obs, fake_act), axis=-1)
        # ksi = torch.zeros(real_obs.shape).to(self.actor.device)
        # nn.init.normal_(tensor=ksi, mean=0.0, std=0.1)

        # recon, mean, std = self.behavior_policy(real_obs_act, real_next_obs)
        # recon_loss = F.mse_loss(recon, real_next_obs)
        # kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        # vae_loss = recon_loss + 0.5 * kl_loss
        # self.behavior_policy_optim.zero_grad()
        # vae_loss.backward()
        # self.behavior_policy_optim.step()

        # with torch.no_grad():
        #     _, fack_raw_sampled_obs = self.behavior_policy.decode_multiple(fack_obs_act, num_decode=self.num_samples_mmd_match)
        #     fake_raw_next_obs = fake_next_obs.unsqueeze(1).repeat(1,4,1)
        #     mmd_loss = self.mmd_loss_laplacian(fake_raw_next_obs, fack_raw_sampled_obs)
        #     test_mmd = (mmd_loss-0.07).mean()

        #     actor_fake_action, _ = self.actforward(fake_obs)
        #     actor_fack_obs_act = torch.cat((fake_obs, actor_fake_action), axis=-1)
        #     _, actor_fack_sampled_obs = self.behavior_policy.decode_multiple(actor_fack_obs_act, num_decode=2)
        #     real_o = [real_obs]
        #     real_o.append(real_next_obs)
        #     real_raw_obs = torch.stack(real_o, dim = 1)
        #     actor_mmd_loss = self.mmd_loss_laplacian(actor_fack_sampled_obs, real_raw_obs)
        #     actor_test_mmd = (actor_mmd_loss-0.07).mean()

        #     loss_deta = 10000/deta
        #     mmd_q_loss = loss_deta *test_mmd * actor_test_mmd
        global deta
        deta += 1
        real_obs = real_batch["observations"]
        real_next_obs = real_batch["next_observations"]
        fake_obs = fake_batch["observations"]
        ksi = torch.zeros(real_obs.shape).to(self.actor.device)
        nn.init.normal_(tensor=ksi, mean=0.0, std=0.1)

        recon, mean, std = self.behavior_policy(real_obs, real_next_obs)
        recon_loss = F.mse_loss(recon, real_next_obs)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss
        self.behavior_policy_optim.zero_grad()
        vae_loss.backward()
        self.behavior_policy_optim.step()

        with torch.no_grad():
            _, fack_raw_sampled_obs = self.behavior_policy.decode_multiple(fake_obs, num_decode=self.num_samples_mmd_match)
            actor_fake_action, _ = self.actforward(fake_obs)
            fake_obs_cpu = fake_obs.cpu().numpy()
            actor_fake_action_cpu = actor_fake_action.cpu().numpy()
            fake_next_obs, _, _, _= self.dynamics.step(fake_obs_cpu, actor_fake_action_cpu)
            fake_raw_next_obs = torch.tensor(fake_next_obs).unsqueeze(1).repeat(1,4,1).to(self.actor.device)
            mmd_loss = self.mmd_loss_laplacian(fake_raw_next_obs, fack_raw_sampled_obs)
            test_mmd = (mmd_loss-0.07).mean()
            loss_deta = 10000/deta
            mmd_q_loss = test_mmd

        # global deta
        # deta += 1
        # real_obs = real_batch["observations"]
        # real_next_obs = real_batch["next_observations"]
        # fake_obs = fake_batch["observations"]
        # ksi = torch.zeros(real_obs.shape).to(self.actor.device)
        # nn.init.normal_(tensor=ksi, mean=0.0, std=0.1)

        # # recon, mean, std = self.behavior_policy(real_obs, real_next_obs)
        # # recon_loss = F.mse_loss(recon, real_next_obs)
        # # kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        # # vae_loss = recon_loss + 0.5 * kl_loss
        # # self.behavior_policy_optim.zero_grad()
        # # vae_loss.backward()
        # # self.behavior_policy_optim.step()

        # with torch.no_grad():
        #     # _, fack_raw_sampled_obs = self.behavior_policy.decode_multiple(fake_obs, num_decode=self.num_samples_mmd_match)
        #     actor_fake_action, _ = self.actforward(fake_obs)
        #     fake_obs_cpu = fake_obs.cpu().numpy()
        #     actor_fake_action_cpu = actor_fake_action.cpu().numpy()
        #     fake_next_obs, _, _, _= self.dynamics.step(fake_obs_cpu, actor_fake_action_cpu)
        #     fake_raw_next_obs = torch.tensor(fake_next_obs).unsqueeze(1).repeat(1,2,1).to(self.actor.device)
        #     real_o = [real_obs]
        #     real_o.append(real_next_obs)
        #     real_raw_obs = torch.stack(real_o, dim = 1)
        #     mmd_loss = self.mmd_loss_laplacian(fake_raw_next_obs, real_raw_obs)
        #     test_mmd = (mmd_loss-0.07).mean()
        #     loss_deta = 10000/deta
        #     mmd_q_loss = test_mmd
        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            next_q = next_q #- mmd_q_loss
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean() #+ mmd_q_loss
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()# + mmd_q_loss
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()


        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean() + 10*mmd_q_loss#- loss_deta * actor_test_mmd
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/mmd": mmd_q_loss.item(),
            # "loss/actor_mmd": actor_test_mmd.item(),
            "loss/fack_mmd": test_mmd.item(),
            # "loss/vae_loss": vae_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result

