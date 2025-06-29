import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Katman ağırlıklarını başlatmak için yardımcı fonksiyon."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = np.array(env.single_observation_space.shape).prod()
        action_shape = np.array(env.single_action_space.shape).prod()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_shape), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class PPOAgent:
    def __init__(self, envs, device, **hyperparams):
        self.envs = envs
        self.device = device
        self.hp = hyperparams
        
        self.obs_shape = envs.single_observation_space.shape
        self.action_shape = envs.single_action_space.shape
        
        self.agent = ActorCritic(envs).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.hp['learning_rate'], eps=1e-5)
        
        self.obs = torch.zeros((self.hp['num_steps'], self.hp['num_envs']) + self.obs_shape).to(device)
        self.actions = torch.zeros((self.hp['num_steps'], self.hp['num_envs']) + self.action_shape).to(device)
        self.logprobs = torch.zeros((self.hp['num_steps'], self.hp['num_envs'])).to(device)
        self.rewards = torch.zeros((self.hp['num_steps'], self.hp['num_envs'])).to(device)
        self.dones = torch.zeros((self.hp['num_steps'], self.hp['num_envs'])).to(device)
        self.values = torch.zeros((self.hp['num_steps'], self.hp['num_envs'])).to(device)
        
    def update(self):
        with torch.no_grad():
            next_obs, _ = self.envs.reset()
            next_obs = torch.Tensor(next_obs).to(self.device)
            next_done = torch.zeros(self.hp['num_envs']).to(self.device)
            
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            last_gae_lam = 0
            for t in reversed(range(self.hp['num_steps'])):
                if t == self.hp['num_steps'] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.hp['gamma'] * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = last_gae_lam = delta + self.hp['gamma'] * self.hp['gae_lambda'] * nextnonterminal * last_gae_lam
            returns = advantages + self.values

        b_obs = self.obs.reshape((-1,) + self.obs_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        # PPO update loop
        b_inds = np.arange(self.hp['batch_size'])
        clipfracs = []
        for epoch in range(self.hp['update_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, self.hp['batch_size'], self.hp['minibatch_size']):
                end = start + self.hp['minibatch_size']
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # watch clip
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.hp['clip_coef']).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss (Clipped Surrogate Objective)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.hp['clip_coef'], 1 + self.hp['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - self.hp['ent_coef'] * entropy_loss + v_loss * self.hp['vf_coef']

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.hp['max_grad_norm'])
                self.optimizer.step()

        return v_loss.item(), pg_loss.item(), entropy_loss.item()