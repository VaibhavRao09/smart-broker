from collections import defaultdict
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import tensor as T
from torch.optim import Adam

from networks.sac.continuous.policy_net import PolicyNetwork
from networks.sac.continuous.q_net import QNetwork
from networks.sac.continuous.value_net import ValueNetwork
from helpers.replay_buffer import ReplayBuffer


DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

class SAC:
    def __init__(
        self,
        env,
        name,
        input_dim,
        networks={},
        optmzrs={},
        log_freq=10,
        hyprprms={},
        save_mdls=True,
        load_mdls=False,
    ):
        self.env = env
        self.env_name = name
        self.input_dim = input_dim
        self.action_space = env.action_space
        self.hyprprms = hyprprms
        self.eps = self.hyprprms.get('eps', 1e-6)
        self.lr = self.hyprprms.get('lr', 0.0003)
        self.gamma = self.hyprprms.get('gamma', 0.95)
        self.eval_ep = self.hyprprms.get('eval_ep', 50)
        self.mem_sz = self.hyprprms.get('mem_sz', 5000)
        self.critic_sync_f = self.hyprprms.get('critic_sync_f', 5)
        self.tau = self.hyprprms.get('tau', 0.005)
        self.save_mdls = save_mdls
        self.load_mdls = load_mdls
        self.memory = ReplayBuffer(self.mem_sz)
        
        # policy network
        self.policy = networks.get(
            'policy_net',
            PolicyNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.shape[0],
                eps=self.eps,
                max_act=env.action_space.high,
            ),
        )
        self.policy.to(DEVICE)
        self.policy_optmz = optmzrs.get(
            'policy_optmz',
            Adam(self.policy.parameters(), lr=self.lr),
        )
        
        # value network
        self.value = networks.get(
            'value_net',
            ValueNetwork(state_dim=input_dim),
        )
        self.value.to(DEVICE)
        self.tgt_value = networks.get(
            'target_value_net',
            ValueNetwork(state_dim=input_dim),
        )
        self.tgt_value.to(DEVICE)
        self.value_optmz = optmzrs.get(
            'value_optmz',
            Adam(self.value.parameters(), lr=self.lr),
        )
        
        # critic network
        self.critic_a = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.shape[0],
            )
        )
        self.critic_a.to(DEVICE)
        self.critic_a_optmz = optmzrs.get(
            'critic_a_optmz',
            Adam(self.critic_a.parameters(), lr=self.lr),
        )
        # target critic network
        self.critic_b = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.shape[0],
            )
        )
        self.critic_b.to(DEVICE)
        self.critic_b_optmz = optmzrs.get(
            'critic_b_optmz',
            Adam(self.critic_b.parameters(), lr=self.lr),
        )

        self.log_freq = log_freq
        self.logs = defaultdict(
            lambda: {
                'reward': 0,
                'avg_reward': 0,
            },
        )
        self.eval_logs = defaultdict(
            lambda: {
                'reward': 0,
                'avg_reward': 0,
            },
        )

    def _get_action(self, state):
        state = torch.cat([state])
        actions, _ = self.policy.sample(state, add_noise=False)
        return actions.cpu().detach().numpy()

    def _sync_weights(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_weights = dict(self.tgt_value.named_parameters())
        value_weights = dict(self.value.named_parameters())

        for name in value_weights:
            value_weights[name] = tau * value_weights[name].clone() + \
                    (1-tau) * target_value_weights[name].clone()

        self.tgt_value.load_state_dict(value_weights)

    def _save_models(
        self,
        policy_path=None,
        critic_b_path=None,
        critic_a_path=None,
        tgt_value_path=None,
        value_path=None,
    ):
        path = Path(f'../models/{self.env_name}/')
        if not path.exists():
            path.mkdir()

        if policy_path is None:
            policy_path = path/'actor'

        if critic_a_path is None:
            critic_a_path = path/'critic_a'

        if critic_b_path is None:
            critic_b_path = path/'critic_b'

        if value_path is None:
            value_path = path/'value'

        if tgt_value_path is None:
            tgt_value_path = path/'target_value'

        self.policy.save(policy_path)
        self.critic_a.save(critic_a_path)
        self.critic_b.save(critic_b_path)
        self.value.save(value_path)
        self.tgt_value.save(tgt_value_path)

    def _load_models(
        self,
        policy_path=None,
        critic_b_path=None,
        critic_a_path=None,
        tgt_value_path=None,
        value_path=None,
    ):
        print('loading models....')
        path = Path(f'../models/{self.env_name}/')

        if policy_path is not None:
            policy_path = path/'actor'
            self.policy.load(policy_path)

        if critic_a_path is not None:
            critic_a_path = path/'critic_a'
            self.critic_a.load(critic_a_path)

        if critic_b_path is not None:
            critic_b_path = path/'critic_b'
            self.critic_b.load(critic_b_path)

        if value_path is not None:
            value_path = path/'value'
            self.value.load(value_path)

        if tgt_value_path is not None:
            tgt_value_path = path/'target_value'
            self.tgt_value.load(tgt_value_path)

    def _train_value_net(self, states):
        pred_values = self.value(states).view(-1)
        actions, log_probs = self.policy.sample(states, add_noise=False)
        log_probs = log_probs.view(-1)

        pred_q1_vals = self.critic_a.forward(states, actions)
        pred_q2_vals = self.critic_b.forward(states, actions)
        pred_q_values = torch.min(pred_q1_vals, pred_q2_vals).view(-1)

        self.value_optmz.zero_grad()
        target_values = pred_q_values - log_probs
        val_loss = 0.5 * F.mse_loss(pred_values, target_values.detach())
        val_loss.backward(retain_graph=True)
        self.value_optmz.step()

        return val_loss.cpu().detach().numpy()

    def _train_critic_net(self, states, actions, rewards, nxt_states, dones):
        self.critic_a_optmz.zero_grad()
        self.critic_b_optmz.zero_grad()

        target_values = self.tgt_value(nxt_states).view(-1)
        target_q_values = rewards + (1 - dones) * self.gamma * target_values
        pred_q1_values = self.critic_a.forward(states, actions).view(-1)
        pred_q2_values = self.critic_b.forward(states, actions).view(-1)

        critic_a_loss = 0.5 * F.mse_loss(pred_q1_values, target_q_values.detach())
        critic_b_loss = 0.5 * F.mse_loss(pred_q2_values, target_q_values.detach())
        critic_loss = critic_a_loss + critic_b_loss

        critic_loss.backward()
        self.critic_a_optmz.step()
        self.critic_b_optmz.step()

        return critic_loss.cpu().detach().numpy()

    def _train_policy_net(self, states):
        actions, log_probs = self.policy.sample(states, add_noise=True)
        log_probs = log_probs.view(-1)
        pred_q1_values = self.critic_a.forward(states, actions)
        pred_q2_values = self.critic_b.forward(states, actions)
        pred_q_value = torch.min(pred_q1_values, pred_q2_values).view(-1)
        policy_loss = (log_probs - pred_q_value).mean()

        self.policy_optmz.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optmz.step()

        return policy_loss.cpu().detach().numpy()

    def train(self, ep_no):
        states, actions, rewards, nxt_states, dones = \
            self.memory.sample(self.mem_sz)

        rewards = T(rewards, dtype=torch.float, device=DEVICE)
        dones = T(dones, dtype=torch.float, device=DEVICE)
        nxt_states = T(nxt_states, dtype=torch.float, device=DEVICE)
        states = T(states, dtype=torch.float, device=DEVICE)
        actions = T(actions, dtype=torch.float, device=DEVICE)

        value_loss = self._train_value_net(states)
        critic_loss = self._train_critic_net(
            states,
            actions,
            rewards,
            nxt_states,
            dones,
        )
        policy_loss = self._train_policy_net(states)

        if ep_no % self.critic_sync_f:
            self._sync_weights()

        return value_loss, critic_loss, policy_loss

    def evaluate(self, ep=None):
        if not ep:
            ep = self.eval_ep

        for ep_no in range(ep):
            state = self.env.reset()
            state = T(state, device=DEVICE)
            ep_ended = False
            ep_reward = 0
            ts = 0

            while not ep_ended and ts < 600:
                action = self._get_action(state)
                nxt_state, reward, ep_ended, _ = self.env.step(action)
                ep_reward += reward
                nxt_state = T(nxt_state, device=DEVICE)
                ts += 1

            self.eval_logs[ep_no]['reward'] = ep_reward

    def run(self, ep=1000):
        print('collecting experience...')
        rewards = []

        if self.load_mdls:
            self._load_models(
                policy_path=f'models/{env_name}/policy',
                critic_a_path=f'models/{env_name}/critic_a',
                critic_b_path=f'models/{env_name}/critic_b',
                value_path=f'models/{env_name}/value',
                tgt_value_path=f'models/{env_name}/tgt_value',
            )

        for ep_no in range(ep):
            state = self.env.reset()
            state = T(state, device=DEVICE)
            ep_ended = False
            ep_reward = 0
            v_loss, c_loss, p_loss = 0, 0, 0
            ts = 0

            while not ep_ended and ts < 200:
                action = self._get_action(state)
                nxt_state, reward, ep_ended, _ = self.env.step(action)
                ep_reward += reward
                action = T(action, device=DEVICE)
                reward = T(reward, device=DEVICE)
                nxt_state = T(nxt_state, device=DEVICE)
                ep_ended = T(ep_ended, device=DEVICE)
                self.memory.add((state, action, reward, nxt_state, ep_ended))
                state = nxt_state
                if self.memory.curr_size > self.mem_sz:
                    v_loss, c_loss, p_loss = self.train(ep_no)

                    if ep_no % 100:
                        self._save_models()
                ts += 1

            rewards.append(ep_reward)
            avg_reward = np.mean(rewards[-50:])
            self.logs[ep_no]['reward'] = ep_reward
            self.logs[ep_no]['avg_reward'] = avg_reward

            if ep_no % self.log_freq == 0:
                if self.memory.curr_size > self.mem_sz:
                    print(f'Episode: {ep_no}, Reward: {ep_reward}, Avg. Reward: {avg_reward}, Policy Loss={round(float(p_loss), 2)}')
                else:
                    print(ep_no, end='..')




