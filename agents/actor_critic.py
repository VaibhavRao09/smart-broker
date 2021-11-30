from collections import defaultdict, deque

import numpy as np
import torch
from torch import FloatTensor as FT, tensor as T

class A2C:
    def __init__(
        self,
        env,
        actor,
        critic,
        n_actns, 
        actor_optmz, 
        critic_optmz,
        mdl_pth='../models/a2c',
        log_freq=100,
        hyprprms={},
    ):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.n_actns = n_actns
        self.actor_optmz = actor_optmz
        self.critic_optmz = critic_optmz
        self.log_freq = log_freq
        self.mdl_pth = mdl_pth
        self.hyprprms = hyprprms
        self.gamma = self.hyprprms.get('gamma', 0.95),
        self.step_sz = self.hyprprms.get('step_sz', 0.001)
        self.eval_ep = self.hyprprms.get('eval_ep', 50)
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
        
    @staticmethod
    def _normalise(arr):
        mean = arr.mean()
        std = arr.std()
        arr -= mean
        arr /= (std + 1e-5)
        return arr
        
        
    def _get_returns(self, trmnl_state_val, rewards, gamma=1, normalise=True):
        R = trmnl_state_val
        returns = []
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R 
            returns.append(R)
    
        returns = returns[::-1]
        if normalise:
            return self._normalise(torch.cat(returns))
            
        return FT(returns)
    
    def _get_action(self, policy):
        actn = T(policy.sample().item())
        actn_log_prob = policy.log_prob(actn).unsqueeze(0)
        return actn, actn_log_prob
        
    def train(self):
        exp = []
        state = self.env.reset()
        ts = 0
        ep_ended = False
        ep_reward = 0
        ep_loss = 0
        net_worth = 0
        profit = 0
        bal = 0
        units_held = 0
        state = FT(state)

        while not ep_ended:
            policy = self.actor(state)
            actn, actn_log_prob = self._get_action(policy)
            state_val = self.critic(state)

            nxt_state, reward, ep_ended, info = self.env.step(action=actn.item())
            nxt_state = FT(nxt_state)
            exp.append((nxt_state, state_val, T([reward]), actn_log_prob))
            ep_reward += info.get('reward')
            profit += info.get('profit')
            bal += info.get('balance')
            units_held += info.get('units_held')
            net_worth += info.get('net_worth')
            state = nxt_state
            ts += 1

        states, state_vals, rewards, actn_log_probs = zip(*exp)
        actn_log_probs = torch.cat(actn_log_probs)
        state_vals = torch.cat(state_vals)
        trmnl_state_val = self.critic(state).item()
        returns = self._get_returns(trmnl_state_val, rewards).detach()
        
        adv = returns - state_vals
        actn_log_probs = actn_log_probs
        actor_loss = (-1.0 * actn_log_probs * adv.detach()).mean()
        critic_loss = adv.pow(2).mean()
        net_loss = (actor_loss + critic_loss).mean()

        self.actor_optmz.zero_grad()
        self.critic_optmz.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optmz.step()
        self.critic_optmz.step()

        return net_loss.item(), ep_reward, profit, bal, units_held, net_worth, ts

    def evaluate(self, ep=None):
        if not ep:
            ep = self.eval_ep

        for ep_no in range(ep):
            state = self.env.reset()
            state = FT(state)
            ep_ended = False
            ep_reward = 0
            ts = 0

            while not ep_ended and ts < 200:
                policy = self.actor(state)
                actn, actn_log_prob = self._get_action(policy)
                nxt_state, reward, ep_ended, _ = self.env.step(actn.item())
                ep_reward += reward
                state = FT(nxt_state)

            self.eval_logs[ep_no]['reward'] = ep_reward

    def run(self, ep=1000):
        rewards = deque(maxlen=50)
        profits = deque(maxlen=50)
        bals = deque(maxlen=50)
        units_held_l = deque(maxlen=50)
        losses = deque(maxlen=50)
        net_worth_l = deque(maxlen=50)

        for ep_no in range(ep):
            ep_loss, ep_reward, profit, bal, units_held, net_worth, ts = self.train()
            ep_loss = round(ep_loss, 3)
            ep_reward = round(ep_reward, 2)
            avg_p = int(profit/ts)
            avg_b = int(bal/ts)
            avg_u_h = int(units_held/ts)

            losses.append(ep_loss)
            avg_loss = round(np.mean(losses), 2)

            rewards.append(ep_reward)
            avg_reward = round(np.mean(rewards), 2)

            bals.append(avg_b)
            avg_bal = int(np.mean(bals))

            profits.append(avg_p)
            avg_profit = int(np.mean(profits))

            units_held_l.append(avg_u_h)
            avg_units_held = int(np.mean(units_held_l))

            net_worth_l.append(net_worth)
            avg_net_worth = round(np.mean(net_worth_l), 2)

            # save logs for analysis
            rewards.append(ep_reward)
            self.logs[ep_no]['reward'] = ep_reward
            self.logs[ep_no]['r_avg_reward'] = avg_reward
            self.logs[ep_no]['r_avg_loss'] = avg_loss
            self.logs[ep_no]['r_avg_net_worth'] = avg_net_worth
            self.logs[ep_no]['r_avg_profit'] = avg_profit
            self.logs[ep_no]['r_avg_bal'] = avg_bal
            self.logs[ep_no]['r_avg_units_held'] = avg_units_held

            if ep_no % self.log_freq == 0:
                print(f'\nEp: {ep_no} | L: {ep_loss} | R: {ep_reward} | R.Avg.R: {avg_reward} | P: {avg_p} | R.Avg P: {avg_profit} | B: {avg_b} | R.Avg B: {avg_bal} | R.N_Units: {avg_units_held}', end='')
                