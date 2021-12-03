import random
from gym import Env as OpenAIEnv
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_INT = 2147483647
MAX_STEPS = 20000


class Actions:
    Buy = 0
    Sell = 1
    Hold = 2
    N = 3


class SmartBrokerEnv(OpenAIEnv):
    def __init__(
        self,
        df_info,
        portfolio,
        batch_dur=30,
        n_actions=Actions.N,
        data_dir='../data',
    ):
        self.df_info = df_info
        self.reward_range = (0, MAX_INT)
        self.batch_dur = batch_dur
        self.df_info = df_info
        self.portfolio = portfolio
        self.n_actions = n_actions
        self.curr_step = 0
        self.data_dir = data_dir
        self._init_portfolio(load_df=True)
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([n_actions, 1]),
            dtype=np.uint8,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.batch_dur*3 + 3, 1),
            dtype=np.uint8,
        )

    def _init_portfolio(self, load_df=False):
        self.source = self.portfolio.get('source', 'Bitstamp')
        self.init_balance = self.portfolio.get('init_balance', 100)
        self.balance = self.init_balance
        self.entity = self.portfolio.get('entity', 'XRP')
        self.year = self.portfolio.get('year', '2021')
        self.market = self.portfolio.get('market', 'USD')
        self.duration_typ = self.portfolio.get('duration_typ', 'minute')
        self.price_typ = self.portfolio.get('price_typ', 'close')
        self.roll_period = self.portfolio.get('roll_period', 30)
        self.units_held = 0
        self.net_worth = self.balance

        if load_df:
            file = f'{self.source}_{self.entity}{self.market}_{self.year}_{self.duration_typ}.csv'
            self.df = pd.read_csv(f'{self.data_dir}/{file}', skiprows=1)
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df.sort_values(by='date', inplace=True, ascending=True)
            self.df.reset_index(inplace=True)
            self.df = self.df[self.df_info.get('cols')]
            # process and initialise dataframe
            self._process_df()

        self.max_step = self.df.shape[0]

    def _process_df(self):
        start_dt = self.df_info.get('start_date')
        end_dt = self.df_info.get('end_date')
        norm_cols = self.df_info.get('norm_cols')

        # filter based on range
        self.df = self.df.loc[(self.df['date'] >= start_dt) & (self.df['date'] <= end_dt)]
        self.df[norm_cols] = self.df[norm_cols].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()),
        )
        self.df['rolling_price'] = self.df[self.price_typ].rolling(self.roll_period).sum()

    def _get_ptfo_ftrs(self):
        # normalise features
        return np.array([self.balance/MAX_INT, self.units_held/MAX_INT, self.net_worth/MAX_INT])

    def _get_obs_between(self, start_dt, end_dt):
        mask = (self.df['date'] >= start_dt) & (self.df['date'] <= end_dt)

        prices = self.df.loc[mask, self.price_typ].values 
        roll_prices = self.df.loc[mask, 'roll_prices'].values
        volumes = self.df.loc[mask, f'Volume {self.entity}'].values
        ptfo_ftrs = self._get_ptfo_ftrs()

        obs = np.concatenate(
            (
                prices,
                roll_prices,
                volumes,
                ptfo_ftrs,
            )
        )

        return obs

    def _get_obs(self):
        prices = self.df.iloc[
            self.curr_step: self.curr_step + self.batch_dur
        ][self.price_typ].values 

        roll_prices = self.df.iloc[
            self.curr_step: self.curr_step + self.batch_dur
        ]['rolling_price'].values 

        volumes = self.df.iloc[
            self.curr_step: self.curr_step + self.batch_dur
        ][f'Volume {self.entity}'].values 

        ptfo_ftrs = self._get_ptfo_ftrs()

        obs = np.concatenate(
            (
                prices,
                roll_prices,
                volumes,
                ptfo_ftrs,
            )
        )

        return obs

    def _act(self, action):
        # default to selling or buying all the stocks
        if isinstance(action, list) or isinstance(action, np.ndarray):
            action_type = abs(action[0])
            amount = abs(action[1])
        else:
            action_type = action
            amount = 1

        curr_price = self.df.iloc[self.curr_step][self.price_typ]
        units_bought = 0
        units_sold = 0
        alpha = self.curr_step / MAX_STEPS

        if action_type < 1:
            action_type = int(action_type * self.action_space.high[0])

        if action_type == Actions.Buy:
            total_possible = int(self.balance / curr_price)
            units_bought = int(total_possible * amount)
            cost = units_bought * curr_price
            self.balance -= cost
            self.units_held += units_bought
        elif action_type == Actions.Sell:
            units_sold = self.units_held * amount
            self.balance += units_sold * curr_price
            self.units_held -= units_sold

        self.net_worth = self.balance + self.units_held * curr_price

        if action_type == Actions.Buy and total_possible == 0:
            reward = -10
        elif action_type == Actions.Sell and units_sold == 0:
            reward = -10
        elif action_type == Actions.Hold:
            reward = -5
        else:
            reward = self.net_worth * alpha

        info = {
            'amount': amount,
            'reward': reward,
            'curr_price': curr_price,
            'curr_step': self.curr_step,
            'units_bought': units_bought,
            'units_sold': units_sold,
            'balance': self.balance,
            'net_worth': self.net_worth,
            'units_held': self.units_held,
            'profit': self.net_worth - self.init_balance,
        }

        return info

    def reset(self, idx=None):
        if idx is None:
            idx = self.roll_period
        self._init_portfolio()
        self.curr_step = idx
        obs = self._get_obs()
        return obs

    def step(self, action):
        info = self._act(action)
        self.curr_step += 1
        reward = info['reward']
        done = self.net_worth <= 0 or self.curr_step == MAX_STEPS
        obs = self._get_obs()

        if self.curr_step > self.df.shape[0] - self.batch_dur:
            self.curr_step = random.randint(self.roll_period, self.df.shape[0] - self.batch_dur)

        return obs, reward, done, info

    def render(self, *args):
        buy_steps, buy_prices, sell_steps, sell_prices, start_step, end_step = args
        start_step = max(self.roll_period, start_step-5)
        end_step = min(self.df.shape[0], end_step+5)
        df = self.df.loc[start_step:end_step]
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.plot(df['date'], df['close'], color='black', label='XRP')
        ax.scatter(df.loc[buy_steps, 'date'].values, buy_prices, c='green', alpha=0.5, label='buy')
        ax.scatter(df.loc[sell_steps, 'date'].values, sell_prices, c='red', alpha=0.5, label='sell')
        ax.legend()
        ax.grid()
        plt.xticks(rotation=45)
        plt.show()

    def close(self):
        print('close')