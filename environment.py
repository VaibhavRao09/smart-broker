import random
from gym import Env as OpenAIEnv
from gym import spaces
import numpy as np
import pandas as pd

MAX_INT = 2147483647
MAX_STEPS = 500


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
        self.roll_period = self.portfolio.get('roll_period', 15)
        self.units_held = 0
        self.net_worth = self.balance

        if load_df:
            file = f'{self.source}_{self.entity}{self.market}_{self.year}_{self.duration_typ}.csv'
            df = pd.read_csv(f'{self.data_dir}/{file}', skiprows=1, parse_dates=True)
            df = df[self.df_info.get('cols')]
            # process and initialise dataframe
            self._process_df(df)

    def _process_df(self, df):
        start_dt = self.df_info.get('start_date')
        end_dt = self.df_info.get('end_date')
        norm_cols = self.df_info.get('norm_cols')
    
        # filter based on range
        self.df = df.loc[(df['date'] > start_dt) & (df['date'] <= end_dt)]
        self.df.reset_index()
        self.df[norm_cols] = self.df[norm_cols].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()),
        )
        self.df['rolling_price'] = self.df[self.price_typ].rolling(self.roll_period).sum()
        
        self.df.sort_values('date', inplace=True)
        
    
    def _get_ptfo_ftrs(self):
        # normalise features
        return np.array([self.balance/MAX_INT, self.units_held/MAX_INT, self.net_worth/MAX_INT])
                          
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
        if isinstance(action, list):
            action_type = action[0]
            amount = action[1]
        else:
            action_type = action
            amount = 1
            
        curr_price = self.df.iloc[self.curr_step][self.price_typ]
        units_bought = 0
        units_sold = 0
        
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
        
        info = {
            'amount': amount,
            'curr_step': self.curr_step,
            'units_bought': units_bought,
            'units_sold': units_sold,
            'balance': self.balance,
            'net_worth': self.net_worth,
            'units_held': self.units_held,
            'profit': self.net_worth - self.init_balance,
        }
        
        return info
        
    def reset(self):
        self._init_portfolio()
        self.curr_step = self.roll_period
        obs = self._get_obs()
        return obs
    
    def step(self, action):
        info = self._act(action)
        self.curr_step += 1

        alpha = (self.curr_step / MAX_STEPS)
        info['reward'] = self.net_worth - self.init_balance
        reward = (self.net_worth - self.init_balance) / MAX_INT
        done = self.net_worth <= 0 or self.curr_step == MAX_STEPS
        obs = self._get_obs()

        if self.curr_step > self.df.shape[0] - self.batch_dur:
            self.curr_step = random.randint(self.roll_period, self.df.shape[0] - self.batch_dur)

        return obs, reward, done, info

    def render(self, show=False):
        print(f'curr_step: {self.curr_step}')
        print(f'balance: {self.balance}')
        print(f'net_worth: {self.net_worth}')
        print(f'units_held: {self.units_held}')
        print(f'net_profit: {self.net_worth - self.init_balance}')