from include.option_functions import calc_impl_volatility
import include.option_functions as option_functions
import numpy as np
import pandas as pd
import include.data_keeper as data_keeper
import include.simulation as simulation

from datetime import datetime
from scipy.stats import norm
from include.settings import getSettings

class Env():
    def __init__(self, s = getSettings()):
        self.sim = simulation.Simulator(s['process'], periods_in_day = s['D'])
        
        # Env Params
        self.transaction_cost = s['transaction_cost']
        self.kappa = s['kappa']
        self.reward_exponent = s['reward_exponent']
        self.SIGMA = s['SIGMA']
        self.process = s['process']
        self.q = s['q']
        self.r_df = pd.read_csv('data/1yr_treasury.csv')        
        self.heston_params = pd.read_csv('data/heston_params.csv')

        self.D, self.steps = s['D'], s['n_steps']
        if self.process == 'Real':
            self.data_keeper = data_keeper.DataKeeper(self.steps)
        self.data_set = pd.DataFrame()
        self.t, self.v, self.date_idx, = 0, 0.0, 0
        
        # holds option's properties
        self.option = {}
        # Spot price time series
        self.S = []
        
    def get_bs_delta(self):
        d1, _ = option_functions._d(self.option['S/K']*self.K, self.K, self.r, self.q, self.v, self.option['T']/365)
        return norm.cdf(d1)

    def __concat_state(self):
        return np.array([self.option['S/K'], self.option['T']/30, self.stockOwned, self.v])
    
    def __update_option(self):
        row = self.data_set.loc[self.t, :]

        spot = row['underlying_bid']
        P = 0.5 * (row['bid'] + row['ask'])
        self.expiry = row['expiration'][0:10]
        self.K = float(row['strike'])
        self.S[self.t] = spot
        self.cur_date = row['quote_datetime'][0:10]
        self.ticker = row['ticker']
        self.option['P'] = P

        try:
            self.r = self.r_df.loc[self.r_df['Date'] == self.cur_date, '1y'].iloc[0]
        except:
            # If r is missing, use previous (shouldn't happen)
            print("r missing:", self.cur_date)
        
        ttm = (datetime.strptime(self.expiry, '%Y-%m-%d') - \
            datetime.strptime(self.cur_date, '%Y-%m-%d')).days - (1 - (self.D - self.t%self.D) / self.D)

        self.option['T'] = ttm
        self.option['S/K'] = spot / self.K
              
        iv = calc_impl_volatility(spot, self.K, self.r, self.q, ttm/365, P)
        # Sometimes impossible to solve IV, have to use the previous value
        if iv:
            self.v = iv
        
    def reset(self, testing = False, start_a = 0.0, start_b = 0.0):
        # Reset must be called when episode begins
        # testing indicates if empirical data should be used
        self.testing = testing
        self.t = 0
        self.S = np.zeros(self.steps + 1)
        
        self.stockOwned, self.b_stockOwned = start_a, start_b
        
        new_set = None
        
        if testing:
            self.data_set = self.data_keeper.next_test_set()
        else:
            if self.process == 'Real':
                self.data_set = self.data_keeper.next_train_set()
            else:
                while new_set is None:
                    dates = self.r_df['Date']
                    dates = dates[dates >='2013-01-01']
                    dates = sorted(dates.unique())[:-90]
                    quote_datetime = np.random.choice(dates)

                    try:
                        self.r = self.r_df.loc[self.r_df['Date'] == quote_datetime, '1y'].iloc[0]
                    except:
                        self.r = 0.01
                    
                    if self.process == 'GBM':
                        self.sim.set_properties_gbm(self.SIGMA, self.q, .0)
                        T = self.steps + 1
                        dt = 1/(252*self.D)
                    else:
                        params = self.heston_params[self.heston_params['date'] == quote_datetime]
                        if params.empty:
                            continue

                        v0 = params.iloc[0]['v0']
                        kappa = params.iloc[0]['kappa']
                        theta = params.iloc[0]['theta']
                        sigma = params.iloc[0]['sigma']
                        rho = params.iloc[0]['rho']

                        self.sim.set_properties_heston(v0, kappa, theta, sigma, rho, self.q, self.r)
                        T = 5
                        dt = 35

                    self.sim.simulate(1.0, T, dt)
                    new_set = self.sim.return_set(.85, 1.15, quote_datetime, 15, 90, sorted(self.r_df['Date'].unique()), self.r)
                    self.data_set = new_set
            
        self.__update_option()
        return self.__concat_state()

    def step(self, delta):
        # Step from T0 to T1
        def reward_func(pnl):
            # Reward scaled for clarity and small positive added
            pnl *= 100
            reward = 0.03 + pnl - self.kappa * (abs(pnl)**self.reward_exponent)
            return reward * 10
        
        infos = {'T':self.option['T'],
                'S/K':self.option['S/K']}
        
        infos['Date'] = self.cur_date
        infos['DateStep'] = self.t % self.D
        
        b_delta = self.get_bs_delta()
        
        #Linear transaction cost based on current (T0) face value and change in position
        t_cost = -abs(-delta - self.stockOwned) * self.S[self.t] * self.transaction_cost
        b_t_cost =  -abs(-b_delta - self.b_stockOwned) * self.S[self.t] * self.transaction_cost
        
        opt_old_price = self.option['P']
        
        self.t += 1
        
        self.__update_option()
        
        done = self.t >= self.steps

        opt_new_price = self.option['P']

        # PnL effect of underlying position
        pnl = -delta * (self.S[self.t] - self.S[self.t - 1])
        b_pnl = -b_delta * (self.S[self.t] - self.S[self.t - 1])
        
        # PnL effect of option price change and transaction cost
        pnl += (opt_new_price - opt_old_price) + t_cost
        b_pnl += (opt_new_price - opt_old_price) + b_t_cost        
        
        self.stockOwned = -delta
        self.b_stockOwned = -b_delta

        reward = reward_func(pnl)
        b_reward = reward_func(b_pnl)
        
        infos['B Reward'] = b_reward
        infos['A Reward'] = reward
        infos['A PnL'] = pnl
        infos['B PnL'] = b_pnl      
        infos['P0'] = opt_new_price  
        infos['P-1'] = opt_old_price
        infos['S0'] = self.S[self.t]
        infos['S-1'] = self.S[self.t - 1]
        infos['A Pos'] = self.stockOwned
        infos['B Pos'] = self.b_stockOwned
        infos['A TC'] = t_cost
        infos['B TC'] = b_t_cost
        infos['A PnL - TC'] = pnl - t_cost
        infos['B PnL - TC'] = b_pnl - b_t_cost
        infos['Expiry'] = self.expiry
        infos['v'] = self.v
        
        return self.__concat_state(), reward, done, infos
