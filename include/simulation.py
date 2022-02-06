import numpy as np
import pandas as pd
import QuantLib as ql
import random
import include.option_functions as option_functions

class Simulator():
    def __init__(self, process, periods_in_day = 1):
        self.process = process
        self.D = periods_in_day
        
    def set_properties_gbm(self, v, q, mu):
        self.v0 = v
        self.q = q
        self.mu = mu

    def set_properties_heston(self, v0, kappa, theta, sigma, rho, q, r):
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.q = q
        self.r = r
            
    def simulate(self, S0, T = 252, dt = 1/252):
        if self.process == 'GBM':
            self._sim_gbm(S0, self.mu, self.v0, T, dt)
        else:
            self._sim_heston(S0, self.v0, self.kappa, self.theta, self.sigma, self.rho, self.q, self.r, T, dt)

    def _sim_gbm(self, S0, mu, stdev, T, dt):
        self.St = np.zeros(T)
        self.St[0] = S0
                
        for t in range(1, T):
            self.St[t] = self.St[t-1] * np.exp(mu * dt + stdev * np.sqrt(dt)*np.random.normal())

    def _sim_heston(self, S0, v0, kappa, theta, sigma, rho, q, r, T, dt):
        r_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), r, ql.Actual360()))
        q_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), q, ql.Actual360()))
        s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
        process = ql.HestonProcess(r_handle, q_handle, s0_handle, v0, kappa, theta, sigma, rho)
        times = ql.TimeGrid(T / 365.25, dt)
        dimension = process.factors()
        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(dimension * dt, ql.UniformRandomGenerator()))
        seq = ql.GaussianMultiPathGenerator(process, list(times), rng, False)
        path = seq.next()
        values = path.value()
        St, Vt = values
        self.St = np.array([x for x in St])
        self.Vt = np.array([x for x in Vt])

    def getS(self):
        return self.St
    
    def return_set(self, strike_min, strike_max, quote_datetime, min_exp, max_exp, datearray, r):
        # Returns a simulated which looks similar to DataKeeper's sets 
        
        strike = random.uniform(strike_min, strike_max)
        strike = [strike] * len(self.St)
        
        exp = random.randint(min_exp, max_exp)
        expiration = datearray[datearray.index(quote_datetime) + int(exp)]
        expiration = [expiration] * len(self.St)
        
        quote_datetimes = []
        
        i = 0
        while len(quote_datetimes) < len(self.St):
            temp = [datearray[datearray.index(quote_datetime) + int(i)]] * self.D
            quote_datetimes += temp
            i = i + 1
            
        quote_datetimes = quote_datetimes[:len(self.St)]

        St = self.St / self.St[0]
        
        Ts = exp - np.arange(0, len(self.St)/(1/self.D), 1/self.D)

        df = pd.DataFrame()
        df['underlying_bid'] = St
        df['expiration'] = expiration
        df['strike'] = strike
        df['quote_datetime'] = quote_datetimes
        df['ticker'] = 'simulated'
        
        prices = []
        
        for i in range(len(self.St)):
            if self.process == 'GBM':
                price = option_functions.call_price(St[i], strike[i], r, self.q, self.v0, Ts[i]/252)
            else:
                price = option_functions.heston_price(St[i], strike[i], r, self.q, self.theta,
                    self.kappa, self.sigma, self.rho, self.v0, expiration[i], quote_datetimes[i])
            prices.append(price)

        df['bid'] = prices
        df['ask'] = prices
        
        return df