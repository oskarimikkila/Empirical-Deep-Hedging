from scipy import log, exp, optimize
import QuantLib as ql
import numpy as np
import scipy.stats as si

def calc_impl_volatility(S, K, r, q, T, P):
    P_adj = P
    def price_comp(sigma):
        return P_adj-call_price(S, K, r, q, sigma, T)

    v = None
    t = 0
    s = -1
    # little noise to price if calculation is unsuccessfull
    while v is None and t < 20:
        P_adj = P + t * s * 0.0001
        try:
            v = optimize.brentq(price_comp, 0.001, 100, maxiter=1000)
        except:
            v = None
            if s > 0:
                t += 1
            s *= -1

    return v

def _d(S, K, r, q, v, T):
    d1 = (log(S / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    return d1, d2

def _N(d1, d2):
    return si.norm.cdf(d1), si.norm.cdf(d2)

def put_price(S, K, r, q, v, T):
    d1, d2 = _d(S, K, r, q, v, T)
    N1, N2 = _N(d1, d2)

    return -S * exp(-q * T) * (1 - N1) + K * exp(-r * T) * (1 - N2)

def call_price(S, K, r, q, v, T):
    if T <= 0.0:
        return max(S-K, 0)

    d1, d2 = _d(S, K, r, q, v, T)
    N1, N2 = _N(d1, d2)

    price = S * exp(-q * T) * N1 - K * exp(-r * T) * N2

    return price

def heston_price(S, K, r, q, theta, kappa, sigma, rho, v0, exp_date, cur_date):
    ql.Settings.instance().evaluationDate = ql.DateParser.parseFormatted(cur_date,'%Y-%m-%d')
    exp_date = ql.DateParser.parseFormatted(exp_date,'%Y-%m-%d')

    S = ql.QuoteHandle(ql.SimpleQuote(S))
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), r, ql.Actual360()))
    q_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), q, ql.Actual360()))

    heston_process = ql.HestonProcess(r_handle, q_handle, S, v0, kappa, theta, sigma, rho)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(exp_date)
    european_option = ql.VanillaOption(payoff, exercise)

    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process), 0.01, 1000)
    european_option.setPricingEngine(engine)
    price = european_option.NPV()
    return price