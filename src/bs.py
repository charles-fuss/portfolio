import os
import pandas as pd
import logging
from datetime import datetime, timezone
import math
from .stocks import Ticker
from scipy.stats import norm
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
tmpdir = os.path.join(os.getcwd(), 'tmp')
now = datetime.now(timezone.utc)

def black_scholes(obj, _type:str, K:int, T:Optional[int]=30):
        '''
        basic theory: options are functions (black scholes for example) of stochastic processes (stocks) and you can derive the greeks by solving PDEs
        - model assumes that the market consists of at least one risky asset ("stock") and one riskless asset, usually called the money market/bond.
        - at expiry, a call is worth the stock price minus the "discounted" strike price (baking in time value of $$$ plus probability of exercising the option) 
        or zero if expires OTM
        - returns fluctuate based on stock (EV)
        - closed form assumes constant params for easy solving, 
        
        formula = theta (time decay PDE) + Gamma (sensitivity of Delta to underlying price) + Drift (option prices grow @ r under risk-neutral measure) - Discounting (adjust for hedging; potential gains if unhedged)
            Delta = sensitivity to price (∂V / ∂S) (1st deriv)
            Gamma =  Delta's sensitivity to price (2nd deriv)
            Vega = option's sensitivity to volatility (deriv w.r.t. σ)
            Rho: interest-rate sensitivity
        S = current stock price
        K = strike
        T = time to expiration (in years)
        r = risk-free rate
        σ = implied volatility
        d1 = EV of S_t under the delta hedged measure (conditional on finishing in the money)
        d2 = probability that we finish ITM (S_k > K = N(d2))
        C = premium of call option (what is this option worth per share?)
        '''

        assert 1 <= T <= 252, 'k must be in terms of days (1<k<252)'
        T /= 252

        df = obj.price_action.sort_values('Date', ascending=False)
        std, S0 = df['log_returns'].apply(lambda x: x*math.sqrt(252)).std(), df['Close'].iloc[0] # std must live in same timeline as T (years); multiply by sqrt since variance scales linearly
        r = 4.148 / 100 # as of 11.16.25
        print(f"std:{std}")
        
        d1 = (math.log(S0/K) + (r+.5*std**2)*T)/(std*math.sqrt(T))
        d2 = d1 - std*math.sqrt(T)
        C = (S0 * norm.cdf(d1)) - (K*(math.e**(-r*T)) * norm.cdf(d2)) # cdf is way more based than pdf; used for calculating probability of all outcomes up until that point
        
        if _type == 'call': return d1, d2, C # probability weighted upside
        P = C - S0 + (K*math.e**-r*T)
        THIS IS WRONG, FIX!!!
        if _type =='put': return P, d1, d2 # probability-weighted downside
             
        return 0
        
from .utils import cereal_ticker

if __name__ == "__main__":
    # t = Ticker('OSCR')
    # cereal_ticker(_type='dump', dmp=[t], update=False)
    t = cereal_ticker(_type='load', raw_ticker_list=['OSCR'], update=False)[0]
    
    closes = t.price_action.sort_values('Date', ascending=False)['Close']
    p50, plus_5pct, minus_5pct = [closes.median(), closes[0]*1.05, closes[0]*.95]
    print(f"p50={p50},plus_5pct={plus_5pct},minus_5pct={minus_5pct}")
    # t of interest - quarters
    tvals, kvals = [63, 126, 189, 252], [p50, plus_5pct, minus_5pct]
    # d1, d2, C = black_scholes(obj=t, _type='put', K=kv,T=tv)
    # exit(0)
    for tv in tvals:
        for kv in kvals:
            print(f"computing {t.ticker} for K={kv} & T={tv}")
            d1, d2, C = black_scholes(obj=t, _type='put', K=kv,T=tv)
            print(f"d1:{round(norm.cdf(d1)*100, 2)}% ({round(d1,ndigits=3)})\nd2:{round(norm.cdf(d2)*100, 2)}% ({round(d2,ndigits=3)}) \nC:${round(C,ndigits=2)}\n")