import os
import yfinance as yf
import numpy as np
import pickle
import pandas as pd
import scipy.stats
import functools
import logging
from typing import Union, Optional
from backtest import *
from datetime import datetime, timedelta, timezone
from utils import cereal_ticker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# download and save to /tmp later; loading from remote takes forever
TICKERS = {
    "blue_chips": [
        "AAPL","MSFT","AMZN","GOOG","META","JNJ","PG","KO","PEP","UNH",
        "JPM","BAC","WMT","HD","MCD","DIS","CVX","XOM","COST","V",
        "MA","ABBV","TMO","AVGO","ORCL","IBM","RTX","PFE","MRK","NKE"
    ],

    "high_risk": [
        "NVDA","TSLA","PLTR","OSCR","COIN","SNOW","AFRM","ROKU","UPST","HOOD",
        "SOFI","LCID","RBLX","TDOC","ZM","DASH","SHOP","LMND","FSLY","CLOV",
        "AI","CRWD","NET","PATH","MDB","CELH","DKNG","W","NKLA","IONQ"
    ],

    "commodities": [
        "GLD","SLV","PALL","PPLT","GDX","GDXJ","XME","FCX","SCCO","BHP",
        "RIO","VALE","USO","BNO","XLE","CVX","XOM","OXY","KOS","DBC",
        "DBA","WEAT","CORN","SOYB","UNG","UGA","CPER","LIT","URA","XOP"
    ],

    "bonds": [
        "TLT","IEF","SHY","BND","AGG","LQD","HYG","JNK","VCIT","VCLT",
        "GOVT","TIP","SCHZ","MBB","BIV","BLV","EDV","IGSB","IGEB","USIG",
        "ANGL","EMB","VWOB","SHV","SPSB","SPIB","FLOT","STIP","BSCM","BSCL"
    ]
}
now = datetime.now(timezone.utc)
tmpdir = os.path.join(os.getcwd(), 'tmp')
MUTED_PROVIDERS = ['motley', 'fool']
class Ticker():
    
    @staticmethod
    def black_scholes():
        return 0
    
    @staticmethod 
    def safe_parse_iso(ts):
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

    def __init__(self, ticker:str, news_lookback:Optional[int]=1, stat_lookback:Optional[int]=30):
        assert stat_lookback <= 120, "Up to 120d supported, gonna be too slow"
        self.stat_lookback = stat_lookback
        self.ytick = yf.Ticker(ticker) # maybe download and convert to polars, probs faster
        self.ticker = ticker
        # self.shares = self.ytick.shares
        self.news = [
            item for item in self.ytick.news
            if (
                (dt := self.safe_parse_iso(item['content'].get('displayTime'))) and dt is not None and dt
                >= now - timedelta(days=news_lookback)
                and not any(
                    muted.lower() in item['content']['provider']['displayName'].lower()
                    for muted in MUTED_PROVIDERS
                )
            )
        ]  

        d = self.ytick.info
        # BIG OL' STAT COMPILATION
        self.price_action = self.ytick.history(period=f"{self.stat_lookback}d").reset_index() # keep as pandas -- quicker than json pulls
        self.core_stats = {
            "currentPrice": d.get("currentPrice"),
            "regularMarketPrice" : d.get("regularMarketPrice"),
            "previousClose" : d.get("previousClose"),
            "_open" : d.get("open"),
            "dayHigh" : d.get("dayHigh"),
            "dayLow" : d.get("dayLow"),
            "regularMarketOpen" : d.get("regularMarketOpen"),
            "regularMarketDayHigh" : d.get("regularMarketDayHigh"),
            "regularMarketDayLow" : d.get("regularMarketDayLow"),
            "fiftyTwoWeekHigh" : d.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow" : d.get("fiftyTwoWeekLow"),
            "twoHundredDayAverage" : d.get("twoHundredDayAverage"),
            "fiftyDayAverage" : d.get("fiftyDayAverage"),
        }
        self.risk_stats = {
            "beta":d.get("beta"),
            "_52WeekChange":d.get("52WeekChange"),
            "fiftyTwoWeekChangePercent":d.get("fiftyTwoWeekChangePercent"),
            "regularMarketChangePercent":d.get("regularMarketChangePercent"),
            "auditRisk":d.get("auditRisk"),
        }
        self.liquidity = {
            "volume":d.get("volume"),
            "averageVolume":d.get("averageVolume"),
            "averageVolume10days":d.get("averageVolume10days"),
            "floatShares":d.get("floatShares"),
            "sharesOutstanding":d.get("sharesOutstanding"),
            "marketCap":d.get("marketCap"),
        }
        self.income = {
            "dividendRate":d.get("dividendRate"),
            "dividendYield":d.get("dividendYield"),
            "exDividendDate":d.get("exDividendDate"),
            "trailingAnnualDividendRate":d.get("trailingAnnualDividendRate"),
            "trailingAnnualDividendYield":d.get("trailingAnnualDividendYield"),
        }
        self.valuation_stats = {
            "trailingPE":d.get("trailingPE"),
            "forwardPE":d.get("forwardPE"),
            "priceToSalesTrailing12Months":d.get("priceToSalesTrailing12Months"),
            "priceToBook":d.get("priceToBook"),
            "enterpriseValue":d.get("enterpriseValue"),
            "enterpriseToRevenue":d.get("enterpriseToRevenue"),
            "enterpriseToEbitda":d.get("enterpriseToEbitda"),
        }
        self.balsheet_stats = {
            "totalDebt":d.get("totalDebt"),
            "totalCash":d.get("totalCash"),
            "totalCashPerShare":d.get("totalCashPerShare"),
            "bookValue":d.get("bookValue"),
            "debtToEquity":d.get("debtToEquity"),
            "quickRatio":d.get("quickRatio"),
            "currentRatio":d.get("currentRatio"),
        }
        self.company_stats = {
            "employees":d.get("fullTimeEmployees"),
            "dividend_yield": d.get("dividendYield"),
            'max_age': d.get("maxAge"),
            'industry':d.get("industry"),
            'sector': d.get("sector"),
            'state': d.get("state"),
        }
        
        # can extend dis if necessary
        e = self.ytick.get_earnings_dates().reset_index().sort_values(['Earnings Date'])
        self.n_earn = dict(e.iloc[-1][['EPS Estimate', 'Reported EPS', 'Surprise(%)', 'Earnings Date']])
        self.p_earn = dict(e.iloc[-2][['EPS Estimate', 'Reported EPS', 'Surprise(%)', 'Earnings Date']])


        # advanced stats
        self._ma10 = self.ma(lookback=10, weighted=True)
        self._ma30 = self.ma(lookback=30, weighted=True)


        self.last_updated = now
        self.action_recency = self.price_action['Date'].max()
        logger.info(f'instantiated {self.ticker}')

        # Do options stuff later
        # self.price = self.ticker()
        # self.expirations = ticker.options
        # self.greeks = black_scholes()     


    # pickling attrs; handle yfinance client 
    def __getstate__(self):
            state = self.__dict__.copy()
            # remove unpicklable things
            if "ytick" in state:
                state["ytick"] = None
            return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if getattr(self, "ticker", None) and self.ytick is None:
            self.ytick = yf.Ticker(self.ticker)


    # Advanced indicators, some stats/strategic frameworks
    def ma(self, lookback:int, weighted:bool) -> pd.DataFrame:
        sma = self.price_action['Close'].rolling(window=lookback).mean().dropna()
        if weighted: 
            _weights = np.arange(1, lookback+1)
            return sma.apply(
                lambda x: (np.dot(x, _weights) / _weights.sum()).cumsum()[-1]
            )
        else: return sma[-1]

    from backtesting import Backtest, Strategy
    def signal_significance(self, rel:Strategy):
        n = 10_000
        p = .05
        test = 1 #np.? --> t-test
        return 0

    def pair_trade(self, s2:yf.Ticker):
        #significance
        return 0

def pull_random_stocks(_type: Union[str|None], p:int) -> list[str]:
    assert _type in [None, 'blue_chips', 'high_risk', 'commodities', 'bonds'], "choose a valid type (['blue_chips', 'high_risk', 'commodities', 'bonds', None])"
    if _type == None: 
        chc = np.random.choice(len(TICKERS))
        _type=list(TICKERS.keys())[chc]
        logger.info(f"chose {_type}")

    pop = TICKERS.get(_type, None)
    stx = set()
    while len(stx) != p:
        choice = np.random.choice(pop)
        stx.add(choice)
    return list(stx)



def main():
    ## == testing ===
    # stx = pull_random_stocks(_type='blue_chips', p=5)
    stx = ['PG', 'WMT', 'NKE', 'JPM']
    cereal_ticker(_type='load', raw_ticker_list=stx, update=False)
    
    
    # Always dump portfolio so we can see if it's already in tmp
    # cereal_ticker(_type='dump', dmp=portfolio, update=False)

if __name__ == '__main__':
    main()


# TODO
    # work on options stuff (black_scholes)
    # basic strategies
    # crypto potentially (CMC)