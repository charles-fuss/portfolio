import os
import pickle
import pandas as pd
import logging
from typing import Union, Optional
from datetime import datetime, timezone
from stocks import Ticker


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
tmpdir = os.path.join(os.getcwd(), 'tmp')
now = datetime.now(timezone.utc)

# serialization
def cereal_ticker(_type:str, update:bool, raw_ticker_list:Optional[list[str]] | None = None, dmp:Optional[list[Ticker]] | None = None) -> Union[int | list[Ticker]]:
    if _type == 'dump':
        for ticker in dmp:
            try:
                with open(f'{tmpdir}/{ticker.ticker}.pkl', 'wb') as f:
                    pickle.dump(ticker, f)
                logger.warning(f"Wrote {ticker.ticker} to {tmpdir}")
            except Exception as e: 
                logger.error(f"Error writing {ticker.ticker} -- {e}")
        return 0

    elif _type == 'load':
        tck = []
        for ticker in raw_ticker_list:
            try:
                with open(f'{tmpdir}/{ticker}.pkl', 'rb') as f:
                    obj = pickle.load(f)
                # this is not the right syntax to check the date... fix later
                if not update and obj.last_updated != now: logger.warning(f"{ticker} loaded successfully (last updated {obj.last_updated} & price action {obj.action_recency})")    
                tck.append(obj)
            except Exception as e: 
                breakpoint()
                logger.error(f"No results for {ticker}; loading manually")
                tck.append(Ticker(ticker))
        return tck
        
