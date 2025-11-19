import json
import websocket
from datetime import datetime
import pandas as pd

# turn on low-level debug logs
websocket.enableTrace(True)


class CryptoTicker:
    def __init__(self, symbol: str = "btcusdt"):
        """
        Simple Binance trade stream -> 1m bar collector.

        symbol: e.g. "btcusdt", "ethusdt"
        """
        self.symbol = symbol.lower()
        self.bars = {}   # {minute_timestamp: [prices]}
        self.ws = None

    # ----- WebSocket callbacks (bound methods) -----

    def _on_open(self, ws):
        print("OPEN!")

    def _on_message(self, ws, message: str):
        msg = json.loads(message)

        # trade price + trade time (ms)
        price = float(msg["p"])
        ts = datetime.fromtimestamp(msg["T"] / 1000.0)

        minute_key = ts.replace(second=0, microsecond=0)
        self.bars.setdefault(minute_key, []).append(price)

    def _on_error(self, ws, error):
        print("ERROR:", error)

    def _on_close(self, ws, close_status_code, close_msg):
        print("CLOSED:", close_status_code, close_msg)

    # ----- Public API -----

    def start(self):
        """Start streaming trades and aggregating into minute buckets."""
        stream_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"
        self.ws = websocket.WebSocketApp(
            stream_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws.run_forever()

    def to_ohlcv(self) -> pd.DataFrame:
        """
        Convert collected ticks to 1-minute OHLCV DataFrame.
        Volume here is just the count of trades in that minute.
        """
        records = []
        for ts, prices in sorted(self.bars.items()):
            records.append(
                {
                    "timestamp": ts,
                    "open": prices[0],
                    "high": max(prices),
                    "low": min(prices),
                    "close": prices[-1],
                    "volume": len(prices),
                }
            )

        if not records:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame.from_records(records).set_index("timestamp")
        return df


if __name__ == "__main__":
    # vpn to ukraine and you're good
    ticker = CryptoTicker("btcusdt")
    ticker.start()
    # when you interrupt the script (Ctrl+C), in a separate run you can do:
    # df = ticker.to_ohlcv()
    # print(df.tail())
