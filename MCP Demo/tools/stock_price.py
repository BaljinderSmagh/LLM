import yfinance as yf

def get_stock_price(ticker: str) -> str:
    """Fetch the current price of a stock using Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            return f"No data found for ticker '{ticker}'."

        price = data['Close'].iloc[-1]
        return f"The current price of {ticker.upper()} is ${price:.2f}."
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"
