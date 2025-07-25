import random

def get_stocks():
    stocks = ['AAPL', 'META', 'NVDA', 'GS', 'MSFT']
    print("Top 3 Stocks:", random.sample(stocks, 3))
