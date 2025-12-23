# Currency Arbitrage Detection

This project detects currency arbitrage opportunities using the **Bellman-Ford algorithm**. It fetches live exchange rates and identifies profitable cycles where starting with one currency and trading through a series of others results in a net gain.

## Features

- Fetches exchange rates from [ExchangeRate API](https://www.exchangerate-api.com/).  
- Converts exchange rates to **negative log weights** for cycle detection.  
- Uses **Bellman-Ford** to find negative cycles corresponding to arbitrage opportunities.  
- Simulates converting a balance along the detected arbitrage paths.
