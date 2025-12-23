from math import log
from math import exp
import requests


# 1. Fetch currency rates and compute -log weights

def get_currencies(api_key, currencies):
    n = len(currencies)
    rates = [[0 for _ in range(n)] for _ in range(n)]

    for i, c1 in enumerate(currencies):
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{c1}"
        response = requests.get(url)
        conversion_rates = response.json()['conversion_rates']
        
        for j, c2 in enumerate(currencies):
            if c1 == c2:
                rates[i][j] = 0  # log(1) = 0
            else:
                rates[i][j] = -log(conversion_rates[c2])

    return rates


# 2. Bellman-Ford arbitrage detection

def bellman_ford(currencies, weights, log_margin=1e-8):
    n = len(currencies)
    min_dist = [float('inf')] * n
    pre = [-1] * n
    start = 0
    min_dist[start] = 0
    opportunities = []

    # Relax edges
    for _ in range(n-1):
        for u in range(n):
            for v in range(n):
                if min_dist[v] > min_dist[u] + weights[u][v]:
                    min_dist[v] = min_dist[u] + weights[u][v]
                    pre[v] = u

    # Check for negative cycles
    for u in range(n):
        for v in range(n):
            if min_dist[v] > min_dist[u] + weights[u][v] + log_margin:
                # Reconstruct cycle
                cycle = [v]
                curr = u
                while curr not in cycle:
                    cycle.append(curr)
                    curr = pre[curr]
                cycle.append(v)
                
                if len(cycle) > 3:
                    path = [currencies[i] for i in cycle[::-1]]
                    if path not in opportunities:
                        opportunities.append(path)
    return opportunities


# 3. Main script
if __name__ == "__main__":
    api_key = 'e3a1e4ac4e4d1d1894c7431d'
    top10_currencies = ['USD', 'EUR', 'CAD', 'JPY', 'SGD']

    weights = get_currencies(api_key, top10_currencies)
    arbitrage_paths = bellman_ford(top10_currencies, weights)

    for path in arbitrage_paths:
        print("Arbitrage path:", path)
        balance = 100
        print("Simulation:")
        for i in range(len(path)-1):
            c1 = path[i]
            c2 = path[i+1]
            idx1 = top10_currencies.index(c1)
            idx2 = top10_currencies.index(c2)
            rate = -weights[idx1][idx2]
            balance *= exp(rate)
            print(f"{c1} -> {c2}: balance = {balance:.2f}")
        print()
