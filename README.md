# Limit-Order-Book-Market-Making-Simulator
Python Limit Order Book Market-Making Simulator with Inventory and Adverse Selection Control

**Background and design objectives**

A limit order book (LOB) is the core matching mechanism in most modern electronic markets: it stores outstanding buy/sell limit orders by price and time priority, and is updated by discrete book events—limit submissions, cancellations, and market orders—whose interaction determines spread, depth, and price dynamics.
For quant research and trading-role credibility, a simulator should do more than “random-walk price + spread”: it should 

(i) explicitly model bid/ask queues, 

(ii) generate fills via queue consumption, 

(iii) expose adverse selection (price moving against the liquidity provider after fills), and 

(iv) support inventory-aware quoting in a way that can be stress-tested across volatility regimes. Inventory control is a central element of canonical optimal market-making models (e.g., Avellaneda–Stoikov), where the dealer balances spread capture against inventory risk.

Adverse selection is structural in market microstructure: informed trading forces liquidity providers to widen spreads or reduce exposure, as formalized in classic theory (Glosten–Milgrom). Empirically and mechanistically, short-horizon price changes are strongly driven by order-book event imbalance (OFI), and order-book signals can be used to mitigate adverse selection in execution/market-making strategies.
