import requests
import json

# Replace with your actual Perplexity Finance API endpoint and key
API_ENDPOINT = "https://api.perplexity.finance/v1/stock_summary"
API_KEY = "your_api_key_here"

def fetch_ticker_data(ticker):
    # Example request payload or parameters (adjust as per actual API spec)
    params = {
        "ticker": ticker,
        "region": "IN"
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.get(API_ENDPOINT, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        # Here, adapt parsing depending on actual response structure from API
        return data
    else:
        print(f"Failed to fetch data for {ticker}. Status code: {response.status_code}")
        return None

def get_tickers_info(ticker_list):
    results = []
    for ticker in ticker_list:
        data = fetch_ticker_data(ticker)

        if data is None:
            continue

        # Construct your JSON output format based on API response fields
        ticker_summary = {
            "ticker": ticker,
            "price_info": {
                "current_price": data.get("current_price"),
                "change_percent": data.get("change_percent"),
                "trend_7days": data.get("trend_7days"),
            },
            "earnings_summary": {
                "quarter": data.get("latest_earnings_quarter"),
                "eps": data.get("eps"),
                "revenue": data.get("revenue"),
                "estimates_met": data.get("estimates_met"),
            },
            "financial_ratios": {
                "pe_ratio": data.get("pe_ratio"),
                "debt_to_equity": data.get("debt_to_equity"),
                "cash_flow": data.get("cash_flow"),
            },
            "analyst_ratings": {
                "buy": data.get("analyst_buy_count"),
                "hold": data.get("analyst_hold_count"),
                "sell": data.get("analyst_sell_count"),
                "target_price_avg": data.get("target_price_avg")
            },
            "news": data.get("recent_news", []),
            "sentiment_summary": data.get("sentiment_summary", ""),
            "references": data.get("references", [])
        }
        results.append(ticker_summary)
    return results

# List of tickers you want to analyze
tickers = ["RELIANCE", "HAL", "INFY", "TCS", "SBIN"]

if __name__ == "__main__":
    analysis_results = get_tickers_info(tickers)
    print(json.dumps(analysis_results, indent=2))
