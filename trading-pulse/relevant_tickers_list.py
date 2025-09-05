import requests
import json

API_URL = "https://api.perplexity.ai/finance/query"
API_KEY = "your_api_key_here"

def fetch_tickers_list():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    query = ("What are the top 100 most news-active tickers traded in the Indian stock market "
             "in the last 7 days? List the company name, ticker symbol for BSE, "
             "and top 3 references in JSON format as "
             '[{"name": "...", "ticker": "...", "references": ["...", "...", "..."]}].')

    payload = {
        "query": query,
        "region": "IN",
        "response_format": "json"
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        # The 'data' field holds the conversational response. Parse as needed.
        return data
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

if __name__ == "__main__":
    analysis_results = fetch_tickers_list()
    print(json.dumps(analysis_results, indent=2))
