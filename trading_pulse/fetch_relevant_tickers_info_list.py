import json
from google_genai_service import google_genai_response

def fetch_relevant_tickers_info_list(ticker_list):
    query = get_query_for_ticker_info(ticker_list)
    ticker_info_list = google_genai_response(query)

    results = []
    for ticker_info in ticker_info_list:
        if ticker_info is None:
            continue

        ticker_summary = get_ticker_info_json(ticker_info)
        results.append(ticker_summary)
    return results


def get_ticker_info_json(ticker_info):
    return {
        "ticker": ticker_info.get("ticker"),
        "price_info": {
            "current_price": ticker_info.get("price_info", {}).get("current_price"),
            "change_percent": ticker_info.get("price_info", {}).get("change_percent"),
            "trend_7days": ticker_info.get("price_info", {}).get("trend_7days"),
        },
        "earnings_summary": {
            "quarter": ticker_info.get("earnings_summary", {}).get("quarter"),
            "eps": ticker_info.get("earnings_summary", {}).get("eps"),
            "revenue": ticker_info.get("earnings_summary", {}).get("revenue"),
            "estimates_met": ticker_info.get("earnings_summary", {}).get("estimates_met"),
        },
        "financial_ratios": {
            "pe_ratio": ticker_info.get("financial_ratios", {}).get("pe_ratio"),
            "debt_to_equity": ticker_info.get("financial_ratios", {}).get("debt_to_equity"),
            "cash_flow": ticker_info.get("financial_ratios", {}).get("cash_flow"),
        },
        "analyst_ratings": {
            "buy": ticker_info.get("analyst_ratings", {}).get("buy"),
            "hold": ticker_info.get("analyst_ratings", {}).get("hold"),
            "sell": ticker_info.get("analyst_ratings", {}).get("sell"),
            "target_price_avg": ticker_info.get("analyst_ratings", {}).get("target_price_avg")
        },
        "news": ticker_info.get("news", []),
        "sentiment_summary": ticker_info.get("sentiment_summary", ""),
        "references": ticker_info.get("references", [])
    }


def get_query_for_ticker_info(tickers_list):
    query = (
        "I want to analyze the tickers - {tickersList}. "
        "Provide the latest comprehensive financial summary for the above tickers traded in India, "
        "including the following for each ticker: "
        "- Current stock price and price change (intraday and 7-day trend) "
        "- Earnings summary for the most recent quarter (EPS, revenue vs estimates) "
        "- Major financial ratios: PE ratio, debt-to-equity, cash flow "
        "- Analyst ratings and target price consensus "
        "- Recent significant news and headlines impacting the stock "
        "- Sentiment analysis summary from the latest news and earnings calls "
        "Present the output strictly in this JSON format: "
        "[ "
        "{{ "
        "  \"ticker\": string, "
        "  \"price_info\": {{ "
        "    \"current_price\": float, "
        "    \"change_percent\": float, "
        "    \"trend_7days\": string "
        "  }}, "
        "  \"earnings_summary\": {{ "
        "    \"quarter\": string, "
        "    \"eps\": float, "
        "    \"revenue\": float, "
        "    \"estimates_met\": boolean "
        "  }}, "
        "  \"financial_ratios\": {{ "
        "    \"pe_ratio\": float, "
        "    \"debt_to_equity\": float, "
        "    \"cash_flow\": float "
        "  }}, "
        "  \"analyst_ratings\": {{ "
        "    \"buy\": int, "
        "    \"hold\": int, "
        "    \"sell\": int, "
        "    \"target_price_avg\": float "
        "  }}, "
        "  \"news\": [ "
        "    {{\"headline\": string, \"date\": string, \"source\": string, \"sentiment\": string}} "
        "  ], "
        "  \"sentiment_summary\": string, "
        "  \"references\": [ "
        "    {{\"source\": string, \"url\": string}} "
        "  ] "
        "}} "
        "] "
        "Only fill the fields that are available. Provide references for all facts and data points at the end. "
        "Respond in this exact JSON format."
    )
    return query.format(tickersList=tickers_list)

if __name__ == "__main__":
    # List of tickers you want to analyze
    tickers = ["RELIANCE", "HAL", "INFY", "TCS", "SBIN"]
    analysis_results = fetch_relevant_tickers_info_list(tickers)
    print(json.dumps(analysis_results, indent=2))
