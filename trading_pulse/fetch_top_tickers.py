import pandas as pd
import json
from relevant_tickers_list import fetch_tickers_list
from ticker_info import get_tickers_info
from trading_prediction_model import StockPredictionModel  # Assuming you have saved the model script

def fetch_top_tickers():
    tickers_list_data = fetch_tickers_list()
    if tickers_list_data is None:
        print("Failed to fetch tickers. Exiting.")
    else:
        # Assuming the API response is a list of dicts with a 'ticker' key
        tickers_list = [item['ticker'] for item in tickers_list_data]
        print(f"Fetched {len(tickers_list)} relevant tickers.")

        tickers_info_list = get_tickers_info(tickers_list)

        df = pd.DataFrame(tickers_info_list)

        model = StockPredictionModel()

        top_picks = model.predict_top_tickers(df)
        print(top_picks)

if __name__ == "__main__":
    print(json.dumps(fetch_top_tickers(), indent=2))