import pandas as pd
from relevant_tickers_list import fetch_tickers_list
from ticker_info import get_tickers_info
from trading_prediction_model import StockPredictionModel  # Assuming you have saved the model script

# Step 1: Fetch the list of relevant tickers
tickers_list_data = fetch_tickers_list()
if tickers_list_data is None:
    print("Failed to fetch tickers. Exiting.")
else:
    # Assuming the API response is a list of dicts with a 'ticker' key
    tickers_list = [item['ticker'] for item in tickers_list_data]
    print(f"Fetched {len(tickers_list)} relevant tickers.")

    # Step 2: Fetch detailed info for each ticker
    tickers_info_list = get_tickers_info(tickers_list)

    # Step 3: Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(tickers_info_list)

    # Step 4: Your existing model's preprocessing and prediction pipeline
    model = StockPredictionModel()
    # Assuming 'df' contains a 'profitable_trade' column for fine-tuning or is new data for prediction

    # Example for prediction
    top_picks = model.predict_top_tickers(df)
    print(top_picks)