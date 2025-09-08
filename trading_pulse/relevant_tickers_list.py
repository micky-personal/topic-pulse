import json
from datetime import date
from google_genai_service import google_genai_response
from store_collection_in_firestore import store_collection_in_firestore

def fetch_tickers_list():
    # query = ("What are the top 100 most news-active tickers traded in the Indian stock market "
    #          "in the last 7 days? List the company name, ticker symbol for BSE, "
    #          "and top 3 references in JSON format as "
    #          '[{"name": "...", "ticker": "...", "references": ["...", "...", "..."]}].')

    query = ("Give me a list of 10 tickers traded in the Indian stock market " 
             "in JSON format as "
             '[{"name": "...", "ticker": "...", "references": ["...", "...", "..."]}].')

    tickers_list_data = google_genai_response(query)

    if tickers_list_data is None:
        print("Failed to get tickers list response.")
    else:
        store_tickers_in_firestore(tickers_list_data)

    return tickers_list_data

def store_tickers_in_firestore(tickers_data):
    current_date_str = date.today().isoformat()
    firestore_document = {
        "date": current_date_str,
        "tickers_list": tickers_data
    }
    collection_name = "daily_tickers_list"
    store_collection_in_firestore(collection_name, firestore_document)

if __name__ == "__main__":
    analysis_results = fetch_tickers_list()
    print(json.dumps(analysis_results, indent=2))
