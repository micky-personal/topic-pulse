import logging
import os
from datetime import datetime, timezone, timedelta

import joblib
import numpy as np
import pandas as pd
from google.cloud import storage, firestore
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Cloud Configuration ---
BUCKET_NAME = 'your-gcs-bucket-name'
MODEL_PATH = 'models/trading_model.pkl'
FIRESTORE_COLLECTION = 'predictions' # Collection to store trade predictions
IST = timezone(timedelta(hours=5, minutes=30))

# Initialize Firestore client
db = firestore.Client()

class StockPredictionModel:
    def __init__(self):
        self.model = load_model_from_gcs()
        if self.model:
            logging.info("Model loaded from GCS successfully.")
        else:
            logging.warning("No pre-trained model found. The model will need to be trained first.")
        self.preprocessor = None
        self.features = None

    def preprocess_data(self, df):
        df['analyst_score'] = (df['buy'] * 2 + df['hold'] * 1 - df['sell'] * 1) / (df['buy'] + df['hold'] + df['sell'])
        df['sentiment_score'] = df['sentiment_summary'].map({'positive': 1, 'neutral': 0, 'negative': -1})
        self.features = [
            'current_price', 'change_percent', 'trend_7days', 'eps', 'revenue',
            'estimates_met', 'pe_ratio', 'debt_to_equity', 'cash_flow',
            'analyst_score', 'sentiment_score'
        ]
        numeric_features = [f for f in self.features if df[f].dtype in ['int64', 'float64']]
        categorical_features = [f for f in self.features if df[f].dtype == 'object']
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        return df

    def train_and_evaluate(self, df, n_splits=5):
        df = self.preprocess_data(df)
        if 'profitable_trade' not in df.columns:
            logging.error("Target column 'profitable_trade' not found in the training data.")
            return
        df['profitable_trade'] = df['profitable_trade'].astype(int)

        x = df[self.features]
        y = df['profitable_trade']
        # ... (rest of the training logic is unchanged) ...
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_y_true, all_y_pred = [], []
        logging.info(f"Starting {n_splits}-fold cross-validation...")
        for fold, (train_index, test_index) in enumerate(skf.split(x, y), 1):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42, class_weight='balanced_subsample'
                ))
            ])
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
        logging.info("Model training complete and saved to GCS.")

    def predict_top_tickers(self, df, top_n=3):
        if self.model is None:
            logging.error("Model not trained. Please run train_and_evaluate() first.")
            return pd.DataFrame()
        df_processed = self.preprocess_data(df)
        x_new = df_processed[self.features]
        probabilities = self.model.predict_proba(x_new)[:, 1]
        df['prediction_probability'] = probabilities
        df['prediction_timestamp'] = datetime.now(IST).isoformat()
        df['actual_profitable_trade'] = None # Use None for Firestore compatibility
        top_picks_df = df.sort_values(by='prediction_probability', ascending=False).head(top_n)
        return top_picks_df

    def retrain_from_firestore(self, collection_name=FIRESTORE_COLLECTION):
        """Queries Firestore for labeled data and retrains the model."""
        logging.info(f"Starting retraining process with data from Firestore collection '{collection_name}'...")
        
        # Query for documents that have been labeled (0 or 1)
        docs = db.collection(collection_name).where('actual_profitable_trade', 'in', [0, 1]).stream()
        
        labeled_data = [doc.to_dict() for doc in docs]

        if not labeled_data:
            logging.warning("No labeled data found in Firestore to retrain the model. Aborting.")
            return

        df_train = pd.DataFrame(labeled_data)
        df_train['profitable_trade'] = df_train['actual_profitable_trade']

        logging.info(f"Retraining model with {len(df_train)} new labeled data points from Firestore.")
        self.train_and_evaluate(df_train)

def store_predictions_in_firestore(prediction_data, collection_name=FIRESTORE_COLLECTION):
    """Stores prediction data as documents in a Firestore collection."""
    if prediction_data.empty:
        return

    logging.info(f"Storing {len(prediction_data)} predictions in Firestore collection '{collection_name}'...")
    for _, row in prediction_data.iterrows():
        # Let Firestore auto-generate the document ID
        doc_ref = db.collection(collection_name).document()
        # Convert numpy types to native Python types for Firestore
        record = row.to_dict()
        for key, value in record.items():
            if isinstance(value, np.generic):
                record[key] = value.item()
        doc_ref.set(record)

# (load_dummy_data, save_model_to_gcs, load_model_from_gcs are unchanged)

def load_dummy_data(size=120):
    data = []
    for _ in range(size):
        data.append({
            'ticker': f'TICK{_}',
            'current_price': np.random.uniform(10, 1000),
            'change_percent': np.random.uniform(-0.05, 0.05),
            'trend_7days': np.random.choice(['up', 'down', 'stable']),
            'eps': np.random.uniform(0.1, 10.0),
            'revenue': np.random.uniform(1e6, 1e8),
            'estimates_met': np.random.choice([True, False]),
            'pe_ratio': np.random.uniform(5, 50),
            'debt_to_equity': np.random.uniform(0.01, 3.0),
            'cash_flow': np.random.uniform(1e5, 1e7),
            'buy': np.random.randint(1, 20),
            'hold': np.random.randint(1, 15),
            'sell': np.random.randint(1, 10),
            'sentiment_summary': np.random.choice(['positive', 'negative', 'neutral']),
            'profitable_trade': 1 if np.random.uniform(0, 1) > 0.6 else 0
        })
    return pd.DataFrame(data)

def save_model_to_gcs(model_path):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_PATH)
        blob.upload_from_filename(model_path)
        logging.info(f"Successfully saved model to gs://{BUCKET_NAME}/{MODEL_PATH}")
    except Exception as e:
        logging.error(f"Error saving model to GCS: {e}")

def load_model_from_gcs():
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_PATH)
        local_path = "temp_trading_model.pkl"
        blob.download_to_filename(local_path)
        model = joblib.load(local_path)
        os.remove(local_path)
        return model
    except Exception as e:
        # It's okay if the model doesn't exist on the first run
        logging.warning(f"Could not load model from GCS: {e}")
        return None

if __name__ == "__main__":
    model_instance = StockPredictionModel()

    # === Use Case 1: Daily Prediction ===
    print("--- Running Daily Prediction ---")
    df_today = load_dummy_data(size=20)
    top_picks_today = model_instance.predict_top_tickers(df_today, top_n=3)
    print(f"Today's top 3 picks:\n{top_picks_today[['ticker', 'prediction_probability']]}\n")
    store_predictions_in_firestore(top_picks_today)
    print("Predictions saved to Firestore.\n")

    # === Use Case 2: Periodic Retraining ===
    # This would be run periodically (e.g., monthly) via a cron job or manually.
    print("--- Running Periodic Retraining ---")
    # In the real world, you would have already manually updated the 'actual_profitable_trade' 
    # field in your Firestore documents from 'null' to 0 or 1.
    model_instance.retrain_from_firestore()
