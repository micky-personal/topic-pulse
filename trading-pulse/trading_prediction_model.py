# filename: trading_prediction_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class StockPredictionModel:
    """
    A machine learning model class for predicting profitable stock trades.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.features = None

    def preprocess_data(self, df):
        """
        Prepares the data by creating features and a preprocessing pipeline.

        Args:
            df (pd.DataFrame): The raw input DataFrame containing stock metrics.

        Returns:
            pd.DataFrame: The DataFrame with engineered features.
        """
        # Feature Engineering: Combine raw data into more informative features
        df['analyst_score'] = (df['buy'] * 2 + df['hold'] * 1 - df['sell'] * 1) / (df['buy'] + df['hold'] + df['sell'])
        df['sentiment_score'] = df['sentiment_summary'].map({'positive': 1, 'neutral': 0, 'negative': -1})

        # Define features and target. Assuming 'profitable_trade' is the target column.
        self.features = [
            'current_price', 'change_percent', 'trend_7days', 'eps', 'revenue',
            'estimates_met', 'pe_ratio', 'debt_to_equity', 'cash_flow',
            'analyst_score', 'sentiment_score'
        ]

        # Identify numerical and categorical features
        numerical_features = [
            'current_price', 'change_percent', 'eps', 'revenue', 'pe_ratio',
            'debt_to_equity', 'cash_flow', 'analyst_score', 'sentiment_score'
        ]
        categorical_features = ['trend_7days', 'estimates_met']

        # Create preprocessing pipeline
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        return df

    def train(self, df_train):
        """
        Trains the RandomForestClassifier model using the provided data.

        Args:
            df_train (pd.DataFrame): The training DataFrame with features and target.
        """
        logging.info("Starting model training...")

        # Separate features and target
        X_train = df_train[self.features]
        y_train = df_train['profitable_trade']

        # Create the full modeling pipeline with the preprocessor and classifier
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
        ])

        # Train the model
        self.model.fit(X_train, y_train)
        logging.info("Model training complete.")

    def evaluate(self, df_test):
        """
        Evaluates the model's performance on a test dataset.

        Args:
            df_test (pd.DataFrame): The testing DataFrame with features and target.
        """
        if self.model is None:
            logging.error("Model has not been trained. Cannot evaluate.")
            return

        X_test = df_test[self.features]
        y_test = df_test['profitable_trade']

        y_pred = self.model.predict(X_test)

        logging.info("--- Model Performance Metrics ---")
        logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        logging.info(f"Precision: {precision_score(y_test, y_pred):.2f}")
        logging.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
        logging.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))

    def predict_top_tickers(self, df_new, num_tickers=3):
        """
        Predicts the top tickers for a given day based on a new dataset.

        Args:
            df_new (pd.DataFrame): DataFrame of new tickers for prediction.
            num_tickers (int): The number of top tickers to recommend.

        Returns:
            pd.DataFrame: A DataFrame of the top recommended tickers with their prediction scores.
        """
        if self.model is None:
            logging.error("Model has not been trained. Cannot make predictions.")
            return pd.DataFrame()

        logging.info("Making daily predictions...")
        # Make predictions (probabilities) for the new data
        prediction_scores = self.model.predict_proba(df_new[self.features])[:, 1]

        # Add the prediction scores to the DataFrame
        df_new = df_new.copy()
        df_new['prediction_score'] = prediction_scores

        # Rank and select the top N tickers
        top_tickers = df_new.sort_values(by='prediction_score', ascending=False).head(num_tickers)

        logging.info("Top %d ticker recommendations:", num_tickers)
        for _, row in top_tickers.iterrows():
            logging.info(f"  - Ticker: {row['ticker']} | Predicted Score: {row['prediction_score']:.2f}")

        return top_tickers


if __name__ == '__main__':
    # --- Example Usage ---

    # NOTE: Replace this with your actual data loading mechanism.
    # This example uses a simple dummy data generator for demonstration.
    def load_dummy_data(size=1000):
        data = []
        for i in range(size):
            data.append({
                'ticker': f'TICKER{i}',
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


    # 1. Load your historical data
    df_historical = load_dummy_data(size=2000)

    # 2. Instantiate the model
    model_instance = StockPredictionModel()

    # 3. Preprocess the data and split into training/testing sets
    df_preprocessed = model_instance.preprocess_data(df_historical)
    df_train, df_test = train_test_split(df_preprocessed, test_size=0.2, random_state=42)

    # 4. Train the model
    model_instance.train(df_train)

    # 5. Evaluate the model's performance
    model_instance.evaluate(df_test)

    # 6. Save the trained model for later use
    joblib.dump(model_instance, 'stock_predictor.joblib')
    logging.info("Model saved as 'stock_predictor.joblib'")

    # --- Daily Prediction Use Case ---
    # 7. Load today's new data (to make a prediction)
    df_today_raw = load_dummy_data(size=50)
    df_today_preprocessed = model_instance.preprocess_data(df_today_raw)

    # 8. Use the trained model to get daily recommendations
    top_picks = model_instance.predict_top_tickers(df_today_preprocessed, num_tickers=3)
    print("\nTop picks for today:")
    print(top_picks[['ticker', 'prediction_score']])