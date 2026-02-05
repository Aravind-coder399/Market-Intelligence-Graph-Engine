import pandas as pd
import os
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FNSPIDLoader:
    """
    Loads, filters, and stores FNSPID data to S3.
    """
    
    DEFAULT_TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
        "META", "NVDA", "JPM", "V", "JNJ"
    ]

    def __init__(self, s3_bucket_uri: str = "s3://rawdata-aravind/stock_news/"):
        self.s3_bucket_uri = s3_bucket_uri
        # Ensure trailing slash
        if not self.s3_bucket_uri.endswith('/'):
            self.s3_bucket_uri += '/'

    def load_and_filter(self, file_path: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Loads the FNSPID CSV, filters by ticker, and returns a DataFrame.
        Expected CSV columns in FNSPID often involve: 'Stock', 'Date', 'Headline', etc.
        We will normalize them to: ticker, date, headline, source.
        """
        target_tickers = tickers or self.DEFAULT_TICKERS
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()

        try:
            logger.info(f"Loading data from {file_path}...")
            # Load only necessary columns if possible to save memory, 
            # but FNSPID structure varies. We'll load and rename.
            df = pd.read_csv(file_path)
            
            # FNSPID typically has 'Stock' column for ticker
            if 'Stock' in df.columns:
                df.rename(columns={'Stock': 'ticker'}, inplace=True)
            elif 'ticker' not in df.columns:
                logger.error("Could not find 'Stock' or 'ticker' column in CSV.")
                return pd.DataFrame()
                
            # Filter
            logger.info(f"Filtering for tickers: {target_tickers}")
            df_filtered = df[df['ticker'].isin(target_tickers)].copy()
            
            logger.info(f"Filtered down to {len(df_filtered)} rows from {len(df)}.")
            return df_filtered

        except Exception as e:
            logger.error(f"Error loading FNSPID data: {e}")
            return pd.DataFrame()

    def save_to_s3(self, df: pd.DataFrame, filename: str = "filtered_fnspid.parquet"):
        """
        Saves the DataFrame to S3 as Parquet.
        """
        if df.empty:
            logger.warning("DataFrame is empty. Nothing to save.")
            return

        s3_path = f"{self.s3_bucket_uri}{filename}"
        
        try:
            logger.info(f"Saving to {s3_path}...")
            # Requires s3fs and pyarrow installed
            df.to_parquet(s3_path, index=False)
            logger.info("Save successful.")
        except Exception as e:
            logger.error(f"Failed to save to S3: {e}")
            # Check for AWS credentials hint
            if "NoCredentialsError" in str(e):
                logger.error("AWS Credentials not found. Please configure AWS CLI or env vars.")

if __name__ == "__main__":
    # Test execution
    # Create a dummy CSV for testing if real one doesn't exist
    dummy_path = "dummy_fnspid.csv"
    if not os.path.exists(dummy_path):
        data = {
            "Stock": ["AAPL", "MSFT", "ZZZZ", "TSLA"],
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "Headline": ["Apple launches thing", "Microsoft AI", "Sleepy stock", "Tesla car"],
            "Source": ["Reuters", "Bloomberg", "Unknown", "CNBC"]
        }
        pd.DataFrame(data).to_csv(dummy_path, index=False)
        print(f"Created temporary test file: {dummy_path}")

    loader = FNSPIDLoader()
    df = loader.load_and_filter(dummy_path)
    print("Preview:\n", df.head())
    
    # Uncomment to test S3 write if credentials exist
    # loader.save_to_s3(df)
    
    # Clean up
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
