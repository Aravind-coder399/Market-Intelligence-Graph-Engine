import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def inspect_s3_file(bucket_uri, filename):
    s3_path = f"{bucket_uri}{filename}"
    logger.info(f"Reading from {s3_path}...")
    
    try:
        # Read parquet directly from S3
        df = pd.read_parquet(s3_path, storage_options={"anon": False})
        
        print("\n--- First 10 Rows ---")
        print(df.head(10).to_string(index=False))
        print("\n--- Data Info ---")
        print(df.info())
        print(f"\nTotal Rows in this batch: {len(df)}")
        
    except Exception as e:
        logger.error(f"Failed to read S3 file: {e}")

if __name__ == "__main__":
    # Inspect the first batch
    inspect_s3_file("s3://rawdata-aravind/stock_news/", "fnspid_batch_0.parquet")
