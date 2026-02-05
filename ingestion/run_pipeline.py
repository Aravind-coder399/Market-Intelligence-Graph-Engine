import os
import sys
import logging
import pandas as pd

# Ensure we can import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from ingestion.fnspid_loader import FNSPIDLoader
import s3fs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(years_back=10):
    """
    Streams FNSPID data from Hugging Face, filters for top tickers,
    and uploads to S3 in batches.
    """
    logger.info("Initializing Ingestion Pipeline...")
    
    # 1. Configuration
    dataset_name = "Zihan1004/FNSPID"
    # S3 bucket from FNSPIDLoader default or overridden
    loader = FNSPIDLoader() 
    s3_base_path = loader.s3_bucket_uri
    
    target_tickers = loader.DEFAULT_TICKERS
    logger.info(f"Targeting Tickers: {target_tickers}")
    
    # 2. Stream Dataset (Streaming prevents downloading 20GB+)
    logger.info(f"Streaming dataset {dataset_name}...")
    try:
        # Check available splits, usually 'train'
        ds = load_dataset(dataset_name, split='train', streaming=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 3. Process in Batches
    TARGET_PER_COMPANY = 100
    BATCH_LIMIT = TARGET_PER_COMPANY * 10
    
    batch_size = 50 # Smaller batch size for faster feedback
    current_batch = []
    batch_idx = 0
    total_processed = 0
    
    # Target Tickers
    target_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
    ticker_counts = {t: 0 for t in target_tickers}
    
    logger.info(f"Filtering for: {target_tickers}")
    logger.info(f"Targeting up to {TARGET_PER_COMPANY} records per company...")

    for sample in ds:
        try:
            ticker = sample.get('Stock_symbol') or sample.get('stock_symbol')
            
            # 1. Ticker Filter
            if ticker not in target_tickers:
                continue

            # Check if we have enough for this ticker
            if ticker_counts[ticker] >= TARGET_PER_COMPANY:
                continue

            # 2. Year Filter (Removed to capture all available history)
            pass

            # Normalize keys
            headline = sample.get('Article_title') or sample.get('title')
            date_str = str(sample.get('Date') or sample.get('date'))
            
            # Basic validation
            if not headline:
                continue

            # Normalize for our pipeline
            row = {
                'ticker': ticker,
                'date': date_str,
                'headline': headline,
                'source': 'FNSPID_HF_Targeted',
            }
            current_batch.append(row)
            
            ticker_counts[ticker] += 1
            total_processed += 1
            
            if total_processed % 1000 == 0:
                 logger.info(f"Processed: {total_processed}. Counts: {ticker_counts}")
            
            # Save Batch
            if len(current_batch) >= batch_size:
                _save_batch(current_batch, batch_idx, loader)
                batch_idx += 1
                current_batch = []
            
            # Stop if all targets are filled
            if all(c >= TARGET_PER_COMPANY for c in ticker_counts.values()):
                logger.info("All target tickers reached 100k limit.")
                break
                
        except Exception as e:
            continue

    # Final Batch
    if current_batch:
        _save_batch(current_batch, batch_idx, loader)

    logger.info("Pipeline Complete.")

def _save_batch(data: list, index: int, loader: FNSPIDLoader):
    """Saves a list of dicts to S3 via dataframe."""
    logger.info(f"Saving Batch {index} with {len(data)} records...")
    df = pd.DataFrame(data)
    filename = f"fnspid_batch_{index}.parquet"
    loader.save_to_s3(df, filename)
    logger.info(f"Batch {index} saved to {filename}")

if __name__ == "__main__":
    # Note: Requires `pip install datasets`
    run_pipeline()
