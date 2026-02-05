import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.graph_loader import GraphLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_batches():
    logger.info("Starting Graph Ingestion from S3 Batches...")
    loader = GraphLoader()
    
    # We expect up to 200 batches (1M records / 5000 per batch)
    # Checking up to 20 to be safe for micro-batch (1000 records total = 20 batches of 50)
    
    base_uri = "s3://rawdata-aravind/stock_news/"
    
    for i in range(20):
        filename = f"fnspid_batch_{i}.parquet"
        s3_path = f"{base_uri}{filename}"
        
        try:
            logger.info(f"Attempting to load {filename}...")
            # We assume load_from_s3 handles "file not found" gracefully or throws
            loader.load_from_s3(s3_path, limit=None) # Start with no limit to load all
        except Exception as e:
            # If standard S3 file not found error, we might stop
            if "NoSuchKey" in str(e) or "FileNotFound" in str(e):
                 logger.info(f"Batch {filename} not found. Stopping.")
                 break
            logger.error(f"Failed to load batch {i}: {e}")

    logger.info("Graph Ingestion Complete.")

if __name__ == "__main__":
    ingest_batches()
