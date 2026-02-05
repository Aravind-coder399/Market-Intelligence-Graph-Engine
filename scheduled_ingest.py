"""
Scheduled Ingestion Script for Market Intelligence
Run via cron or Windows Task Scheduler to keep knowledge graph fresh.

Usage:
    # One-time run
    python scheduled_ingest.py
    
    # Schedule with cron (Linux):
    # 0 6 * * * cd /path/to/project && python scheduled_ingest.py >> logs/ingest.log 2>&1
    
    # Schedule with Task Scheduler (Windows):
    # Create task to run: python C:\path\to\project\scheduled_ingest.py
"""
import os
import sys
import logging
from datetime import datetime, timedelta

# Setup path
sys.path.append(os.path.dirname(__file__))

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'ingest_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_scheduled_ingestion():
    """Run scheduled ingestion of new news data."""
    
    logger.info("=" * 50)
    logger.info("SCHEDULED INGESTION STARTED")
    logger.info("=" * 50)
    
    try:
        # 1. Load from S3 (recent batches only)
        logger.info("Step 1: Loading recent data from S3...")
        from ingestion.fnspid_loader import FNSPIDLoader
        
        loader = FNSPIDLoader()
        
        # Get records from last 7 days (if available)
        target_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
        
        all_records = []
        for ticker in target_tickers:
            try:
                records = loader.load_for_ticker(ticker, limit=50)  # 50 recent per company
                all_records.extend(records)
                logger.info(f"  {ticker}: {len(records)} records")
            except Exception as e:
                logger.warning(f"  {ticker}: Failed - {e}")
        
        logger.info(f"Total records loaded: {len(all_records)}")
        
        # 2. Ingest into Graph
        logger.info("Step 2: Ingesting into Knowledge Graph...")
        from ingestion.graph_loader import GraphLoader
        
        graph_loader = GraphLoader()
        graph_loader.load_news(all_records)
        
        logger.info("Step 3: Ingestion complete!")
        
        # 3. Log statistics
        from graph.falkor_client import FalkorClient
        client = FalkorClient()
        
        news_count = client.query("MATCH (n:News) RETURN count(n)")[0][0]
        company_count = client.query("MATCH (c:Company) RETURN count(c)")[0][0]
        
        logger.info(f"Graph Statistics:")
        logger.info(f"  - Total News: {news_count}")
        logger.info(f"  - Total Companies: {company_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return False
    
    finally:
        logger.info("=" * 50)
        logger.info("SCHEDULED INGESTION ENDED")
        logger.info("=" * 50)


if __name__ == "__main__":
    success = run_scheduled_ingestion()
    sys.exit(0 if success else 1)
