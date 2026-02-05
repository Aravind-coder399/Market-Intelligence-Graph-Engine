import pandas as pd
import requests
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsLoader:
    """
    Loads news from local CSV (historical) or Finnhub API (live).
    """

    def __init__(self):
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.finnhub_base_url = "https://finnhub.io/api/v1/news?category=general&token="

    def load_from_csv(self, file_path: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Loads news from a local CSV file.
        Expected columns: headline, url, publisher, date, stock
        
        Args:
            file_path: Absolute path to the CSV file.
            limit: Number of rows to load (default 10 for testing).
            
        Returns:
            List of dictionaries with keys: headline, date, ticker, source.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []

        try:
            logger.info(f"Loading top {limit} rows from {file_path}")
            df = pd.read_csv(file_path, nrows=limit)
            
            # Map columns to standard format
            # CSV: ,headline,url,publisher,date,stock
            # Output: headline, date, ticker, source
            
            news_items = []
            for _, row in df.iterrows():
                try:
                    # Parse date - usually in 'YYYY-MM-DD HH:MM:SS' or similar. 
                    # We'll keep it as string YYYY-MM-DD for consistency with market_fetcher
                    date_val = str(row['date'])
                    
                    # Basic date parsing/cleaning if needed. Assuming the format is parseable by pd.to_datetime
                    dt = pd.to_datetime(date_val)
                    date_str = dt.strftime('%Y-%m-%d')
                    
                    item = {
                        "headline": row['headline'],
                        "date": date_str,
                        "ticker": row['stock'],
                        "source": row.get('publisher', 'Unknown')
                    }
                    news_items.append(item)
                except Exception as e:
                    logger.warning(f"Error parsing row: {row}. Error: {e}")
                    continue
            
            return news_items

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return []

    def fetch_live_news(self) -> List[Dict[str, str]]:
        """
        Fetches live general news from Finnhub API.
        
        Returns:
            List of dictionaries with keys: headline, date, ticker, source.
        """
        if not self.finnhub_api_key:
            logger.error("FINNHUB_API_KEY not found in environment variables.")
            return []

        url = f"{self.finnhub_base_url}{self.finnhub_api_key}"
        
        try:
            logger.info("Fetching live news from Finnhub...")
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            news_items = []
            for item in data:
                # Finnhub 'datetime' is unix timestamp
                ts = item.get('datetime', 0)
                date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                
                # Finnhub general news might not have a ticker, or it might be in 'related' field
                # For this specific endpoint (general news), 'category' is used. 
                # If we want specific ticker news, we'd use a different endpoint.
                # However, the user prompt said "live news enters the pipeline... identifying most relevant historical precedents".
                # The user's CSV has 'stock' column. Live data MIGHT not have it if it's 'general' news.
                # If we use company-news endpoint, we need a ticker symbol.
                # The prompt provided: https://finnhub.io/api/v1/news?category=general&token=
                # This endpoint returns general market news, often without a specific single ticker.
                # But for the graph correlation, we need to know WHICH stock is affected to check returns.
                # Agent 1 (Classifier) is supposed to "Extract Ticker symbols and Sentiment".
                # So here we just load the text.
                
                # We will leave 'ticker' empty if not provided, allowing Agent 1 to fill it later if needed.
                # OR if the user intends this strictly for the ingestion phase, we might need a different endpoint.
                # Given instructions: "Agent 1 (Classifier): Extracts Ticker symbols..."
                # implies we just ingest text here.
                
                news_item = {
                    "headline": item.get('headline', ''),
                    "date": date_str,
                    "ticker": "", # To be extracted by Agent 1
                    "source": item.get('source', 'Finnhub')
                }
                news_items.append(news_item)
                
            return news_items

        except Exception as e:
            logger.error(f"Error fetching live news: {e}")
            return []

if __name__ == "__main__":
    # Test execution
    loader = NewsLoader()
    
    # 1. Test CSV
    csv_path = os.path.join(os.getcwd(), 'raw_partner_headlines.csv')
    print(f"Testing CSV load from {csv_path}...")
    csv_news = loader.load_from_csv(csv_path, limit=5)
    for n in csv_news:
        print(n)
        
    # 2. Test API
    print("\nTesting Live API...")
    live_news = loader.fetch_live_news()
    # Print first 3 to avoid clutter
    for n in live_news[:3]:
        print(n)
