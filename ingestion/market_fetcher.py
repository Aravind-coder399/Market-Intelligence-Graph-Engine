import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketFetcher:
    """
    Fetches stock market data using yfinance to calculate post-event returns.
    """
    
    def __init__(self):
        pass

    def get_market_impact(self, ticker: str, event_date_str: str) -> Dict[str, Any]:
        """
        Calculates the percentage change for 1, 7, and 30 days post-event.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').
            event_date_str: Date of the event in 'YYYY-MM-DD' format.
            
        Returns:
            Dictionary containing returns for day_1, day_7, and day_30.
        """
        try:
            event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {event_date_str}. Expected YYYY-MM-DD.")
            return {}

        # We need data from event_date to event_date + 35 days (buffer for weekends/holidays)
        start_date = event_date
        end_date = event_date + timedelta(days=45) 
        
        logger.info(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}")
        
        try:
            # Download data
            df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
            
            if df.empty:
                logger.warning(f"No data found for {ticker} starting {event_date_str}")
                return {}
            
            # Use 'Adj Close' if available, else 'Close'
            # yfinance structure might vary by version; generally we want the price Series
            if 'Adj Close' in df.columns:
                prices = df['Adj Close']
            elif 'Close' in df.columns:
                prices = df['Close']
            else:
                logger.error(f"Price columns not found in response for {ticker}")
                return {}

            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)

            # Find the closest trading day to event_date (or the date itself)
            # If event was on a weekend, we might take the preceding Friday or following Monday.
            # Usually "reaction" is measured from the Close of the event day (or previous close if pre-market)
            # For simplicity: Base Price is the Close price of the event date (or next trading day if holiday)
            
            # Get valid trading dates sorted
            valid_dates = df.index.sort_values()
            
            # Find base date: First trading day >= event_date
            base_date = valid_dates[valid_dates >= event_date].min()
            
            if pd.isna(base_date):
                 logger.warning(f"No trading data available on or after {event_date_str} for {ticker}")
                 return {}

            base_price_val = prices.loc[base_date]
            # yfinance sometimes returns a Series for a single scalar if MultiIndex. 
            # safe access:
            base_price = float(base_price_val.iloc[0]) if hasattr(base_price_val, 'iloc') else float(base_price_val)

            results = {
                "ticker": ticker,
                "event_date": event_date_str,
                "base_date": base_date.strftime('%Y-%m-%d'),
                "base_price": base_price,
                "returns": {}
            }

            for days in [1, 7, 30]:
                target_date_approx = base_date + timedelta(days=days)
                
                # Find the trading day closest to target (>= target)
                # If target is weekend, taking next Monday is standard for "liquidity available"
                future_dates = valid_dates[valid_dates >= target_date_approx]
                
                if future_dates.empty:
                    logger.warning(f"Not enough data to calculate {days}-day return for {ticker}")
                    results["returns"][f"day_{days}"] = None
                else:
                    actual_date = future_dates.min()
                    # Calculate ROI
                    future_price_val = prices.loc[actual_date]
                    future_price = float(future_price_val.iloc[0]) if hasattr(future_price_val, 'iloc') else float(future_price_val)
                    
                    pct_change = ((future_price - base_price) / base_price) * 100
                    results["returns"][f"day_{days}"] = round(pct_change, 2)
                    results["returns"][f"day_{days}_date"] = actual_date.strftime('%Y-%m-%d')

            return results

        except Exception as e:
            logger.error(f"Error fetching/processing data for {ticker}: {e}")
            return {}

    def process_news_list(self, news_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process a list of news items.
        Each item must have 'ticker' and 'date' keys. 'headline' is optional but good for context.
        """
        results = []
        for item in news_items:
            ticker = item.get('ticker')
            date_str = item.get('date')
            headline = item.get('headline', '')
            
            if not ticker or not date_str:
                logger.warning(f"Skipping item due to missing ticker or date: {item}")
                continue
                
            impact = self.get_market_impact(ticker, date_str)
            if impact:
                impact['headline'] = headline
                results.append(impact)
        
        return results

if __name__ == "__main__":
    # Example usage / Test
    fetcher = MarketFetcher()
    
    # Test Data: Apple release or random event
    test_data = [
        {"ticker": "AAPL", "date": "2023-09-12", "headline": "iPhone 15 Launch Event"},
        {"ticker": "NVDA", "date": "2024-02-21", "headline": "NVIDIA Q4 Earnings Report"}
    ]
    
    print("Running MarketFetcher Test...")
    results = fetcher.process_news_list(test_data)
    
    for res in results:
        print(f"\n--- Result for {res['ticker']} ---")
        print(f"Event: {res['headline']} on {res['event_date']}")
        print(f"Base Price: {res['base_price']} ({res['base_date']})")
        print(f"Returns: {res['returns']}")
