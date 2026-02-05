from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dates():
    logger.info("Streaming dataset to check date ranges...")
    ds = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)
    
    count = 0
    years = {}
    tickers = set()
    
    for sample in ds:
        date_str = str(sample.get('Date') or sample.get('date'))
        ticker = sample.get('Stock_symbol') or sample.get('stock_symbol')
        
        try:
            year = date_str[:4]
            years[year] = years.get(year, 0) + 1
        except:
            pass
            
        tickers.add(ticker)
        
        count += 1
        if count >= 10000:
            break
            
    print("--- Analysis of first 10k rows ---")
    print(f"Years found: {sorted(years.keys())}")
    print(f"Sample Tickers: {list(tickers)[:10]}")

if __name__ == "__main__":
    check_dates()
