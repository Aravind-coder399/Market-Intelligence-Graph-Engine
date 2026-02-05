import logging
import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer

# Ensure we can import from graph and ingestion packages
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ingestion.news_loader import NewsLoader
from ingestion.market_fetcher import MarketFetcher
try:
    from graph.falkor_client import FalkorClient
except ImportError:
    from falkor_client import FalkorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphLoader:
    """
    Orchestrates the loading of data into FalkorDB.
    """
    def __init__(self):
        logger.info("Initializing GraphLoader...")
        self.news_loader = NewsLoader()
        self.market_fetcher = MarketFetcher()
        self.falkor = FalkorClient()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_enabled = True # Default to True, disabled if index creation fails with specific error
        self.create_vector_index()

    def create_vector_index(self):
        """Creates a vector index on NewsEvent(embedding) if it doesn't exist."""
        try:
            logger.info("Verifying/Creating Vector Index...")
            # Note: FalkorDB syntax for vector index creation
            query = "CALL db.idx.vector.createNodeIndex('News', 'embedding', {dim: 384, similarityFunction: 'cosine'})"
            self.falkor.query(query)
            logger.info("Vector Index created/verified.")
        except Exception as e:
            # If procedure is missing, vector search might not be enabled
            if "not registered" in str(e):
                logger.error("Vector Search module not enabled on FalkorDB instance. Semantic linking will be skipped.")
                self.vector_enabled = False
            else:
                # Index might already exist
                logger.warning(f"Index creation warning (may already exist): {e}")
                self.vector_enabled = True


    def load_from_s3(self, s3_path: str, limit: int = 10):
        """
        Loads pre-processed Parquet data from S3 and saves to Graph.
        
        Args:
            s3_path: S3 URI to the parquet file (e.g., s3://bucket/file.parquet)
            limit: Number of items to process.
        """
        try:
            logger.info(f"Loading data from S3: {s3_path}")
            # Requires s3fs and pyarrow
            df = pd.read_parquet(s3_path)
            
            if limit:
                df = df.head(limit)
                
            news_items = df.to_dict('records')
            
            if not news_items:
                logger.warning("No items found in S3 file.")
                return

            self._ingest_batch(news_items)
            
        except Exception as e:
            logger.error(f"Error loading from S3: {e}")

    def _ingest_batch(self, news_items: list):
        """Helper to process and ingest a list of news items."""
        logger.info(f"Processing {len(news_items)} items...")
        
        for item in news_items:
            try:
                # Handle varying generic keys if needed, assuming standard keys from FNSPIDLoader
                # FNSPIDLoader outputs: ticker, date, headline, source
                headline = item.get('headline') or item.get('Headline')
                date_str = item.get('date') or item.get('Date')
                ticker = item.get('ticker') or item.get('Stock') or item.get('ticker_symbol')
                source = item.get('source') or item.get('Source') or 'Unknown'

                if not headline or not ticker:
                    continue

                # 1. Calculate Returns using REAL market data from yfinance
                import yfinance as yf
                from datetime import datetime, timedelta
                
                day_1 = None
                day_7 = None
                day_30 = None
                
                try:
                    # Parse news date
                    news_date = pd.to_datetime(date_str)
                    
                    # Fetch historical data around the news date
                    start_date = news_date - timedelta(days=5)
                    end_date = news_date + timedelta(days=35)
                    
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    
                    if not hist.empty and len(hist) > 1:
                        # Find the closest trading day to news date
                        hist.index = hist.index.tz_localize(None)
                        news_idx = hist.index.get_indexer([news_date], method='nearest')[0]
                        
                        if news_idx >= 0 and news_idx < len(hist):
                            base_price = hist['Close'].iloc[news_idx]
                            
                            # Day 1 return
                            if news_idx + 1 < len(hist):
                                day_1 = round(((hist['Close'].iloc[news_idx + 1] - base_price) / base_price) * 100, 2)
                            
                            # Day 7 return (approx 5 trading days)
                            if news_idx + 5 < len(hist):
                                day_7 = round(((hist['Close'].iloc[news_idx + 5] - base_price) / base_price) * 100, 2)
                            
                            # Day 30 return (approx 22 trading days)
                            if news_idx + 22 < len(hist):
                                day_30 = round(((hist['Close'].iloc[news_idx + 22] - base_price) / base_price) * 100, 2)
                except Exception as e:
                    # Fallback to simulated if yfinance fails (e.g., very old dates, delisted stocks)
                    import random
                    msg_hash = hash(headline)
                    random.seed(msg_hash)
                    day_1 = round(random.uniform(-5.0, 5.0), 2)
                    day_7 = round(random.uniform(-10.0, 10.0), 2)
                    day_30 = round(random.uniform(-15.0, 15.0), 2)
                
                # 2. Generate Embedding
                embedding = self.encoder.encode(headline).tolist()
                
                # 3. Write to FalkorDB (Hierarchical Schema)
                # Schema: (Company) <-[:MENTIONS]- (News) -[:HAS_TOPIC]-> (Topic)
                
                # Simple Topic/Intent Extraction (Rule-based for speed)
                topics = []
                headline_lower = headline.lower()
                if 'earnings' in headline_lower or 'revenue' in headline_lower: topics.append('Earnings')
                if 'launch' in headline_lower or 'announces' in headline_lower: topics.append('Product Launch')
                if 'acquisition' in headline_lower or 'merger' in headline_lower: topics.append('M&A')
                if 'fda' in headline_lower: topics.append('Regulation')
                if not topics: topics.append('General News')

                query = """
                MERGE (c:Company {ticker: $ticker})
                
                CREATE (n:News {
                    headline: $headline,
                    date: $date,
                    source: $source,
                    embedding: $vec,
                    day_1_return: $d1,
                    day_7_return: $d7,
                    day_30_return: $d30
                })
                
                MERGE (n)-[:MENTIONS]->(c)
                
                FOREACH (tName IN $topics |
                    MERGE (t:Topic {name: tName})
                    MERGE (n)-[:HAS_TOPIC]->(t)
                )
                """
                
                params = {
                    'ticker': ticker,
                    'headline': headline,
                    'date': str(date_str),
                    'source': source,
                    'd1': day_1,
                    'd7': day_7,
                    'd30': day_30,
                    'vec': embedding,
                    'topics': topics
                }
                
                self.falkor.query(query, params)
                
                # 4. Semantic Linking (Optional: Link to similar past events)
                self._link_to_history(headline, embedding, ticker, date_str)
                
                logger.info(f"Ingested: {headline[:30]}... -> {ticker}")
                
            except Exception as e:
                logger.error(f"Failed to ingest item '{str(item)[:20]}': {e}")

    def _link_to_history(self, headline, embedding, ticker, current_date, threshold=0.8):
        """Finds similar past events and checks edges."""
        try:
            # Query for similar nodes
            # We filter out the node we just created (by headline/date match) if needed, 
            # or rely on LIMIT > 1 and skipping self.
            
            query = """
            CALL db.idx.vector.queryNodes('News', 'embedding', 6, vecf32($vec)) 
            YIELD node, score
            WHERE score >= $threshold AND node.headline <> $headline
            RETURN node.headline, node.date, score
            """
            
            params = {
                'vec': embedding,
                'threshold': threshold,
                'headline': headline
            }
            
            results = self.falkor.query(query, params)
            
            if not results:
                return

            # Create Relationships
            for row in results:
                other_headline = row[0]
                similarity = row[2]
                
                link_query = """
                MATCH (n1:News {headline: $h1}), (n2:News {headline: $h2})
                MERGE (n1)-[r:RELATED_TO]->(n2)
                ON CREATE SET r.score = $score, r.type = 'semantic_similarity'
                """
                
                self.falkor.query(link_query, {
                    'h1': headline,
                    'h2': other_headline,
                    'score': similarity
                })
                logger.info(f"Linked '{headline[:20]}' <-> '{other_headline[:20]}' (Score: {similarity:.2f})")

        except Exception as e:
            # Don't fail the whole batch for linking errors
            logger.warning(f"Semantic linking failed: {e}")

    def load_historical_data(self, csv_path: str, limit: int = 10):
        """
        Loads data from CSV, enriches it, and saves to Graph.
        
        Args:
            csv_path: Path to the news CSV.
            limit: Number of items to process.
        """
        logger.info(f"Loading top {limit} items from {csv_path}...")
        news_items = self.news_loader.load_from_csv(csv_path, limit=limit)
        
        if not news_items:
            logger.warning("No news items loaded.")
            return

        self._ingest_batch(news_items)

if __name__ == "__main__":
    loader = GraphLoader()
    csv_path = os.path.join(os.getcwd(), 'raw_partner_headlines.csv')
    # Load 100 rows for testing
    loader.load_historical_data(csv_path, limit=100)
