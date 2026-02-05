import logging
try:
    from graph.falkor_client import FalkorClient
except ImportError:
    from falkor_client import FalkorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_schema():
    """
    Sets up the schema for the Market Intelligence Graph.
    - Creates Vector Index for NewsEvent embeddings.
    - Creates Range Index for NewsEvent dates.
    - Creates Index for Stock tickers.
    """
    client = FalkorClient()
    
    logger.info("Setting up Graph Schema...")
    
    # 1. Vector Index for NewsEvent
    # Note: falkordb-py syntax for vector indexing might vary slightly, 
    # but generally it's via Cypher or specific index creation methods.
    # We will use the Cypher command for vector indexing provided by FalkorDB.
    # Syntax: CREATE VECTOR INDEX FOR (n:NewsEvent) ON (n.embedding) OPTIONS {dimension:384, similarityFunction:'cosine'}
    
    try:
        # Check if index exists or just try to create it (ignore if exists error logic might be needed, 
        # but pure Cypher usually errors if exists. standard practice: try-catch)
        
        # Dimensions for all-MiniLM-L6-v2 is 384
        vector_index_query = """
        CREATE VECTOR INDEX FOR (n:NewsEvent) ON (n.embedding) 
        OPTIONS {dimension: 384, similarityFunction: 'cosine'}
        """
        # Note: FalkorDB might return an error if index already exists.
        try:
            client.query(vector_index_query)
            logger.info("Created Vector Index for NewsEvent.")
        except Exception as e:
             if "Index already exists" in str(e):
                 logger.info("Vector Index for NewsEvent already exists.")
             else:
                 logger.warning(f"Could not create Vector Index (might already exist): {e}")

        # 2. Index for Stock ticker (Exact match lookup)
        stock_index_query = "CREATE INDEX FOR (s:Stock) ON (s.ticker)"
        try:
            client.query(stock_index_query)
            logger.info("Created Index for Stock ticker.")
        except Exception as e:
             # loose check for existence
             logger.warning(f"Could not create Stock Index: {e}")

        # 3. Index for NewsEvent date (Range queries)
        # Note: In FalkorDB/RedisGraph, standard indexes support range queries on numeric/string.
        date_index_query = "CREATE INDEX FOR (n:NewsEvent) ON (n.date)"
        try:
            client.query(date_index_query)
            logger.info("Created Index for NewsEvent date.")
        except Exception as e:
             logger.warning(f"Could not create NewsEvent date Index: {e}")

        logger.info("Schema setup complete.")

    except Exception as e:
        logger.error(f"Schema setup failed: {e}")
        raise

if __name__ == "__main__":
    setup_schema()
