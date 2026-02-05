import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph.falkor_client import FalkorClient

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def inspect_graph_structure():
    client = FalkorClient()
    logger.info("Querying Graph Structure...")
    
    # Query: Find a Company, its linked News, and their Topics
    # We limit to 1 Company and 3 News items for readability
    query = """
    MATCH (c:Company)<-[:MENTIONS]-(n:News)
    OPTIONAL MATCH (n)-[:HAS_TOPIC]->(t:Topic)
    WITH c, n, collect(t.name) as topics
    ORDER BY n.date DESC
    LIMIT 3
    RETURN c.ticker, n.headline, n.date, n.day_1_return, topics
    """
    
    try:
        results = client.query(query)
        
        if not results:
            print("\nNo data found in graph yet. Ingestion might still be initializing.")
            return

        current_ticker = None
        for row in results:
            ticker = row[0]
            headline = row[1]
            date = row[2]
            d1_return = row[3]
            topics = row[4]
            
            if ticker != current_ticker:
                print(f"\n[PARENT NODE] Company: {ticker}")
                print("=" * 40)
                current_ticker = ticker
            
            print(f"  └── [CHILD NODE] News: {headline[:60]}...")
            print(f"      ├── Date: {date}")
            print(f"      ├── Market Impact (1d): {d1_return if d1_return else 'N/A'}")
            print(f"      └── [CHILD NODE] Topics: {topics if topics else 'None'}")
            print("")
            
    except Exception as e:
        logger.error(f"Query failed: {e}")

if __name__ == "__main__":
    inspect_graph_structure()
