import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph.falkor_client import FalkorClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    client = FalkorClient()
    logger.info("Dropping existing graph 'market_intelligence'...")
    try:
        # Note: The python client usually has a delete_graph method or we execute generic query
        # But FalkorDB-py doesn't always expose 'DROP GRAPH'. 
        # We can delete all nodes to clear it.
        client.query("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared successfully.")
    except Exception as e:
        logger.error(f"Error clearing graph: {e}")
