import os
import logging
from falkordb import FalkorDB
from dotenv import load_dotenv
from typing import Any, List, Dict, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FalkorClient:
    _instance = None
    _client = None
    _graph = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FalkorClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._connect()

    def _connect(self):
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", 6379))
        username = os.getenv("FALKORDB_USERNAME", None)
        password = os.getenv("FALKORDB_PASSWORD", None)
        
        logger.info(f"Connecting to FalkorDB at {host}:{port} as {username}...")
        try:
            self._client = FalkorDB(host=host, port=port, username=username, password=password)
            # Create or select the graph
            self._graph = self._client.select_graph("market_intelligence")
            logger.info("Successfully connected to FalkorDB graph 'market_intelligence'.")
        except Exception as e:
            logger.error(f"Failed to connect to FalkorDB: {e}")
            raise

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Executes a Cypher query against the graph.
        
        Args:
            query: The Cypher query string.
            params: Dictionary of parameters for the query.
            
        Returns:
            List of results from the query.
        """
        if not self._graph:
            self._connect()
            
        try:
            # logger.debug(f"Executing Query: {query} | Params: {params}")
            result = self._graph.query(query, params)
            return result.result_set
        except Exception as e:
            logger.error(f"Query Execution Error: {e}")
            logger.error(f"Failed Query: {query}")
            raise

    def get_graph(self):
        """Returns the raw graph object if needed for advanced operations."""
        if not self._graph:
            self._connect()
        return self._graph
