import logging
import json
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from graph.falkor_client import FalkorClient
from agents.state import AgentState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearcherAgent:
    """
    Finds historical news events similar to the input news item using Vector Search.
    """
    def __init__(self):
        # Initialize embedding model (all-MiniLM-L6-v2 is standard and efficient)
        logger.info("Initializing ResearcherAgent and loading embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.falkor = FalkorClient()
        
    def run(self, state: AgentState) -> AgentState:
        """
        Executes the research step.
        """
        news_item = state.get("news_item")
        if not news_item:
            logger.error("No news item found in state.")
            state["errors"] = state.get("errors", []) + ["No news item found."]
            return state

        headline = news_item.get("headline", "")
        logger.info(f"Researching similar events for: '{headline}'")

        try:
            # 1. Generate Embedding
            embedding = self.encoder.encode(headline).tolist()
            
            # Query FalkorDB using Vector Search
            # Validated Syntax: db.idx.vector.queryNodes(index_name, attribute, n, vecf32(vector_param))
            
            query = """
            CALL db.idx.vector.queryNodes('NewsEvent', 'embedding', 5, vecf32($vec)) 
            YIELD node, score
            RETURN node.headline, node.date, node.ticker,
                   node.day_1_return, node.day_7_return, node.day_30_return, 
                   score
            """
            
            # Ensure embedding is a flat list of floats (vecf32 handles the cast)
            params = {'vec': embedding}
            
            results = self.falkor.query(query, params)
            
            similar_events = []
            for row in results:
                # Row format depends on client, usually [col1, col2...]
                # Assuming standard falkordb-py handling where result is a list of lists or similar
                # Check column headers from Result object if possible, but here we assume order.
                
                event = {
                    "headline": row[0],
                    "date": row[1],
                    "ticker": row[2],
                    "day_1_return": row[3],
                    "day_7_return": row[4],
                    "day_30_return": row[5],
                    "similarity_score": row[6]
                }
                similar_events.append(event)
                
            logger.info(f"Found {len(similar_events)} similar events.")
            state["similar_events"] = similar_events

        except Exception as e:
            logger.error(f"Error in ResearcherAgent: {e}")
            state["errors"] = state.get("errors", []) + [str(e)]
            
        return state
